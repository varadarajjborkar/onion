
import re
import sys

import numpy as np

from pathlib import Path

from ingest import load_json, save_json
from onion import POINTING_DIRECTION, POINTING_THRESHOLD


# -----------------------------------------
# CONFIGURATION
# -----------------------------------------

# Learning the direction from marks needs at least this many marked segments, and some of each answer
MIN_MARKS = 20
MIN_PER_CLASS = 5

# A learned direction must beat the current one by this much (F1) to be recommended
MIN_GAIN = 0.03

THRESHOLDS = [round(x, 2) for x in np.arange(0.20, 0.86, 0.05)]

LAYERS = list(POINTING_DIRECTION)


# -----------------------------------------
# HELPERS
# -----------------------------------------

def find_feedback(given):

    if given:
        return Path(given)

    local = Path("processed/review/feedback.json")

    if local.exists():
        return local

    found = sorted(Path.home().joinpath("Downloads").glob("feedback*.json"), key=lambda p: p.stat().st_mtime)

    if not found:
        raise SystemExit("No feedback.json found. Press Download feedback.json on the review page first.")

    return found[-1]


def f1_at(scores, y, threshold):

    pred = scores >= threshold

    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return precision, recall, f1


def best_threshold(scores, y):

    # Ties go to the middle of the best range, which is steadier than an edge
    results = [(t, *f1_at(scores, y, t)) for t in THRESHOLDS]

    top = max(r[3] for r in results)
    best = [r for r in results if r[3] >= top - 1e-9]

    return best[len(best) // 2], results


def rewrite_constant(path, pattern, replacement):

    text = path.read_text(encoding="utf-8")

    new, count = re.subn(pattern, replacement, text, flags=re.M)

    if count != 1:
        raise SystemExit(f"Could not find exactly one {pattern!r} in {path}")

    path.write_text(new, encoding="utf-8")


# -----------------------------------------
# TUNE
# -----------------------------------------

def tune(feedback_path=None, onion_path="processed/onion.json", labels_out="processed/labels.json", apply=False):

    feedback_path = find_feedback(feedback_path)
    feedback = load_json(feedback_path)

    print(f"Feedback: {feedback_path}")

    # ---- the cue direction and pointing threshold: every mark is one cue, keyed by its start time
    cues = {f"{c['start']:.2f}": c for s in load_json(Path(onion_path))["segments"] for c in s["cues"]}

    marked = [(cues[k], bool(v)) for k, v in feedback.get("pointing", {}).items() if k in cues]

    missing = len(feedback.get("pointing", {})) - len(marked)

    print(f"Cue marks: {len(marked)} ({sum(v for _, v in marked)} yes, {sum(not v for _, v in marked)} no)")

    if missing:
        print(f"{missing} marks belong to cues that no longer exist (the chunking changed since you marked); they are ignored.")

    if marked:

        y = np.array([v for _, v in marked])
        now = np.array([c["score"] for c, _ in marked])

        (thr, p, r, f), table = best_threshold(now, y)

        print("\nThreshold on the current score (precision, recall, F1):")

        for t, pp, rr, ff in table[::2]:
            print(f"  {t:.2f}  {pp:.2f}  {rr:.2f}  {ff:.2f}{'   <- best' if t == thr else ''}")

        current_f1 = f1_at(now, y, POINTING_THRESHOLD)[2]

        print(f"\nCurrent threshold {POINTING_THRESHOLD}: F1 {current_f1:.2f}. Best threshold {thr}: F1 {f:.2f}.")

        new_direction, new_threshold = None, thr

        if len(marked) >= MIN_MARKS and min(y.sum(), (~y).sum()) >= MIN_PER_CLASS:

            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_predict

            X = np.array([[c["vector"][k] for k in LAYERS] for c, _ in marked])

            model = LogisticRegression(class_weight="balanced", C=1.0)

            folds = int(min(5, y.sum(), (~y).sum()))
            held_out = cross_val_predict(model, X, y, cv=folds, method="predict_proba")[:, 1]

            held_f1 = best_threshold(held_out, y)[0][3]

            weights = np.clip(model.fit(X, y).coef_[0], 0, None)

            print(f"\nLearned direction check: F1 on held-out marks {held_f1:.2f}, against {f:.2f} for the current direction.")

            if weights.sum() > 0 and held_f1 >= f + MIN_GAIN:

                weights = weights / weights.sum()
                new_direction = {k: round(float(w), 2) for k, w in zip(LAYERS, weights)}

                new_scores = X @ np.array([new_direction[k] for k in LAYERS])
                new_threshold = best_threshold(new_scores, y)[0][0]

                print(f"Recommended direction: {new_direction}, threshold {new_threshold}")

            else:
                print("The current direction is as good, keeping it.")

        else:
            print(f"\nNot enough marks to learn the direction (need {MIN_MARKS}, with {MIN_PER_CLASS} of each answer). Only the threshold is suggested.")

        flags = [f"--set=onion.POINTING_THRESHOLD={new_threshold}"]

        if new_direction:
            flags.append("'--set=onion.POINTING_DIRECTION=" + str(new_direction).replace("'", '"') + "'")

        print("\nTry it:   python pipeline/run_onion.py " + " ".join(flags))

        if apply:

            onion_file = Path(__file__).with_name("onion.py")

            rewrite_constant(onion_file, r"^POINTING_THRESHOLD = .*$", f"POINTING_THRESHOLD = {new_threshold}")

            if new_direction:
                rewrite_constant(onion_file, r"^POINTING_DIRECTION = \{.*\}$", "POINTING_DIRECTION = " + str(new_direction).replace("'", '"'))

            print(f"Written into {onion_file.name}.")

    # ---- the situation labels for the boosting model
    given = feedback.get("situation", [])

    if given:

        out = Path(labels_out)

        kept = load_json(out) if out.exists() else []

        keys = {(g["start"], g["end"]) for g in given}
        merged = [k for k in kept if (k["start"], k["end"]) not in keys] + given

        save_json(out, merged)

        print(f"\nSituation labels: {len(given)} new, {len(merged)} in total, written to {out}")
        print("Retrain with: python pipeline/run_plan.py")

    if not marked and not given:
        print("Nothing marked yet.")


if __name__ == "__main__":

    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    tune(args[0] if args else None, apply="--apply" in sys.argv)
