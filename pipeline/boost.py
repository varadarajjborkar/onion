
import numpy as np

from pathlib import Path

import frame_features as ff

from ingest import load_json, save_json
from onion import flatten_words
from video_processor import clock


# -----------------------------------------
# CONFIGURATION
# Every number here is a knob to tune per kind of video
# -----------------------------------------

WINDOW_S = 2.0

# Colour histogram distance between neighbouring samples that counts as a cut
CUT_HIST = 0.35

# Rules that give the first labels (the teacher). Replace them by your own labels in labels.json.
SPEED_MOVE = 0.02       # picture slides this many widths per second: the camera is moving
JERK_SHAKY = 1.5        # pixels of change in the slide between samples: the camera is wobbling
RESID_SUBJECT = 0.025   # mean change left after undoing the slide: something moves inside the picture

LABELS = ["static", "subject_motion", "camera_move", "camera_shaky", "cut"]

# Frame features of a window, then what onion said about the speech in it
FRAME_COLUMNS = ["cuts", "diff_mean", "diff_max", "hist_max", "speed", "jerk", "resid", "sharp", "bright_range"]
ONION_COLUMNS = ["seg_weight", "seg_pointing", "cue_score", "cue_lexical", "cue_llm", "speech_rate"]
COLUMNS = FRAME_COLUMNS + ONION_COLUMNS

# How the segment focus is built (sums to 1): onion weight, and how much is going on in the picture
FOCUS_WEIGHTS = {"onion": 0.6, "activity": 0.4}

# Segment decision tree
PLAN_POINTING = 0.5          # onion pointing score that counts as pointing
PLAN_FOLLOW_FRACTION = 0.15  # share of cut or camera move that makes the target hard to find
PLAN_SHAKY_FRACTION = 0.3
PLAN_CUT_FRACTION = 0.1

# Segments are longer now, so each gets more keyframes; the strongest cues are the first to get one
MIN_KEYFRAMES = 4
MAX_KEYFRAMES = 10
MAX_CUE_KEYFRAMES = 4
KEYFRAME_MIN_GAP_S = 0.75


# -----------------------------------------
# WINDOW FEATURES
# -----------------------------------------

def window_table(samples, onion_result, words, duration):

    t = samples["t"]

    cues = [c for s in onion_result["segments"] for c in s["cues"]]

    rows, spans = [], []

    for t0 in np.arange(0.0, duration, WINDOW_S):

        t1 = min(t0 + WINDOW_S, duration)

        pick = (t >= t0) & (t < t1)

        if pick.sum() < 2:
            continue

        shift = np.stack([samples["dx"][pick], samples["dy"][pick]], axis=1)
        cut = samples["hist"][pick] > CUT_HIST

        # Slides during a cut are noise; do not count them as camera movement
        shift[cut] = 0.0

        speed = np.hypot(shift[:, 0], shift[:, 1]).mean() * ff.SAMPLE_FPS / ff.WORK_SIZE[0]
        jerk = np.hypot(*np.diff(shift, axis=0).T).mean()

        mid = (t0 + t1) / 2
        seg = next((s for s in onion_result["segments"] if s["start"] <= mid <= s["end"]), None)

        near = [c for c in cues if c["start"] < t1 and c["end"] > t0]
        best = max(near, key=lambda c: c["score"], default=None)

        row = {
            "cuts": float(cut.sum()),
            "diff_mean": float(samples["diff"][pick].mean()),
            "diff_max": float(samples["diff"][pick].max()),
            "hist_max": float(samples["hist"][pick].max()),
            "speed": float(speed),
            "jerk": float(jerk),
            "resid": float(np.where(cut, 0.0, samples["resid"][pick]).mean()),
            "sharp": float(samples["sharp"][pick].mean()),
            "bright_range": float(np.ptp(samples["bright"][pick])),
            "seg_weight": seg["weight"] if seg else 0.0,
            "seg_pointing": seg["pointing"] if seg else 0.0,
            "cue_score": best["score"] if best else 0.0,
            "cue_lexical": best["vector"]["lexical"] if best else 0.0,
            "cue_llm": best["vector"]["llm"] if best else 0.0,
            "speech_rate": sum(1 for w in words if t0 <= w["start"] < t1) / (t1 - t0)
        }

        rows.append(row)
        spans.append((float(t0), float(t1)))

    return rows, spans


def teacher_label(row):

    if row["cuts"] > 0:
        return "cut"

    if row["jerk"] > JERK_SHAKY:
        return "camera_shaky"

    if row["speed"] > SPEED_MOVE:
        return "camera_move"

    if row["resid"] > RESID_SUBJECT:
        return "subject_motion"

    return "static"


def apply_human_labels(labels, spans, path):

    # labels.json: [{"start": 12.0, "end": 16.0, "label": "camera_move"}, ...] overrides the teacher
    if not Path(path).exists():
        return labels, 0

    given = load_json(Path(path))

    count = 0

    for i, (t0, t1) in enumerate(spans):
        for g in given:
            if g["label"] in LABELS and g["start"] < t1 and g["end"] > t0:
                labels[i] = g["label"]
                count += 1
                break

    return labels, count


def fit(rows, labels):

    X = np.array([[r[c] for c in COLUMNS] for r in rows], dtype=np.float32)

    present = sorted(set(labels))

    # One class only (a very uniform video): nothing to learn, the labels are the answer
    if len(present) < 2:
        return None, X, present

    import lightgbm as lgb

    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=8, min_child_samples=5, verbose=-1)
    model.fit(X, labels)

    return model, X, present


# -----------------------------------------
# SEGMENT PLAN
# -----------------------------------------

def choose_mode(frac, pointing):

    disturbed = frac["cut"] + frac["camera_move"]

    if pointing >= PLAN_POINTING and disturbed >= PLAN_FOLLOW_FRACTION:
        return "follow_to_target"

    if pointing >= PLAN_POINTING:
        return "mask_and_zoom"

    if frac["camera_shaky"] >= PLAN_SHAKY_FRACTION:
        return "stabilize_sample"

    if frac["cut"] >= PLAN_CUT_FRACTION:
        return "per_shot"

    return "light"


def sharpest_near(samples, t, tol):

    t_all = samples["t"]
    pick = np.where((t_all >= t - tol) & (t_all <= t + tol))[0]

    if not len(pick):
        pick = [int(np.abs(t_all - t).argmin())]

    return float(t_all[pick[int(np.argmax(samples["sharp"][pick]))]])


def keyframe_targets(mode, seg, cuts):

    start, end = seg["start"], seg["end"]
    mid = (start + end) / 2

    strong = sorted(seg["cues"], key=lambda c: -c["score"])[:MAX_CUE_KEYFRAMES]

    targets = []

    if mode in ("follow_to_target", "mask_and_zoom"):

        for c in strong:
            targets.append(((c["start"] + c["end"]) / 2, "cue", 0.4))

        if mode == "follow_to_target":

            for c in strong:
                targets.append((c["end"] + 1.0, "after_cue", 0.4))

            targets += [(x + 0.5, "after_cut", 0.4) for x in cuts]

    elif mode == "per_shot":
        targets += [(x + 0.5, "after_cut", 0.4) for x in cuts]

    elif mode == "stabilize_sample":

        quarter = (end - start) / 4

        targets += [(start + quarter * (k + 0.5), "sharpest_of_quarter", quarter / 2) for k in range(4)]

    targets += [(mid, "segment_mid", 0.4), (start + 0.5, "segment_start", 0.4)]

    return targets


def pick_keyframes(samples, mode, seg, cuts, focus):

    budget = MIN_KEYFRAMES + round(focus * (MAX_KEYFRAMES - MIN_KEYFRAMES))

    chosen = []

    for t, reason, tol in keyframe_targets(mode, seg, cuts):

        t = sharpest_near(samples, min(max(t, seg["start"]), seg["end"]), tol)

        if all(abs(t - c["t"]) >= KEYFRAME_MIN_GAP_S for c in chosen):
            chosen.append({"t": round(t, 2), "reason": reason})

        if len(chosen) == budget:
            break

    return sorted(chosen, key=lambda c: c["t"])


def plan(manifest_path="processed/ingest.json", onion_path="processed/onion.json", force=False):

    manifest_path = Path(manifest_path)
    out_dir = manifest_path.parent

    manifest = load_json(manifest_path)
    onion_result = load_json(Path(onion_path))

    samples = ff.extract(manifest["video"]["path"], out_dir / "frames.npz", force=force)

    words = flatten_words(manifest["transcript"])

    rows, spans = window_table(samples, onion_result, words, manifest["video"]["duration"])

    labels = [teacher_label(r) for r in rows]
    labels, human = apply_human_labels(labels, spans, out_dir / "labels.json")

    model, X, present = fit(rows, labels)

    if model is None:
        proba = np.array([[1.0 if lab == present[0] else 0.0 for lab in LABELS] for _ in rows])
    else:
        # Columns in LABELS order, zero for a class the video never showed
        got = model.predict_proba(X)
        proba = np.zeros((len(rows), len(LABELS)))
        for k, cls in enumerate(model.classes_):
            proba[:, LABELS.index(cls)] = got[:, k]

    mids = np.array([(a + b) / 2 for a, b in spans])
    cut_times = samples["t"][samples["hist"] > CUT_HIST]

    segments = []

    for seg in onion_result["segments"]:

        inside = (mids >= seg["start"]) & (mids <= seg["end"])

        share = proba[inside].mean(axis=0) if inside.any() else np.eye(len(LABELS))[0]
        frac = {lab: float(share[k]) for k, lab in enumerate(LABELS)}

        mode = choose_mode(frac, seg["pointing"])

        focus = FOCUS_WEIGHTS["onion"] * seg["weight"] + FOCUS_WEIGHTS["activity"] * (1.0 - frac["static"])

        cuts = [float(x) for x in cut_times if seg["start"] <= x <= seg["end"]]

        segments.append({
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "mode": mode,
            "focus": round(float(focus), 3),
            "situation": {k: round(v, 3) for k, v in frac.items()},
            "cuts": [round(x, 2) for x in cuts],
            "keyframes": pick_keyframes(samples, mode, seg, cuts, focus)
        })

    windows = [
        {"start": a, "end": b, "label": lab, "predicted": LABELS[int(proba[i].argmax())]}
        for i, ((a, b), lab) in enumerate(zip(spans, labels))
    ]

    result = {
        "windows": windows,
        "model": {
            "kind": "lightgbm" if model is not None else "teacher_only",
            "labels_from": "rules" if not human else f"rules plus {human} human windows",
            "window_counts": {lab: labels.count(lab) for lab in LABELS},
            "importance": (
                {c: int(v) for c, v in sorted(zip(COLUMNS, model.feature_importances_), key=lambda x: -x[1])}
                if model is not None else {}
            )
        },
        "segments": segments
    }

    save_json(out_dir / "plan.json", result)

    return result
