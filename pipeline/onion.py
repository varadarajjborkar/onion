
import json
import math
import re

import numpy as np

from collections import Counter
from pathlib import Path

from cross_encoder import CrossEncoder
from ingest import ask, atomic_write, load_json, save_json
from video_processor import clock


# -----------------------------------------
# CONFIGURATION
# Every number here is a knob to tune per kind of video
# -----------------------------------------

# Segment borders ---------------------------------

# How strong a border must be (0 to 1) to cut there; higher gives fewer, longer segments
BORDER_THRESHOLD = 0.5

MIN_SEGMENT_S = 12.0
MAX_SEGMENT_S = 75.0

# After the first strong border, look this far ahead for an even stronger one
PEAK_LOOKAHEAD_S = 2.0

# A pause this long (seconds) is a possible border even without a full stop
PAUSE_CANDIDATE_S = 0.6

# A pause this long counts as a full pause signal
PAUSE_FULL_S = 1.0

# Words compared on each side of a border for topic and entity shift
SHIFT_WINDOW_WORDS = 80

# How the border signals are combined (sums to 1). The topic signal is ranked against the rest of the
# video, so it says "more of a shift than most places here", whatever the video. "llm" is the LLM's
# own reading of where a new part starts; without it the others are scaled up to fill its place.
BORDER_WEIGHTS = {"pause": 0.12, "sentence": 0.08, "topic": 0.25, "entity": 0.10, "llm": 0.45, "cross": 0.0}

# The cross-encoder signal reads the words before and after a border together and asks whether they belong
# to one passage. It is off (weight 0) until you give it a weight, for example
#   --set=onion.BORDER_WEIGHTS='{"pause":0.12,"sentence":0.08,"topic":0.0,"entity":0.10,"llm":0.45,"cross":0.25}'
CROSS_MODEL = "cross-encoder/stsb-TinyBERT-L-4"
CROSS_ONNX = "onnx/model.onnx"

# Words taken on each side of a border; several sizes are averaged so both a short and a long view count
CROSS_SCALES = [30, 80]

# Lines (sentences or pauses) shown to the LLM per call when it looks for new parts, and how many repeat
LLM_BORDER_LINES = 50
LLM_BORDER_OVERLAP = 10

# Pointing cues -----------------------------------

# The pointing direction: how much each layer pushes a phrase towards "something is shown here"
# (sums to 1). The score of a phrase is its layer vector projected on this direction.
POINTING_DIRECTION = {"lexical": 0.35, "entity": 0.20, "llm": 0.35, "emphasis": 0.10}

# A segment is "likely pointing" when its best cue scores at least this (0 to 1)
POINTING_THRESHOLD = 0.5

# Cues below this are not kept at all
CUE_FLOOR = 0.2

# Words after a demonstrative that name the thing pointed at ("this laptop")
LOOK_AHEAD_WORDS = 2

# How the segment weight is built (sums to 1)
SEGMENT_WEIGHTS = {"pointing": 0.50, "entity": 0.25, "emphasis": 0.25}

# Entity mentions per 10 words that count as fully entity dense
ENTITY_DENSE_PER_10 = 2.0

# Lexical layer: weak prior from closed-class words, not a final answer
DEICTIC_WORDS = {"this": 0.6, "these": 0.6, "here": 0.6, "there": 0.5, "those": 0.5, "that": 0.25}

DEICTIC_PHRASES = {
    ("as", "you", "can", "see"): 1.0,
    ("right", "here"): 1.0,
    ("over", "here"): 1.0,
    ("over", "there"): 1.0,
    ("take", "a", "look"): 0.9,
    ("look", "at"): 0.9,
    ("looking", "at"): 0.9,
    ("you", "can", "see"): 0.8,
    ("check", "out"): 0.6,
}

STOPWORDS = set(
    "a an the and or but if so of to in on at for with by from as is are was were be been it its i you he she we they "
    "me my your our their his her this that these those there here then than not no do does did have has had will "
    "would can could should just about into out up down very really also what which who whom how when where why "
    "all any some more most one two like get got going go yeah okay".split()
)

LLM_BORDER_PROMPT = (
    "Below are numbered lines of the speech in a video. Find where a new part begins: the topic changes, "
    "the speaker moves to a new point or example, or a new thing starts being shown. "
    'Reply with only a JSON list of {{"line": number, "strength": number}}, where line is the FIRST line of the new part '
    "and strength is 1 for a clear change and 0.5 for a mild one. "
    "Do not list lines that only continue the same point, even when they start a new sentence. Reply [] if there is none.\n\n"
    "What the whole video is about: {summary}\n\nLines:\n"
)

LLM_PROMPT = (
    "Below is part of the speech in a video. While speaking, the speaker may be pointing at, showing or "
    "referring to something visible on screen at that moment (for example 'look at this', "
    "'as you can see here', 'this laptop'). "
    'Reply with only a JSON list. Each item has "phrase" (copied exactly from the text, two to six words) '
    'and "score" (0 to 1, how likely something is being visually pointed at or shown while the phrase is spoken). '
    "Reply [] if nothing is. Do not include phrases that are only a figure of speech.\n\n"
    "What the whole video is about: {summary}\n\nSpeech:\n"
)


# -----------------------------------------
# HELPERS
# -----------------------------------------

def norm(text):

    return re.sub(r"[^\w']", "", text.lower())


def flatten_words(transcript):

    words = []

    for seg in transcript:

        pieces = seg.get("words") or []

        if not pieces:

            # No word timings: spread the words evenly across the segment
            tokens = seg["text"].split()
            step = (seg["end"] - seg["start"]) / max(len(tokens), 1)

            pieces = [
                {"start": seg["start"] + k * step, "end": seg["start"] + (k + 1) * step, "text": t}
                for k, t in enumerate(tokens)
            ]

        for w in pieces:
            words.append({"start": w["start"], "end": w["end"], "text": w["text"], "norm": norm(w["text"])})

    return words


def sigmoid(x):

    return 1.0 / (1.0 + math.exp(-x))


def cosine_distance(a, b):

    if not a or not b:
        return 0.0

    dot = sum(a[k] * b.get(k, 0) for k in a)
    size = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))

    return 1.0 - dot / size if size else 0.0


def bag(words):

    return Counter(w["norm"] for w in words if w["norm"] and w["norm"] not in STOPWORDS)


def entity_tokens(context):

    tokens = set()

    for entity in context.get("entities", []):
        for part in re.split(r"\W+", entity.lower()):
            if len(part) >= 3 and part not in STOPWORDS:
                tokens.add(part)

    return tokens


# -----------------------------------------
# AUDIO LAYER: how strongly each word is spoken compared with the rest of the video
# -----------------------------------------

def word_emphasis(words, stft_path):

    if not stft_path or not Path(stft_path).exists():
        return [0.5] * len(words)

    with np.load(stft_path) as data:
        spectrograms = data["spectrograms"]

    # Loudness of every 10 ms frame: mean over frequencies of the log spectrogram
    energy = spectrograms.mean(axis=1)

    hop = 0.01
    times = (np.arange(energy.shape[0])[:, None] + np.arange(energy.shape[1]) * hop).ravel()
    energy = energy.ravel()

    level = []

    for w in words:

        lo = np.searchsorted(times, w["start"])
        hi = max(np.searchsorted(times, w["end"]), lo + 1)

        level.append(energy[lo:hi].mean() if lo < len(energy) else energy[-1])

    level = np.array(level)

    z = (level - level.mean()) / max(level.std(), 1e-6)

    return [sigmoid(v) for v in z]


# -----------------------------------------
# BORDER LAYERS: where one segment ends and the next begins
# -----------------------------------------

def border_candidates(words, context):

    ents = entity_tokens(context)

    candidates = []

    for i in range(len(words) - 1):

        gap = words[i + 1]["start"] - words[i]["end"]
        sentence = words[i]["text"].rstrip('"\')').endswith((".", "?", "!"))

        if not (sentence or gap >= PAUSE_CANDIDATE_S):
            continue

        before = words[max(0, i + 1 - SHIFT_WINDOW_WORDS):i + 1]
        after = words[i + 1:i + 1 + SHIFT_WINDOW_WORDS]

        e_before = {w["norm"] for w in before if w["norm"] in ents}
        e_after = {w["norm"] for w in after if w["norm"] in ents}
        union = e_before | e_after

        candidates.append({
            "idx": i,
            "time": (words[i]["end"] + words[i + 1]["start"]) / 2,
            "signals": {
                "pause": min(max(gap, 0.0) / PAUSE_FULL_S, 1.0),
                "sentence": 1.0 if sentence else 0.0,
                "topic": cosine_distance(bag(before), bag(after)),
                "entity": 1.0 - len(e_before & e_after) / len(union) if union else 0.0
            }
        })

    if BORDER_WEIGHTS.get("cross", 0) > 0 and candidates:
        add_cross_signal(words, candidates)

    # Rank the topic shift against this video's other places: 1.0 is its biggest shift, 0.0 its smallest
    raw = np.array([c["signals"]["topic"] for c in candidates])

    if len(raw) > 1:
        ranks = raw.argsort().argsort() / (len(raw) - 1)
        for c, r in zip(candidates, ranks):
            c["signals"]["topic"] = float(r)

    return candidates


def add_cross_signal(words, candidates):

    encoder = CrossEncoder(CROSS_MODEL, CROSS_ONNX)

    pairs = []

    for c in candidates:

        i = c["idx"]

        for n in CROSS_SCALES:

            before = " ".join(w["text"] for w in words[max(0, i + 1 - n):i + 1])
            after = " ".join(w["text"] for w in words[i + 1:i + 1 + n])

            # Both orders, so which side is read first does not matter
            pairs += [(before, after), (after, before)]

    scores = encoder.score(pairs).reshape(len(candidates), len(CROSS_SCALES) * 2).mean(axis=1)

    # How much the passages belong together, ranked against this video; low belonging is a likely border
    belong = scores.argsort().argsort() / max(len(scores) - 1, 1)

    for c, b in zip(candidates, belong):
        c["signals"]["cross"] = float(1.0 - b)


def llm_border_strengths(words, candidates, summary, cache, model, save):

    # Lines are the stretches of speech between candidate borders; the LLM says which lines start a new part
    edges = [-1] + [c["idx"] for c in candidates] + [len(words) - 1]

    lines = [" ".join(w["text"] for w in words[edges[k] + 1:edges[k + 1] + 1]) for k in range(len(edges) - 1)]

    strengths = {}

    step = LLM_BORDER_LINES - LLM_BORDER_OVERLAP

    for lo in range(0, len(lines), step):

        hi = min(lo + LLM_BORDER_LINES, len(lines))
        key = f"{lo}-{hi}"

        if key not in cache:

            text = "\n".join(f"{n + 1}. {lines[n]}" for n in range(lo, hi))

            reply = ask(LLM_BORDER_PROMPT.format(summary=summary or "unknown") + text, model)
            match = re.search(r"\[.*\]", reply, re.S)

            try:
                items = json.loads(match.group(0)) if match else []
            except json.JSONDecodeError:
                items = []

            cache[key] = [
                [int(i["line"]), min(max(float(i.get("strength", 1.0)), 0.0), 1.0)]
                for i in items if isinstance(i, dict) and str(i.get("line", "")).isdigit()
            ]

            save()

        for line, strength in cache[key]:

            # Line n starts a new part, so the border is after candidate n-2 (line 1 has nothing before it)
            if 2 <= line <= len(candidates) + 1:
                strengths[line - 2] = max(strengths.get(line - 2, 0.0), strength)

        if hi == len(lines):
            break

    return strengths


def score_candidates(candidates, strengths=None):

    # Without the LLM signal the others are scaled up to fill its weight
    weights = {k: v for k, v in BORDER_WEIGHTS.items() if v > 0 and (k != "llm" or strengths is not None)}

    total = sum(weights.values())

    for k, c in enumerate(candidates):

        signals = dict(c["signals"])

        if strengths is not None:
            signals["llm"] = strengths.get(k, 0.0)

        c["signals"] = signals
        c["score"] = sum(weights[n] * signals[n] for n in weights) / total

    return candidates


def pick_borders(words, candidates):

    borders = []

    first = 0
    seg_start = words[0]["start"]
    end_time = words[-1]["end"]

    while True:

        remaining = end_time - seg_start

        # Leave at least MIN_SEGMENT_S for the tail
        window = [
            c for c in candidates
            if c["idx"] >= first
            and seg_start + MIN_SEGMENT_S <= c["time"] <= min(seg_start + MAX_SEGMENT_S, end_time - MIN_SEGMENT_S)
        ]

        strong = [c for c in window if c["score"] >= BORDER_THRESHOLD]

        if strong:

            head = strong[0]
            near = [c for c in window if head["time"] <= c["time"] <= head["time"] + PEAK_LOOKAHEAD_S]
            pick = max(near, key=lambda c: c["score"])

        elif remaining > MAX_SEGMENT_S and window:
            # Too long and nothing strong: cut at the best weak border
            pick = max(window, key=lambda c: c["score"])

        elif remaining > MAX_SEGMENT_S:
            # A run of speech with no pauses or full stops at all: cut by length
            cut = max(i for i in range(first, len(words)) if words[i]["end"] <= seg_start + MAX_SEGMENT_S)
            pick = {"idx": max(cut, first), "time": words[max(cut, first)]["end"], "score": 0.0, "signals": {}}

        else:
            break

        borders.append(pick)

        first = pick["idx"] + 1
        seg_start = words[first]["start"]

    return borders


# -----------------------------------------
# CUE LAYERS: phrases where something may be shown
# -----------------------------------------

def lexical_matches(words):

    # (first word, last word, weight) for every deictic word or phrase
    found = []

    for i in range(len(words)):

        for phrase, weight in DEICTIC_PHRASES.items():

            n = len(phrase)

            if tuple(w["norm"] for w in words[i:i + n]) == phrase:
                found.append((i, i + n - 1, weight))

        weight = DEICTIC_WORDS.get(words[i]["norm"])

        if weight:
            found.append((i, i, weight))

    return found


def entity_matches(words, ents):

    return [(i, i, 1.0) for i, w in enumerate(words) if w["norm"] in ents]


def llm_matches(words, lo, hi, summary, model):

    text = " ".join(w["text"] for w in words[lo:hi + 1])

    reply = ask(LLM_PROMPT.format(summary=summary or "unknown") + text, model)

    match = re.search(r"\[.*\]", reply, re.S)

    try:
        items = json.loads(match.group(0)) if match else []
    except json.JSONDecodeError:
        items = []

    found = []

    for item in items:

        if not isinstance(item, dict):
            continue

        phrase = [norm(x) for x in str(item.get("phrase", "")).split() if norm(x)]

        try:
            score = min(max(float(item.get("score", 0)), 0.0), 1.0)
        except (TypeError, ValueError):
            continue

        # Map the phrase back to word positions in this segment
        for i in range(lo, hi - len(phrase) + 2):

            if phrase and [w["norm"] for w in words[i:i + len(phrase)]] == phrase:
                found.append([i, i + len(phrase) - 1, score])
                break

    return found


def merge_spans(spans):

    # Overlapping or touching spans become one cue
    merged = []

    for lo, hi in sorted(spans):

        if merged and lo <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])

    return merged


def layer_value(matches, lo, hi):

    return max((m[2] for m in matches if m[0] >= lo and m[1] <= hi), default=0.0)


def build_cues(words, lo, hi, llm_found, ents, emphasis):

    lexical = [m for m in lexical_matches(words[lo:hi + 1])]
    lexical = [(a + lo, b + lo, w) for a, b, w in lexical]

    entity = [(a + lo, b + lo, w) for a, b, w in entity_matches(words[lo:hi + 1], ents)]

    # A demonstrative points at what comes right after it
    spans = [(a, min(b + LOOK_AHEAD_WORDS, hi)) for a, b, _ in lexical]
    spans += [(a, b) for a, b, _ in llm_found]

    cues = []

    for a, b in merge_spans(spans):

        vector = {
            "lexical": layer_value(lexical, a, b),
            "entity": 1.0 if any(m[0] >= a and m[1] <= b for m in entity) else 0.0,
            "llm": layer_value(llm_found, a, b),
            "emphasis": max(emphasis[a:b + 1])
        }

        score = sum(POINTING_DIRECTION[k] * vector[k] for k in vector)

        if score >= CUE_FLOOR:

            cues.append({
                "start": words[a]["start"],
                "end": words[b]["end"],
                "text": " ".join(w["text"] for w in words[a:b + 1]),
                "vector": {k: round(v, 3) for k, v in vector.items()},
                "score": round(score, 3)
            })

    return cues


# -----------------------------------------
# THE ONION: layers combined into weighted segments
# -----------------------------------------

def onion(manifest_path="processed/ingest.json", use_llm=True, model=None, force=False, tag=None, llm_cues=True):

    manifest_path = Path(manifest_path)
    out_dir = manifest_path.parent

    manifest = load_json(manifest_path)
    context = manifest["context"]

    words = flatten_words(manifest["transcript"])

    out_name = f"onion.{tag}.json" if tag else "onion.json"

    if not words:
        result = {"params": params(), "llm_layer": "skipped", "segments": []}
        save_json(out_dir / out_name, result)
        return result

    ents = entity_tokens(context)
    emphasis = word_emphasis(words, manifest["audio"].get("stft_path"))

    # LLM answers are saved as they arrive, so a cut mid-run resumes where it stopped
    cache_path = out_dir / "onion_llm.json"
    stamp = manifest_path.stat().st_mtime_ns

    cache = {"stamp": stamp, "segments": {}, "borders": {}}

    if cache_path.exists() and not force:
        saved = load_json(cache_path)
        if saved.get("stamp") == stamp:
            cache = {"borders": {}, **saved}

    llm_state = "ok" if use_llm else "skipped"

    candidates = border_candidates(words, context)

    # Where does a new part begin? The LLM reads the speech; if it cannot, the other signals decide alone
    strengths = None

    if use_llm:

        try:
            strengths = llm_border_strengths(words, candidates, context.get("summary"), cache["borders"], model, lambda: save_json(cache_path, cache))

        except (RuntimeError, OSError):
            llm_state = "failed"

    borders = pick_borders(words, score_candidates(candidates, strengths))

    # Segment i covers words[starts[i] .. ends[i]]
    ends = [b["idx"] for b in borders] + [len(words) - 1]
    starts = [0] + [b["idx"] + 1 for b in borders]

    segments = []

    for n, (lo, hi) in enumerate(zip(starts, ends)):

        key = f"{lo}-{hi}"

        if use_llm and llm_cues and key not in cache["segments"] and llm_state == "ok":

            try:
                cache["segments"][key] = llm_matches(words, lo, hi, context.get("summary"), model)
                save_json(cache_path, cache)

            except (RuntimeError, OSError):
                # Network is down: finish without the LLM layer instead of failing the whole run
                llm_state = "failed"

        llm_found = cache["segments"].get(key, [])

        cues = build_cues(words, lo, hi, llm_found, ents, emphasis)

        pointing = max((c["score"] for c in cues), default=0.0)

        seg_words = words[lo:hi + 1]

        mentions = sum(1 for w in seg_words if w["norm"] in ents)
        density = min(mentions / len(seg_words) * 10 / ENTITY_DENSE_PER_10, 1.0)

        weight = (
            SEGMENT_WEIGHTS["pointing"] * pointing
            + SEGMENT_WEIGHTS["entity"] * density
            + SEGMENT_WEIGHTS["emphasis"] * float(np.mean(emphasis[lo:hi + 1]))
        )

        segments.append({
            "id": n,
            "start": seg_words[0]["start"],
            "end": seg_words[-1]["end"],
            "text": " ".join(w["text"] for w in seg_words),
            "border_score": round(borders[n]["score"], 3) if n < len(borders) else 0.0,
            "border_signals": {k: round(v, 2) for k, v in borders[n]["signals"].items()} if n < len(borders) else {},
            "weight": round(weight, 3),
            "pointing": round(pointing, 3),
            "likely_pointing": pointing >= POINTING_THRESHOLD,
            "cues": cues
        })

    if use_llm and llm_state == "ok" and not cache["segments"]:
        llm_state = "skipped"

    result = {"params": params(), "llm_layer": llm_state, "segments": segments}

    save_json(out_dir / out_name, result)

    return result


def params():

    return {
        "border_threshold": BORDER_THRESHOLD,
        "min_segment_s": MIN_SEGMENT_S,
        "max_segment_s": MAX_SEGMENT_S,
        "pointing_threshold": POINTING_THRESHOLD,
        "border_weights": BORDER_WEIGHTS,
        "pointing_direction": POINTING_DIRECTION,
        "segment_weights": SEGMENT_WEIGHTS
    }
