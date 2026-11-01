
import sys

import numpy as np

from pathlib import Path

import onion

from ingest import load_json
from video_processor import clock


# -----------------------------------------
# YOUR CODE: compare chunking variants made with run_onion.py --tag=name
# usage: python pipeline/compare_borders.py base cross_swap cross_add ...
# -----------------------------------------

# Two borders this close (seconds) count as the same place
TOLERANCE_S = 6.0

# The LLM's own votes at or above this strength count as a border it wanted
LLM_VOTE = 0.5


def border_times(result):

    return [s["end"] for s in result["segments"][:-1]]


def hit_rate(a, b):

    # Share of borders in a that have a border in b close by
    return sum(any(abs(x - y) <= TOLERANCE_S for y in b) for x in a) / len(a) if a else 0.0


def llm_votes(base="processed"):

    manifest = load_json(Path(base) / "ingest.json")

    words = onion.flatten_words(manifest["transcript"])

    candidates = onion.border_candidates(words, manifest["context"])

    cache = load_json(Path(base) / "onion_llm.json")["borders"]

    times = []

    for votes in cache.values():
        for line, strength in votes:
            if strength >= LLM_VOTE and 2 <= line <= len(candidates) + 1:
                times.append(candidates[line - 2]["time"])

    return sorted(set(round(t, 1) for t in times))


names = sys.argv[1:]

if not names:
    raise SystemExit("Give the tags to compare, for example: base cross_swap cross_add")

variants = {n: load_json(Path(f"processed/onion.{n}.json")) for n in names}
times = {n: border_times(r) for n, r in variants.items()}

votes = llm_votes()

print(f"The LLM voted for {len(votes)} border places on its own.\n")

print(f"{'variant':<14}{'segments':>9}{'median s':>10}{'shortest':>10}{'longest':>9}{'borders also voted by the LLM':>32}")

for n, r in variants.items():

    length = np.array([s["end"] - s["start"] for s in r["segments"]])

    print(f"{n:<14}{len(length):>9}{np.median(length):>10.0f}{length.min():>10.0f}{length.max():>9.0f}{hit_rate(times[n], votes):>31.0%}")

print("\nAgreement: share of the row's borders that the column also has (within 6 s)")
print(f"{'':<14}" + "".join(f"{n:>14}" for n in names))

for a in names:
    print(f"{a:<14}" + "".join(f"{hit_rate(times[a], times[b]):>14.0%}" for b in names))

# One line per border place, with the words around it, and which variants cut there
places = sorted({round(t) for ts in times.values() for t in ts})

merged = []

for t in places:
    if merged and t - merged[-1] <= TOLERANCE_S:
        continue
    merged.append(t)

print("\nBorder places (a letter for each variant that cuts there):")
print("  " + "  ".join(f"{chr(65 + k)}={n}" for k, n in enumerate(names)))

def around(t):

    # The words either side of a border place, taken from the first variant that cuts there
    for n in names:
        segs = variants[n]["segments"]
        for k in range(len(segs) - 1):
            if abs(segs[k]["end"] - t) <= TOLERANCE_S:
                return segs[k]["text"], segs[k + 1]["text"]

    return "", ""


for t in merged:

    marks = "".join(chr(65 + k) if any(abs(t - x) <= TOLERANCE_S for x in times[n]) else "." for k, n in enumerate(names))

    before, after = around(t)

    print(f"  {clock(t):>5}  {marks}  ...{' '.join(before.split()[-7:])} | {' '.join(after.split()[:7])}...")
