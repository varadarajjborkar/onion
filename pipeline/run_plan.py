
import sys

import knobs

from boost import plan
from video_processor import clock


knobs.apply()


# -----------------------------------------
# YOUR CODE: run phase 3 on the output of phases 1 and 2
# usage: python run_plan.py [processed/ingest.json] [--force]
# -----------------------------------------

args = [a for a in sys.argv[1:] if not a.startswith("--")]

manifest = args[0] if args else "processed/ingest.json"

result = plan(manifest, force="--force" in sys.argv)

print(f"Model   : {result['model']['kind']}, labels from {result['model']['labels_from']}")
print(f"Windows : {result['model']['window_counts']}")
print(f"Features: {dict(list(result['model']['importance'].items())[:6])}\n")

for s in result["segments"]:

    top = max(s["situation"], key=s["situation"].get)

    print(f"[{s['id']:2d}] {clock(s['start'])}-{clock(s['end'])}  {s['mode']:<17} focus {s['focus']:.2f}  mostly {top:<14} cuts {len(s['cuts'])}  keyframes {[k['t'] for k in s['keyframes']]}")

print("\nSaved: processed/plan.json")
