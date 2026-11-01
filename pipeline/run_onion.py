
import sys

import knobs

from onion import onion
from video_processor import clock


knobs.apply()


# -----------------------------------------
# YOUR CODE: run phase 2 on the output of phase 1
# usage: python run_onion.py [processed/ingest.json] [--no-llm] [--no-cue-llm] [--tag=name] [--force]
# --tag saves to onion.<name>.json instead of onion.json, so variants can be compared
# -----------------------------------------

args = [a for a in sys.argv[1:] if not a.startswith("--")]

manifest = args[0] if args else "processed/ingest.json"

tag = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--tag=")), None)

result = onion(manifest, use_llm="--no-llm" not in sys.argv, force="--force" in sys.argv, tag=tag, llm_cues="--no-cue-llm" not in sys.argv)

segments = result["segments"]

print(f"LLM layer: {result['llm_layer']}")
print(f"{len(segments)} segments\n")

for s in segments:

    flag = "POINT" if s["likely_pointing"] else "     "

    print(f"[{s['id']:2d}] {clock(s['start'])}-{clock(s['end'])}  weight {s['weight']:.2f}  pointing {s['pointing']:.2f} {flag}  border {s['border_score']:.2f}")

    for c in s["cues"]:
        print(f"       cue {c['score']:.2f}  \"{c['text']}\"  {c['vector']}")

print(f"\nSaved: processed/onion{'.' + tag if tag else ''}.json")
