
import sys

import knobs

from ingest import Ingest
from video_processor import clock


knobs.apply()


# -----------------------------------------
# YOUR CODE: pick a video and run phase 1
# usage: python run_ingest.py [video] [--force]
# -----------------------------------------

args = [a for a in sys.argv[1:] if not a.startswith("--")]

video = args[0] if args else "sample/sample.mp4"
force = "--force" in sys.argv

manifest = Ingest(video).run(force=force)

v, a = manifest["video"], manifest["audio"]

print(f"Video   : {v['width']}x{v['height']} @ {v['fps']:.1f} fps, {v['frames']} frames, {clock(v['duration'])}")
print(f"Audio   : {clock(a['duration'])}, STFT shape {tuple(a['stft_shape'])}")
print(f"Speech  : {len(manifest['transcript'])} transcript segments")

for t in manifest["transcript"]:
    print(f"  [{clock(t['start'])}-{clock(t['end'])}] {t['text']}")

print("Context :")

for key, value in manifest["context"].items():
    print(f"  {key}: {value}")

print("\nSaved   : processed/ingest.json")
