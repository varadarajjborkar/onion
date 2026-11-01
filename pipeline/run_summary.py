
import json

from datetime import datetime
from pathlib import Path

from video_processor import VideoProcessor, clock


# -----------------------------------------
# YOUR CODE: pick a video and run it
# -----------------------------------------

processor = VideoProcessor("sample/sample.mp4")

segments = processor.summarize(segment_seconds=10, frames_per_segment=2)

print(f"\nDone: {len(segments)} segments summarized.")


# -----------------------------------------
# SAVE THE LOG (one text and one JSON file per run)
# -----------------------------------------

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

text_log = log_dir / f"summary-{stamp}.txt"
json_log = log_dir / f"summary-{stamp}.json"

lines = [
    f"[{n}/{len(segments)}] {clock(s['start'])}-{clock(s['end'])}  {s['summary']}"
    for n, s in enumerate(segments, 1)
]

text_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
json_log.write_text(json.dumps(segments, indent=2), encoding="utf-8")

print(f"Log saved: {text_log}")
print(f"JSON saved: {json_log}")
