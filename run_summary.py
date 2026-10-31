
from video_processor import VideoProcessor


# -----------------------------------------
# YOUR CODE: pick a video and run it
# -----------------------------------------

processor = VideoProcessor("sample/sample.mp4")

segments = processor.summarize(segment_seconds=10, frames_per_segment=2)

print(f"\nDone: {len(segments)} segments summarized.")
