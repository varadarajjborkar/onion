
from video_processor import VideoProcessor


# -----------------------------------------
# YOUR CODE: pick a video and run it
# -----------------------------------------

processor = VideoProcessor("sample.mp4")

summary = processor.summarize(max_frames=8)

print(summary)
