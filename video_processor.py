
import subprocess
import sys
import threading

import cv2
import numpy as np
import soundfile as sf

from faster_whisper import WhisperModel
from pathlib import Path
from scipy.signal import stft

import llm_env


# -----------------------------------------
# CONFIGURATION
# -----------------------------------------

AUDIO_SAMPLE_RATE = 16000
VIDEO_FPS = 2
AUDIO_CHUNK_SECONDS = 1
VIDEO_FRAME_SIZE = (224, 224)

# Audio STFT configuration
N_FFT = 400
HOP_LENGTH = 160

# Speech-to-text model (faster-whisper, runs locally on CPU)
WHISPER_MODEL = "base"


# -----------------------------------------
# HELPERS
# -----------------------------------------

def encode_jpeg(rgb_frame):

    bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    ok, buffer = cv2.imencode(".jpg", bgr)

    return buffer.tobytes()


def clock(seconds):

    seconds = int(seconds)

    return f"{seconds // 60}:{seconds % 60:02d}"


class Spinner:

    FRAMES = "|/-\\"
    WIDTH = 100

    def __init__(self, text=""):

        self.text = text
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def _spin(self):

        i = 0

        while not self._stop.is_set():

            with self._lock:
                sys.stdout.write(f"\r{self.FRAMES[i % len(self.FRAMES)]} {self.text}".ljust(self.WIDTH))
                sys.stdout.flush()

            i += 1
            self._stop.wait(0.1)

    def __enter__(self):

        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

        return self

    def __exit__(self, *exc):

        self._stop.set()
        self._thread.join()

        with self._lock:
            sys.stdout.write("\r" + " " * self.WIDTH + "\r")
            sys.stdout.flush()

    def log(self, line):

        # Print a finished line above the spinner without breaking it
        with self._lock:
            sys.stdout.write("\r" + " " * self.WIDTH + "\r" + line + "\n")
            sys.stdout.flush()


# -----------------------------------------
# VIDEO PROCESSOR
# -----------------------------------------

class VideoProcessor:

    def __init__(self, video_path, output_dir="processed"):

        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)

    # -----------------------------------------
    # 1. EXTRACT AUDIO AND VIDEO STREAMS
    # -----------------------------------------

    def extract_streams(self):

        self.output_dir.mkdir(parents=True, exist_ok=True)

        audio_path = self.output_dir / "audio.wav"

        # Extract audio independently using FFmpeg
        command = [
            "ffmpeg",
            "-y",
            "-i", str(self.video_path),
            "-map", "0:a:0",
            "-vn",
            "-ac", "1",
            "-ar", str(AUDIO_SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            str(audio_path)
        ]

        subprocess.run(
            command,
            check=True,
            capture_output=True
        )

        return {
            "video_path": str(self.video_path),
            "audio_path": str(audio_path)
        }

    # -----------------------------------------
    # 2. VIDEO PROCESSING
    # -----------------------------------------

    def process_video(self, target_fps=VIDEO_FPS, frame_size=VIDEO_FRAME_SIZE):

        cap = cv2.VideoCapture(str(self.video_path))

        if not cap.isOpened():
            raise ValueError("Could not open video.")

        source_fps = cap.get(cv2.CAP_PROP_FPS)

        if source_fps <= 0:
            raise ValueError("Invalid video FPS.")

        video_matrices = []
        timestamps = []

        next_sample_time = 0.0

        while True:

            success, frame = cap.read()

            if not success:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            if timestamp + 1e-6 >= next_sample_time:

                # OpenCV returns BGR; convert to RGB
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # Resize for consistent matrix dimensions
                if frame_size is not None:
                    frame = cv2.resize(
                        frame,
                        frame_size,
                        interpolation=cv2.INTER_AREA
                    )

                # Each frame is an individual image matrix
                video_matrices.append(frame)

                timestamps.append(timestamp)

                next_sample_time += 1.0 / target_fps

        cap.release()

        if not video_matrices:
            raise ValueError("No video frames extracted.")

        # Shape: (Number of frames, Height, Width, Channels)
        video_array = np.stack(video_matrices)

        return {
            "frames": video_array,
            "timestamps": np.array(timestamps),
            "fps": target_fps,
            "shape": video_array.shape
        }

    # -----------------------------------------
    # 3. AUDIO PROCESSING
    # -----------------------------------------

    def process_audio(self, audio_path,
                      sample_rate=AUDIO_SAMPLE_RATE,
                      chunk_seconds=AUDIO_CHUNK_SECONDS):

        # Load audio as a floating-point waveform
        waveform, sr = sf.read(
            audio_path,
            dtype="float32"
        )

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        if sr != sample_rate:
            raise ValueError("Unexpected audio sample rate.")

        samples_per_chunk = int(sample_rate * chunk_seconds)

        audio_matrices = []
        timestamps = []

        # Split waveform into fixed-duration chunks
        for start in range(0, len(waveform), samples_per_chunk):

            chunk = waveform[start:start + samples_per_chunk]

            # Pad the final chunk to maintain consistent dimensions
            if len(chunk) < samples_per_chunk:
                chunk = np.pad(
                    chunk,
                    (0, samples_per_chunk - len(chunk))
                )

            # Short-Time Fourier Transform
            frequencies, times, Zxx = stft(
                chunk,
                fs=sample_rate,
                window="hann",
                nperseg=N_FFT,
                noverlap=N_FFT - HOP_LENGTH,
                boundary=None,
                padded=False
            )

            # Power spectrogram
            power_spectrogram = np.abs(Zxx) ** 2

            # Convert to logarithmic scale
            log_spectrogram = 10 * np.log10(
                power_spectrogram + 1e-10
            )

            # Each audio chunk becomes a 2D matrix
            audio_matrices.append(log_spectrogram.astype(np.float32))

            timestamps.append(start / sample_rate)

        # Shape: (Number of audio chunks, Frequency bins, Time bins)
        audio_array = np.stack(audio_matrices)

        return {
            "spectrograms": audio_array,
            "timestamps": np.array(timestamps),
            "sample_rate": sample_rate,
            "shape": audio_array.shape
        }

    # -----------------------------------------
    # 4. SPEECH TO TEXT (timed transcript of the audio)
    # -----------------------------------------

    def transcribe(self, audio_path=None, model_size=WHISPER_MODEL):

        audio_path = audio_path or self.output_dir / "audio.wav"

        # Already 16 kHz mono float32 from extract_streams, so pass the samples directly
        waveform, _ = sf.read(audio_path, dtype="float32")

        model = WhisperModel(model_size, device="cpu", compute_type="int8")

        segments, _ = model.transcribe(waveform, vad_filter=True)

        return [
            {"start": s.start, "end": s.end, "text": s.text.strip()}
            for s in segments
        ]

    # -----------------------------------------
    # 5. COMPLETE PIPELINE
    # -----------------------------------------

    def process_video_file(self):

        # Step 1: Separate audio and video
        streams = self.extract_streams()

        # Step 2: Process video independently
        video_data = self.process_video()

        # Step 3: Process audio independently
        audio_data = self.process_audio(
            streams["audio_path"]
        )

        return {
            "video": video_data,
            "audio": audio_data
        }

    # -----------------------------------------
    # 6. SEGMENT SUMMARIES (frames + speech + audio levels per time window -> LLM)
    # -----------------------------------------

    def summarize(self, segment_seconds=10, frames_per_segment=2, model=None):

        with Spinner("Decoding video and audio"):
            result = self.process_video_file()

        with Spinner("Transcribing speech"):
            speech = self.transcribe()

        frames = result["video"]["frames"]
        frame_times = result["video"]["timestamps"]

        audio = result["audio"]["spectrograms"]
        audio_times = result["audio"]["timestamps"]

        # Audio level per second, from the mean of each log spectrogram
        levels = audio.mean(axis=(1, 2))

        total = len(audio_times) * AUDIO_CHUNK_SECONDS

        starts = np.arange(0, total, segment_seconds)

        results = []

        with Spinner("") as spinner:

            for n, start in enumerate(starts, 1):

                end = min(start + segment_seconds, total)

                spinner.text = f"Summarizing segment {n}/{len(starts)}"

                # Evenly spaced frames inside this window
                in_window = np.where((frame_times >= start) & (frame_times < end))[0]

                picks = in_window[np.linspace(0, len(in_window) - 1, min(frames_per_segment, len(in_window))).astype(int)] if len(in_window) else []

                images = [encode_jpeg(frames[i]) for i in picks]

                labels = [f"Frame {k + 1} at {clock(frame_times[i])}" for k, i in enumerate(picks)]

                window_audio = np.where((audio_times >= start) & (audio_times < end))[0]

                # Transcript lines that overlap this window
                spoken = " ".join(t["text"] for t in speech if t["start"] < end and t["end"] > start)

                prompt = (
                    f"Summarize this part of a video, from {clock(start)} to {clock(end)}. "
                    "Describe what happens, what is said, and any on-screen text in one or two sentences. "
                    "If the frames and speech are not enough to tell, say so.\n\n"
                    + "\n".join(labels)
                    + f"\n\nAudio: average level {levels[window_audio].mean():.1f} dB."
                    + f"\nSpeech: {spoken or 'none'}"
                )

                summary = llm_env.describe(images, prompt, model=model)

                spinner.log(f"[{n}/{len(starts)}] {clock(start)}-{clock(end)}  {summary}")

                results.append({
                    "start": float(start),
                    "end": float(end),
                    "summary": summary
                })

        return results
