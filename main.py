
import cv2
import numpy as np
import soundfile as sf
import subprocess

from pathlib import Path
from scipy.signal import stft


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


# -----------------------------------------
# 1. EXTRACT AUDIO AND VIDEO STREAMS
# -----------------------------------------

def extract_streams(video_path, output_dir="processed"):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = output_dir / "audio.wav"

    # Extract audio independently using FFmpeg
    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
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
        "video_path": str(video_path),
        "audio_path": str(audio_path)
    }


# -----------------------------------------
# 2. VIDEO PROCESSING
# -----------------------------------------

def process_video(video_path, target_fps=VIDEO_FPS,
                  frame_size=VIDEO_FRAME_SIZE):

    cap = cv2.VideoCapture(str(video_path))

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

def process_audio(audio_path,
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
# 4. COMPLETE PIPELINE
# -----------------------------------------

def process_video_file(video_path):

    # Step 1: Separate audio and video
    streams = extract_streams(video_path)

    # Step 2: Process video independently
    video_data = process_video(
        streams["video_path"]
    )

    # Step 3: Process audio independently
    audio_data = process_audio(
        streams["audio_path"]
    )

    return {
        "video": video_data,
        "audio": audio_data
    }


# -----------------------------------------
# 5. EXECUTION
# -----------------------------------------

if __name__ == "__main__":

    result = process_video_file("input.mp4")

    print("VIDEO")
    print("Shape:", result["video"]["shape"])
    print("Timestamps:", result["video"]["timestamps"][:5])

    print("\nAUDIO")
    print("Shape:", result["audio"]["shape"])
    print("Timestamps:", result["audio"]["timestamps"][:5])

    # Access individual matrices
    first_frame = result["video"]["frames"][0]
    first_audio_matrix = result["audio"]["spectrograms"][0]

    print("\nFirst frame:", first_frame.shape)
    print("First audio matrix:", first_audio_matrix.shape)
