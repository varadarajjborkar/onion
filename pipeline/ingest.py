
import json
import os
import re
import time

import cv2
import numpy as np
import soundfile as sf

from pathlib import Path

import llm_env

from video_processor import VideoProcessor, Spinner, clock


# -----------------------------------------
# CONFIGURATION
# -----------------------------------------

# Longest slice of transcript sent to the LLM in one call; longer videos are done in slices
CONTEXT_CHUNK_CHARS = 12000

# How many partial contexts are merged per LLM call when a video needs several slices
MERGE_FAN_IN = 8

# Network cuts are expected, so every LLM call is retried with growing waits
LLM_TRIES = 5
LLM_BACKOFF_S = 2

CONTEXT_PROMPT = (
    "Below is the timed transcript of the speech in a video. "
    "Reply with only a JSON object with these keys: "
    '"summary" (two or three sentences on what the speaker talks about), '
    '"setting" (where and how the video seems to be set, only if the speech says so, else an empty string), '
    '"topics" (short list of topics, in the order they come up), '
    '"entities" (list of the people, objects, animals and places mentioned that could be visible on screen). '
    "Use only what the transcript says.\n\n"
)

MERGE_PROMPT = (
    "Below are JSON notes about consecutive parts of one video, one per line. "
    "Merge them into one JSON object with the same keys: "
    '"summary", "setting", "topics", "entities". '
    "Reply with only the JSON object. Keep the summary to three or four sentences, "
    "keep topics in order, and drop duplicate entities.\n\n"
)


# -----------------------------------------
# HELPERS
# -----------------------------------------

def atomic_write(path, data):

    # Write beside the target, then swap in one step, so a power cut never leaves a torn file
    tmp = path.with_name(path.name + ".tmp")

    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


def save_json(path, obj):

    atomic_write(path, json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8"))


def load_json(path):

    return json.loads(path.read_text(encoding="utf-8"))


def ask(prompt, model=None, images=()):

    for attempt in range(1, LLM_TRIES + 1):

        try:
            return llm_env.describe(list(images), prompt, model=model)

        except (RuntimeError, OSError) as err:

            if attempt == LLM_TRIES:
                raise

            time.sleep(LLM_BACKOFF_S * 2 ** (attempt - 1))


def chunk_lines(lines, limit):

    chunks, current, size = [], [], 0

    for line in lines:

        if current and size + len(line) > limit:
            chunks.append(current)
            current, size = [], 0

        current.append(line)
        size += len(line) + 1

    if current:
        chunks.append(current)

    return chunks


def read_context(text):

    match = re.search(r"\{.*\}", text, re.S)

    try:
        raw = json.loads(match.group(0))
    except (AttributeError, json.JSONDecodeError):
        # The model did not return JSON; keep what it said as the summary
        raw = {"summary": text.strip()}

    return {
        "summary": str(raw.get("summary", "")),
        "setting": str(raw.get("setting", "")),
        "topics": [str(x) for x in raw.get("topics", [])],
        "entities": [str(x) for x in raw.get("entities", [])]
    }


def build_context(transcript, model=None):

    if not transcript:
        return {"summary": "", "setting": "", "topics": [], "entities": [], "note": "no speech detected"}

    lines = [f"[{clock(t['start'])}] {t['text']}" for t in transcript]

    chunks = chunk_lines(lines, CONTEXT_CHUNK_CHARS)

    partials = [read_context(ask(CONTEXT_PROMPT + "\n".join(c), model)) for c in chunks]

    # Long videos: merge the partial contexts in groups until one is left
    while len(partials) > 1:

        groups = [partials[i:i + MERGE_FAN_IN] for i in range(0, len(partials), MERGE_FAN_IN)]

        partials = [
            read_context(ask(MERGE_PROMPT + "\n".join(json.dumps(p) for p in g), model))
            for g in groups
        ]

    return partials[0]


# -----------------------------------------
# PHASE 1: INGEST
# video -> (muted video, audio) ; audio -> STFT -> transcript ; transcript -> LLM context
# -----------------------------------------

class Ingest:

    ARTIFACTS = ["source.json", "audio_stft.npz", "transcript.json", "context.json", "ingest.json"]

    def __init__(self, video_path, output_dir="processed", model=None):

        self.processor = VideoProcessor(video_path, output_dir)

        self.video_path = self.processor.video_path
        self.output_dir = self.processor.output_dir
        self.model = model

    # A finished stage is saved to disk; a rerun after a crash or power cut skips it
    def _stage(self, name, build, load, force):

        path = self.output_dir / name

        if path.exists() and not force:
            return load(path)

        return build(path)

    def _fingerprint(self):

        stat = self.video_path.stat()

        return {"path": str(self.video_path), "size": stat.st_size, "mtime": int(stat.st_mtime)}

    def _reset(self):

        for name in self.ARTIFACTS:
            (self.output_dir / name).unlink(missing_ok=True)

        for path in self.output_dir.glob("*.part*"):
            path.unlink()

        for path in self.output_dir.glob("*.tmp"):
            path.unlink()

    def run(self, force=False):

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # A different or changed video must not reuse old results
        fingerprint = self._fingerprint()
        stored = self.output_dir / "source.json"

        if force or not stored.exists() or load_json(stored) != fingerprint:
            self._reset()
            force = True
            save_json(stored, fingerprint)

        audio_path = self.output_dir / "audio.wav"
        muted_path = self.output_dir / f"video_muted{self.video_path.suffix}"

        # 1. Split the video into a muted video and an audio track
        if force or not (audio_path.exists() and muted_path.exists()):
            with Spinner("Splitting audio and video"):
                self.processor.extract_streams()

        # 2. STFT of the audio
        def build_stft(path):

            with Spinner("Computing STFT of the audio"):
                audio = self.processor.process_audio(audio_path)

            tmp = path.with_name(path.name + ".tmp")

            with open(tmp, "wb") as f:
                np.savez_compressed(f, spectrograms=audio["spectrograms"], timestamps=audio["timestamps"])
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp, path)

            return audio["shape"]

        def load_stft(path):

            with np.load(path) as data:
                return data["spectrograms"].shape

        stft_shape = self._stage("audio_stft.npz", build_stft, load_stft, force)

        # 3. Transcript as {timestamps : text}
        def build_transcript(path):

            with Spinner("Transcribing speech"):
                transcript = self.processor.transcribe(audio_path)

            save_json(path, transcript)

            return transcript

        transcript = self._stage("transcript.json", build_transcript, load_json, force)

        # 4. LLM turns the transcript into a context for the later stages
        def build_ctx(path):

            with Spinner("Building context from the transcript"):
                context = build_context(transcript, self.model)

            save_json(path, context)

            return context

        context = self._stage("context.json", build_ctx, load_json, force)

        # 5. One manifest that the next phase reads
        cap = cv2.VideoCapture(str(muted_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        manifest = {
            "source": fingerprint,
            "video": {
                "path": str(muted_path),
                "fps": fps,
                "frames": frames,
                "width": width,
                "height": height,
                "duration": frames / fps if fps > 0 else 0.0
            },
            "audio": {
                "path": str(audio_path),
                "duration": sf.info(audio_path).duration,
                "stft_path": str(self.output_dir / "audio_stft.npz"),
                "stft_shape": list(stft_shape)
            },
            "transcript": transcript,
            "context": context
        }

        save_json(self.output_dir / "ingest.json", manifest)

        return manifest
