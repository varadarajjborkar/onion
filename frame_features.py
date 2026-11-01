
import os

import cv2
import numpy as np

from pathlib import Path


# -----------------------------------------
# CONFIGURATION
# -----------------------------------------

# Frames looked at per second; features are about change between neighbouring samples
SAMPLE_FPS = 4

# Working size; wide enough to see blur and shifts, small enough to be fast
WORK_SIZE = (320, 180)

HIST_BINS = 16

# Save progress this often (in samples), so a power cut costs minutes, not the whole decode
CHECKPOINT_EVERY = 1000

FIELDS = ["t", "diff", "hist", "dx", "dy", "resid", "sharp", "bright"]


# -----------------------------------------
# FRAME FEATURES
# -----------------------------------------

def color_hist(frame):

    return [
        cv2.normalize(cv2.calcHist([frame], [c], None, [HIST_BINS], [0, 256]), None, 1, 0, cv2.NORM_L1)
        for c in range(3)
    ]


def sample_features(frame, gray, prev):

    # prev is (gray, hists) of the previous sample, or None for the very first one
    hists = color_hist(frame)

    row = {
        "diff": 0.0, "hist": 0.0, "dx": 0.0, "dy": 0.0, "resid": 0.0,
        "sharp": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
        "bright": float(gray.mean() / 255.0)
    }

    if prev is not None:

        prev_gray, prev_hists = prev

        row["diff"] = float(np.abs(gray.astype(np.int16) - prev_gray).mean() / 255.0)

        row["hist"] = float(np.mean([cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA) for a, b in zip(prev_hists, hists)]))

        # Camera movement: how far the whole picture slid between the two samples
        (dx, dy), response = cv2.phaseCorrelate(prev_gray.astype(np.float32), gray.astype(np.float32))

        # What is left after undoing that slide is things moving inside the picture
        shift = np.float32([[1, 0, -dx], [0, 1, -dy]])
        shifted = cv2.warpAffine(gray, shift, WORK_SIZE, borderMode=cv2.BORDER_REPLICATE)

        row["dx"], row["dy"] = float(dx), float(dy)
        row["resid"] = float(np.abs(shifted.astype(np.int16) - prev_gray).mean() / 255.0)

    return row, (gray.astype(np.int16), hists)


def save_arrays(path, rows, next_frame):

    tmp = path.with_name(path.name + ".tmp")

    with open(tmp, "wb") as f:
        np.savez_compressed(f, next_frame=next_frame, **{k: np.array([r[k] for r in rows], dtype=np.float32) for k in FIELDS})
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


def load_arrays(path):

    with np.load(path) as data:
        rows = [{k: float(data[k][i]) for k in FIELDS} for i in range(len(data["t"]))]
        return rows, int(data["next_frame"])


def extract(video_path, out_path, force=False):

    out_path = Path(out_path)
    part_path = out_path.with_name(out_path.stem + ".partial.npz")

    if out_path.exists() and not force:
        with np.load(out_path) as data:
            return {k: data[k] for k in FIELDS}

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(int(round(fps / SAMPLE_FPS)), 1)

    rows, frame_idx = [], 0

    # Pick up from the last checkpoint
    if part_path.exists() and not force:
        rows, frame_idx = load_arrays(part_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    prev = None

    while True:

        if not cap.grab():
            break

        if frame_idx % step == 0:

            ok, frame = cap.retrieve()

            if not ok:
                break

            frame = cv2.resize(frame, WORK_SIZE, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            row, prev_gray = sample_features(frame, gray, prev)

            row["t"] = frame_idx / fps
            rows.append(row)

            prev = (prev_gray[0], prev_gray[1])

            if len(rows) % CHECKPOINT_EVERY == 0:
                save_arrays(part_path, rows, frame_idx + 1)

        frame_idx += 1

    cap.release()

    save_arrays(out_path, rows, frame_idx)
    part_path.unlink(missing_ok=True)

    return {k: np.array([r[k] for r in rows], dtype=np.float32) for k in FIELDS}
