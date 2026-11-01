
import json
import re
import threading

import cv2

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ingest import ask, load_json, save_json


# -----------------------------------------
# CONFIGURATION
# -----------------------------------------

# Calls to the vision LLM running at the same time
WORKERS = 3

# Images are sent no wider than this; the full size stays on disk
SEND_WIDTH = 1280

# At most this many named things are taken per keyframe (the biggest, most visible come first in the reply)
MAX_ITEMS = 10

KINDS = ["object", "person", "animation", "text", "logo", "diagram", "screen", "other"]

DISCOVER_PROMPT = (
    "List the distinct things a viewer could be shown or told about in this video frame: objects, people, "
    "animated or cartoon characters, drawings, icons, logos, memes, diagrams, screens and documents. "
    "Skip the plain background. For each give: "
    '"name" (two to four concrete words that an image segmentation model could find, such as "cartoon student" or "red bicycle helmet"), '
    f'"kind" (one of: {", ".join(KINDS)}), '
    '"detail" (one short factual sentence; for an animated or unusual thing say how it looks). '
    'Also give "text": every piece of readable text on screen, exactly as written. '
    f'Reply with only a JSON object: {{"items": [...], "text": [...]}}. At most {MAX_ITEMS} items. Do not use em dashes.'
)

CAPTION_PROMPT = (
    "This is a cropped piece of a video frame. In one short sentence say what it is. "
    "If it is an animation, drawing, meme, icon or logo, say so and describe how it looks. "
    "If it contains text, quote it. If it is only background or you cannot tell, say that. Do not use em dashes."
)


# -----------------------------------------
# HELPERS
# -----------------------------------------

def jpeg_bytes(path, width=SEND_WIDTH):

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Could not read {path}")

    if image.shape[1] > width:
        image = cv2.resize(image, (width, int(image.shape[0] * width / image.shape[1])), interpolation=cv2.INTER_AREA)

    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def clean_discovery(reply):

    match = re.search(r"\{.*\}", reply, re.S)

    try:
        obj = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        obj = {}

    items = []

    for item in obj.get("items", []) if isinstance(obj, dict) else []:

        if not isinstance(item, dict) or not str(item.get("name", "")).strip():
            continue

        kind = str(item.get("kind", "other")).strip().lower()

        items.append({
            "name": str(item["name"]).strip().lower(),
            "kind": kind if kind in KINDS else "other",
            "detail": str(item.get("detail", "")).strip()
        })

    text = [str(t).strip() for t in (obj.get("text", []) if isinstance(obj, dict) else []) if str(t).strip()]

    return {"items": items[:MAX_ITEMS], "text": text}


def run_parallel(tasks, work, out_path, model=None):

    # tasks: {key: input}. Each finished answer is saved at once, so a cut resumes where it stopped.
    # Returns (answers, number that could not be done).
    out_path = Path(out_path)

    done = load_json(out_path) if out_path.exists() else {}

    pending = {k: v for k, v in tasks.items() if k not in done}

    lock = threading.Lock()
    failed = []

    def one(key):

        try:
            answer = work(pending[key], model)
        except (RuntimeError, OSError, ValueError):
            failed.append(key)
            return

        with lock:
            done[key] = answer
            save_json(out_path, done)

    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(one, pending))

    return {k: done[k] for k in tasks if k in done}, len(failed)


def discover_one(path, model):

    return clean_discovery(ask(DISCOVER_PROMPT, model, [jpeg_bytes(path)]))


def caption_one(path, model):

    return ask(CAPTION_PROMPT, model, [jpeg_bytes(path, 640)]).strip()


def discover(images, out_path, model=None):

    # images: {id: path to the keyframe}
    return run_parallel(images, discover_one, out_path, model)


def caption(crops, out_path, model=None):

    # crops: {id: path to the cropped image}
    return run_parallel(crops, caption_one, out_path, model)
