import json
import os
import re
import sys
import time

import numpy as np
import torch

from PIL import Image, ImageDraw
from pycocotools import mask as mask_util
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


CHECKPOINT = r"C:\sam3\checkpoints\sam3.pt"

MAX_PER_PROMPT = 5
CROP_MARGIN = 0.15
CROP_MIN_SCORE = 0.5
OVERLAY_WIDTH = 960
CROP_MAX_SIDE = 768

PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230)]


def slug(text):

    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def append_line(path, obj):

    # Each finished image is on disk before the next one starts, so a power cut loses at most one image
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def finished_ids(path):

    ids = set()

    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass

    return ids


def draw_overlay(image, found, path):

    scale = OVERLAY_WIDTH / image.width
    small = image.resize((OVERLAY_WIDTH, int(image.height * scale)))
    canvas = np.array(small).astype(np.float32)

    for k, (prompt, mask, box, score) in enumerate(found):

        color = np.array(PALETTE[k % len(PALETTE)], dtype=np.float32)
        m = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize(small.size)) > 127

        canvas[m] = canvas[m] * 0.5 + color * 0.5

    out = Image.fromarray(canvas.astype(np.uint8))
    draw = ImageDraw.Draw(out)

    for k, (prompt, mask, box, score) in enumerate(found):

        color = PALETTE[k % len(PALETTE)]
        x0, y0, x1, y1 = [v * scale for v in box]

        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0 + 4, y0 + 2), f"{prompt} {score:.2f}", fill=color)

    out.save(path, quality=88)


def save_crop(image, box, path):

    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * CROP_MARGIN, (y1 - y0) * CROP_MARGIN

    crop = image.crop((max(0, x0 - mx), max(0, y0 - my), min(image.width, x1 + mx), min(image.height, y1 + my)))
    crop.thumbnail((CROP_MAX_SIDE, CROP_MAX_SIDE))
    crop.save(path, quality=92)


def process(processor, item, job_dir, out_dir):

    image = Image.open(os.path.join(job_dir, "in", item["file"])).convert("RGB")

    t0 = time.time()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)

        detections, found, missed = [], [], []

        for prompt in item["prompts"]:

            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(prompt=prompt, state=state)

            scores = state["scores"].float().cpu().numpy()
            boxes = state["boxes"].float().cpu().numpy()
            masks = state["masks"].cpu().numpy()[:, 0] if len(scores) else []

            order = np.argsort(-scores)[:MAX_PER_PROMPT]

            if not len(order):
                missed.append(prompt)
                continue

            for rank, j in enumerate(order):

                mask = masks[j].astype(bool)
                box = [float(v) for v in boxes[j]]

                rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")

                crop = None

                if item.get("zoom") and rank == 0 and scores[j] >= CROP_MIN_SCORE:
                    crop = f"crops/{item['id']}_{slug(prompt)}.jpg"
                    save_crop(image, box, os.path.join(out_dir, crop))

                detections.append({
                    "prompt": prompt, "score": round(float(scores[j]), 4), "box": [round(v, 1) for v in box],
                    "area_frac": round(float(mask.mean()), 5), "rle": rle, "crop": crop
                })

                found.append((prompt, mask, box, float(scores[j])))

    overlay = f"overlays/{item['id']}.jpg"
    draw_overlay(image, found, os.path.join(out_dir, overlay))

    return {
        "id": item["id"], "segment": item["segment"], "t": item["t"], "size": [image.width, image.height],
        "detections": detections, "missed": missed, "overlay": overlay, "seconds": round(time.time() - t0, 2)
    }


def main():

    job_dir = sys.argv[1]
    out_dir = os.path.join(job_dir, "out")

    os.makedirs(os.path.join(out_dir, "overlays"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "crops"), exist_ok=True)

    spec = json.load(open(os.path.join(job_dir, "jobs.json"), encoding="utf-8"))

    results = os.path.join(out_dir, "results.jsonl")
    done = finished_ids(results)

    todo = [item for item in spec["images"] if item["id"] not in done]

    print(f"{len(done)} done, {len(todo)} to do", flush=True)

    if todo:

        model = build_sam3_image_model(checkpoint_path=CHECKPOINT, load_from_HF=False)
        processor = Sam3Processor(model, confidence_threshold=spec.get("threshold", 0.4))

    for n, item in enumerate(todo, 1):

        for attempt in (1, 2):

            try:
                result = process(processor, item, job_dir, out_dir)
                break

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                result = {"id": item["id"], "segment": item["segment"], "t": item["t"], "error": "out of memory"}

            except Exception as exc:
                result = {"id": item["id"], "segment": item["segment"], "t": item["t"], "error": f"{type(exc).__name__}: {exc}"}
                break

        append_line(results, result)

        print(f"[{n}/{len(todo)}] {item['id']} {result.get('error') or str(len(result['detections'])) + ' detections'}", flush=True)

    with open(os.path.join(out_dir, "done.txt"), "w") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S"))


main()
