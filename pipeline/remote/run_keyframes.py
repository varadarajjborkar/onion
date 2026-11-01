import json
import os
import re
import sys
import time

import numpy as np
import torch

from PIL import Image, ImageDraw
from pycocotools import mask as mask_util
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


CHECKPOINT = r"C:\sam3\checkpoints\sam3.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_CHECKPOINT = r"C:\sam2\checkpoints\sam2.1_hiera_base_plus.pt"

# "Segment everything" keeps regions no name matched: not tiny specks, not the whole frame, not already found
AMG_SIDE = 1024
UNLABELED_MIN_AREA = 0.004
UNLABELED_MAX_AREA = 0.6
KNOWN_OVERLAP = 0.6
SAME_OVERLAP = 0.5

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


def find_unlabeled(amg, image, known, item_id, out_dir, limit):

    scale = min(1.0, AMG_SIDE / max(image.size))
    small = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    total = small.width * small.height

    known = [np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(small.size, Image.NEAREST)) > 127 for m in known]

    kept = []

    for ann in sorted(amg.generate(np.array(small)), key=lambda a: -(a["predicted_iou"] * a["stability_score"])):

        seg = ann["segmentation"]
        area = int(seg.sum())

        if not (UNLABELED_MIN_AREA <= area / total <= UNLABELED_MAX_AREA):
            continue

        if any((seg & k).sum() / area > KNOWN_OVERLAP for k in known):
            continue

        if any((seg & other).sum() / (seg | other).sum() > SAME_OVERLAP for _, other in kept):
            continue

        kept.append((ann, seg))

        if len(kept) == limit:
            break

    found, out = [], []

    image_scale = 1.0 / scale

    for n, (ann, seg) in enumerate(kept):

        x, y, w, h = ann["bbox"]
        box = [x * image_scale, y * image_scale, (x + w) * image_scale, (y + h) * image_scale]

        crop = f"crops/{item_id}_u{n}.jpg"
        save_crop(image, box, os.path.join(out_dir, crop))

        out.append({"box": [round(v, 1) for v in box], "area_frac": round(float(seg.mean()), 5),
                    "score": round(float(ann["predicted_iou"]), 4), "crop": crop})

        found.append(("?", seg, box, float(ann["predicted_iou"])))

    return out, found


def process(processor, amg, item, job_dir, out_dir, limit):

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

    unlabeled = []

    if amg is not None:

        with torch.autocast("cuda", dtype=torch.bfloat16):
            unlabeled, extra = find_unlabeled(amg, image, [m for _, m, _, _ in found], item["id"], out_dir, limit)

        found += extra

    overlay = f"overlays/{item['id']}.jpg"
    draw_overlay(image, found, os.path.join(out_dir, overlay))

    return {
        "id": item["id"], "segment": item["segment"], "t": item["t"], "size": [image.width, image.height],
        "detections": detections, "unlabeled": unlabeled, "missed": missed, "overlay": overlay, "seconds": round(time.time() - t0, 2)
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

        amg = None

        if spec.get("everything"):
            sam2 = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device="cuda", apply_postprocessing=False)
            amg = SAM2AutomaticMaskGenerator(sam2, points_per_side=16, points_per_batch=32, pred_iou_thresh=0.8,
                                             stability_score_thresh=0.9, min_mask_region_area=200)

    for n, item in enumerate(todo, 1):

        for attempt in (1, 2):

            try:
                result = process(processor, amg, item, job_dir, out_dir, spec.get("max_unlabeled", 4))
                break

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                result = {"id": item["id"], "segment": item["segment"], "t": item["t"], "error": "out of memory"}

            except Exception as exc:
                result = {"id": item["id"], "segment": item["segment"], "t": item["t"], "error": f"{type(exc).__name__}: {exc}"}
                break

        append_line(results, result)

        print(f"[{n}/{len(todo)}] {item['id']} {result.get('error') or str(len(result['detections'])) + ' named, ' + str(len(result['unlabeled'])) + ' unnamed'}", flush=True)

    with open(os.path.join(out_dir, "done.txt"), "w") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S"))


main()
