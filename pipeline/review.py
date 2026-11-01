
import base64
import json

import cv2

from pathlib import Path

from boost import LABELS
from ingest import load_json
from onion import POINTING_THRESHOLD

THUMB_WIDTH = 420
THUMB_QUALITY = 72


# -----------------------------------------
# THUMBNAILS
# -----------------------------------------

def to_uri(image, width=THUMB_WIDTH):

    scale = width / image.shape[1]

    small = cv2.resize(image, (width, int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    ok, buffer = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, THUMB_QUALITY])

    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def file_thumb(path, width=THUMB_WIDTH):

    image = cv2.imread(str(path)) if path and Path(path).exists() else None

    return to_uri(image, width) if image is not None else None


def video_thumb(cap, t):

    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)

    ok, frame = cap.read()

    return to_uri(frame) if ok else None


# -----------------------------------------
# THE PAGE
# -----------------------------------------

def build(manifest_path="processed/ingest.json"):

    manifest_path = Path(manifest_path)
    base = manifest_path.parent

    manifest = load_json(manifest_path)
    onion = load_json(base / "onion.json")
    plan = load_json(base / "plan.json")
    masks = load_json(base / "masks" / "masks.json")

    agents_path = base / "agent" / "summary.json"
    agents = load_json(agents_path) if agents_path.exists() else None

    # Results from an earlier chunking do not belong to these segments; leave them out until the agents run again
    if agents is not None and [r["id"] for r in agents["segments"]] != [s["id"] for s in onion["segments"]]:
        agents = None

    cap = cv2.VideoCapture(manifest["video"]["path"])

    plan_by_id = {s["id"]: s for s in plan["segments"]}
    onion_by_id = {s["id"]: s for s in onion["segments"]}

    def keyframe_thumb(seg_id, n, t):

        # The cut keyframe if it exists, otherwise read that moment from the video
        return file_thumb(base / "masks" / "keyframes" / f"s{seg_id:03d}_k{n}.jpg") or video_thumb(cap, t)

    data = {
        "video": {
            "name": Path(manifest["source"]["path"]).name,
            "duration": manifest["video"]["duration"],
            "key": f"{manifest['source']['size']}-{manifest['source']['mtime']}"
        },
        "context": manifest["context"],
        "transcript": [{"t": t["start"], "text": t["text"]} for t in manifest["transcript"]],
        "onion": {
            "params": onion["params"],
            "llm_layer": onion["llm_layer"],
            "threshold": onion["params"].get("pointing_threshold", POINTING_THRESHOLD),
            "segments": [
                {
                    "id": s["id"], "start": s["start"], "end": s["end"], "text": s["text"], "weight": s["weight"],
                    "pointing": s["pointing"], "likely": s["likely_pointing"], "border": s["border_score"], "signals": s.get("border_signals", {}),
                    "cues": [{**c, "key": f"{c['start']:.2f}", "thumb": video_thumb(cap, (c["start"] + c["end"]) / 2)} for c in s["cues"]],
                    "thumbs": [
                        {"src": keyframe_thumb(s["id"], n, k["t"]), "cap": f"{int(k['t']) // 60}:{int(k['t']) % 60:02d}"}
                        for n, k in enumerate(plan_by_id[s["id"]]["keyframes"][:2])
                    ]
                }
                for s in onion["segments"]
            ]
        },
        "plan": {
            "labels": LABELS,
            "windows": plan["windows"],
            "model": plan["model"],
            "segments": [
                {
                    **{k: s[k] for k in ("id", "start", "end", "mode", "focus", "situation", "cuts")},
                    "keyframes": [
                        {"t": k["t"], "reason": k["reason"], "thumb": keyframe_thumb(s["id"], n, k["t"])}
                        for n, k in enumerate(s["keyframes"])
                    ]
                }
                for s in plan["segments"]
            ]
        },
        "masks": {
            "prompts": masks["prompts"],
            "segments": [
                {
                    "id": s["id"], "mode": s["mode"], "start": onion_by_id[s["id"]]["start"],
                    "skipped": s.get("skipped"), "objects": s["objects"],
                    "keyframes": [
                        {
                            "t": k["t"], "thumb": file_thumb(base / "masks" / k["overlay"]) if k["overlay"] else None,
                            "missed": k["missed"], "items": k["items"], "text": k["text"],
                            "unlabeled": [
                                {"caption": u.get("caption"), "score": u["score"], "thumb": file_thumb(base / "masks" / u["crop"], 220)}
                                for u in k["unlabeled"]
                            ],
                            "crops": [
                                {"prompt": d["prompt"], "caption": d.get("caption"), "thumb": file_thumb(base / "masks" / d["crop"], 220)}
                                for d in k["detections"] if d.get("crop")
                            ]
                        }
                        for k in s["keyframes"]
                    ]
                }
                for s in masks["segments"]
            ]
        },
        "agents": None if agents is None else {
            "final": agents["final"],
            "chapters": agents["chapters"],
            "segments": [
                {
                    "id": r["id"], "start": onion_by_id[r["id"]]["start"], "summary": r["summary"],
                    "moments": [{k: m.get(k) for k in ("at", "said", "what")} | {"target": (m.get("target") or {}).get("name")} for m in r["moments"]]
                }
                for r in agents["segments"]
            ]
        }
    }

    cap.release()

    template = (Path(__file__).with_name("review.html")).read_text(encoding="utf-8")

    out = base / "review"
    out.mkdir(exist_ok=True)

    # </ inside the data must not close the script tag
    page = template.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))

    (out / "index.html").write_text(page, encoding="utf-8")

    return out / "index.html"
