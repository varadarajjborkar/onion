
import hashlib
import json
import re
import subprocess
import time

import cv2

from collections import Counter
from pathlib import Path

import discover

from ingest import ask, load_json, save_json
from onion import STOPWORDS, norm
from video_processor import Spinner, clock


# -----------------------------------------
# CONFIGURATION
# -----------------------------------------

HOST = "alienware"
REMOTE_TOOLS = "C:/sam3/keyframes"
REMOTE_JOBS = "C:/jobs/keyframes"

SSH_OPTIONS = ["-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2"]

# A cut in power or internet is expected: every remote call is retried with growing waits
SSH_TRIES = 6
POLL_S = 20
MAX_WAIT_S = 6 * 3600

LINK_ERRORS = re.compile(
    r"timed out|connection (closed|refused|reset)|no route|could not resolve|broken pipe|network is|host is down|kex_exchange|Connection to",
    re.I
)

# How sure SAM3 must be before it reports an object
DETECTION_THRESHOLD = 0.4

# Segments with more focus get more text prompts, and the person is always added when something is pointed at
PROMPT_MIN = 2
PROMPT_MAX = 6

ZOOM_MODES = {"mask_and_zoom", "follow_to_target"}

# A calm segment with little speech about showing things is not worth GPU time
SKIP_BELOW_POINTING = 0.3

# Names sent to SAM3 for one picture: from the speech first, then what the vision LLM saw
PROMPTS_PER_IMAGE = 10

# Regions kept per picture that no name matched, found by segmenting everything
MAX_UNLABELED = 4

VISUAL_PROMPT = (
    "Below is a list of things mentioned in the speech of a video. "
    "Reply with only a JSON list of the items that are physical objects or things that could be seen in a camera image "
    "and found by an image segmentation model, each as a short concrete noun phrase of one to three words. "
    "Leave out names of people, brands, apps, websites and abstract ideas. "
    "Keep the wording of the list where you can.\n\nList: "
)


class LinkDown(Exception):
    pass


# -----------------------------------------
# REMOTE CALLS (ssh and scp that survive the link dropping)
# -----------------------------------------

def remote(args):

    for attempt in range(1, SSH_TRIES + 1):

        done = subprocess.run(args, capture_output=True, text=True)

        if done.returncode == 0:
            return done.stdout

        if not (done.returncode == 255 or LINK_ERRORS.search(done.stderr)):
            raise RuntimeError(f"{args[0]} failed: {done.stderr.strip()[:300]}")

        if attempt < SSH_TRIES:
            time.sleep(min(2 ** attempt, 30))

    raise LinkDown(done.stderr.strip()[:200])


def ssh(command):

    return remote(["ssh", *SSH_OPTIONS, HOST, command])


def scp(*paths, recursive=False):

    return remote(["scp", "-q", *(["-r"] if recursive else []), *SSH_OPTIONS, *map(str, paths)])


def powershell(script, **named):

    flags = " ".join(f"-{k} {v}" for k, v in named.items())

    return ssh(f"powershell -NoProfile -ExecutionPolicy Bypass -File {REMOTE_TOOLS}/{script} {flags}")


# -----------------------------------------
# PROMPTS: which things to look for in which segment
# -----------------------------------------

def stem(word):

    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def visual_prompts(context, cache_path, model=None):

    entities = context.get("entities", [])

    if cache_path.exists():
        saved = load_json(cache_path)
        if saved["entities"] == entities:
            return saved["prompts"]

    try:
        reply = ask(VISUAL_PROMPT + json.dumps(entities), model)
        match = re.search(r"\[.*\]", reply, re.S)
        prompts = [str(p).strip() for p in json.loads(match.group(0)) if str(p).strip()]

    except (RuntimeError, OSError, AttributeError, json.JSONDecodeError):
        # No LLM: use the entities as they are
        return entities

    save_json(cache_path, {"entities": entities, "prompts": prompts})

    return prompts


def segment_prompts(text, prompts, mode, focus):

    seen = Counter(stem(norm(w)) for w in text.split())

    scored = []

    for p in prompts:

        tokens = [stem(norm(w)) for w in p.split() if len(norm(w)) >= 3 and norm(w) not in STOPWORDS]
        count = sum(seen[t] for t in tokens)

        if count:
            scored.append((count, p))

    chosen = [p for _, p in sorted(scored, key=lambda x: -x[0])]

    budget = PROMPT_MIN + round(focus * (PROMPT_MAX - PROMPT_MIN))

    if mode in ZOOM_MODES:
        chosen = ["person"] + [p for p in chosen if p != "person"]

    return chosen[:budget]


# -----------------------------------------
# JOB: keyframe images plus what to look for in each
# -----------------------------------------

def save_keyframe(cap, t, path):

    if path.exists():
        return

    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)

    ok, frame = cap.read()

    if not ok:
        raise ValueError(f"Could not read the frame at {t:.2f}s.")

    tmp = path.with_name(path.stem + ".tmp.jpg")

    cv2.imwrite(str(tmp), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    tmp.replace(path)


def cut_keyframes(manifest, onion_result, plan_result, prompts, out_dir):

    key_dir = out_dir / "keyframes"
    key_dir.mkdir(parents=True, exist_ok=True)

    on = {s["id"]: s for s in onion_result["segments"]}

    cap = cv2.VideoCapture(manifest["video"]["path"])

    images, skipped = [], {}

    for seg in plan_result["segments"]:

        if seg["mode"] == "light" and on[seg["id"]]["pointing"] < SKIP_BELOW_POINTING:
            skipped[seg["id"]] = "nothing to look for"
            continue

        base = segment_prompts(on[seg["id"]]["text"], prompts, seg["mode"], seg["focus"])

        for n, key in enumerate(seg["keyframes"]):

            name = f"s{seg['id']:03d}_k{n}"

            save_keyframe(cap, key["t"], key_dir / f"{name}.jpg")

            images.append({"id": name, "file": f"{name}.jpg", "segment": seg["id"], "t": key["t"], "prompts": base, "zoom": seg["mode"] in ZOOM_MODES})

    cap.release()

    return images, skipped


def build_job(images, seen, out_dir):

    # Each picture is searched for what the speech names, then for what the vision LLM saw in it
    full = []

    for img in images:

        names = list(img["prompts"])

        for item in seen.get(img["id"], {}).get("items", []):
            if item["kind"] != "text" and item["name"] not in names:
                names.append(item["name"])

        full.append({**img, "prompts": names[:PROMPTS_PER_IMAGE]})

    spec = {"threshold": DETECTION_THRESHOLD, "everything": True, "max_unlabeled": MAX_UNLABELED, "images": full}

    job_id = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]

    save_json(out_dir / "jobs.json", spec)

    return job_id, spec


# -----------------------------------------
# THE REMOTE GPU RUN
# -----------------------------------------

class RemoteSam:

    def __init__(self, job_id, spec, out_dir, key_dir=None):

        self.job_id = job_id
        self.spec = spec
        self.out_dir = out_dir
        self.key_dir = key_dir or out_dir / "keyframes"
        self.remote_dir = f"{REMOTE_JOBS}/{job_id}"

    def install(self):

        ssh(f'powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path {REMOTE_TOOLS} | Out-Null"')

        tools = sorted(Path(__file__).with_name("remote").glob("*"))

        scp(*tools, f"{HOST}:{REMOTE_TOOLS}/")

    def state(self):

        out = powershell("state.ps1", JobDir=self.remote_dir)

        info = {}

        for line in out.splitlines():
            key, _, value = line.partition(" ")
            info[key] = value.strip()

        return info.get("TASK", "none"), info.get("DONE") == "True", set(filter(None, info.get("HAVE", "").split(",")))

    def upload(self):

        ssh(f'powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path {self.remote_dir}/in | Out-Null"')

        _, _, have = self.state()

        files = sorted(f for f in self.key_dir.glob("*.jpg") if f.name in {i["file"] for i in self.spec["images"]} and f.name not in have)

        for k in range(0, len(files), 20):
            scp(*files[k:k + 20], f"{HOST}:{self.remote_dir}/in/")

        scp(self.out_dir / "jobs.json", f"{HOST}:{self.remote_dir}/jobs.json")

    def start(self):

        return powershell("launch.ps1", JobDir=self.remote_dir).strip()

    def pull_results(self):

        try:
            scp(f"{HOST}:{self.remote_dir}/out/results.jsonl", self.out_dir / "results.jsonl")
        except RuntimeError:
            # Not written yet
            return 0

        return sum(1 for _ in open(self.out_dir / "results.jsonl", encoding="utf-8"))

    def wait(self, spinner, max_wait=MAX_WAIT_S):

        total = len(self.spec["images"])

        deadline = time.time() + max_wait

        idle = 0

        while time.time() < deadline:

            try:
                task, done, _ = self.state()
                count = self.pull_results()

            except LinkDown:
                # Power or internet is out: keep trying, the job keeps running on the laptop
                spinner.text = "Link to the GPU laptop is down, retrying"
                time.sleep(POLL_S)
                continue

            spinner.text = f"Masking keyframes on the GPU laptop {count}/{total}"

            if done:
                return

            # The task is gone without finishing (power cut, reboot): start it again, it resumes
            idle = idle + 1 if task != "Running" else 0

            if idle >= 2:
                self.start()
                idle = 0

            time.sleep(POLL_S)

        raise TimeoutError("The masking job did not finish in time; run again to resume.")

    def fetch(self):

        for folder in ("overlays", "crops"):
            scp(f"{HOST}:{self.remote_dir}/out/{folder}", self.out_dir, recursive=True)

        self.pull_results()


# -----------------------------------------
# RESULT: what was found, per segment and keyframe
# -----------------------------------------

def assemble(plan_result, spec, skipped, out_dir, seen=None, captions=None):

    seen, captions = seen or {}, captions or {}

    by_id = {}

    for line in open(out_dir / "results.jsonl", encoding="utf-8"):
        try:
            row = json.loads(line)
            by_id[row["id"]] = row
        except (json.JSONDecodeError, KeyError):
            pass

    segments = []

    for seg in plan_result["segments"]:

        if seg["id"] in skipped:
            segments.append({"id": seg["id"], "mode": seg["mode"], "skipped": skipped[seg["id"]], "keyframes": [], "objects": {}})
            continue

        keyframes, found = [], {}

        for item in [i for i in spec["images"] if i["segment"] == seg["id"]]:

            row = by_id.get(item["id"], {"error": "no result"})

            detections = [{**d, "caption": captions.get(d["crop"])} for d in row.get("detections", [])]
            unlabeled = [{**u, "caption": captions.get(u["crop"])} for u in row.get("unlabeled", [])]

            keyframes.append({
                "t": item["t"], "image": f"keyframes/{item['file']}",
                "overlay": f"overlays/{item['id']}.jpg" if "overlay" in row else None,
                "items": seen.get(item["id"], {}).get("items", []), "text": seen.get(item["id"], {}).get("text", []),
                "detections": detections, "unlabeled": unlabeled, "missed": row.get("missed", item["prompts"]),
                "error": row.get("error")
            })

            for d in detections:
                found.setdefault(d["prompt"], []).append(d)

        # Per object across the keyframes: how often and how surely it was found
        objects = {
            p: {
                "keyframes_found": len({k["t"] for k in keyframes for d in k["detections"] if d["prompt"] == p}),
                "best_score": max(d["score"] for d in ds),
                "mean_area_frac": round(sum(d["area_frac"] for d in ds) / len(ds), 5)
            }
            for p, ds in found.items()
        }

        segments.append({"id": seg["id"], "mode": seg["mode"], "keyframes": keyframes, "objects": objects})

    return {"segments": segments}


def mask(manifest_path="processed/ingest.json", onion_path="processed/onion.json", plan_path="processed/plan.json", model=None):

    manifest_path = Path(manifest_path)
    out_dir = manifest_path.parent / "masks"
    out_dir.mkdir(exist_ok=True)

    manifest = load_json(manifest_path)
    onion_result = load_json(Path(onion_path))
    plan_result = load_json(Path(plan_path))

    prompts = visual_prompts(manifest["context"], out_dir / "prompts.json", model)

    with Spinner("Cutting keyframes"):
        images, skipped = cut_keyframes(manifest, onion_result, plan_result, prompts, out_dir)

    # What is actually in each picture, in words, including animations, memes and text
    with Spinner(f"Looking at {len(images)} keyframes"):
        seen, unseen = discover.discover({i["id"]: out_dir / "keyframes" / i["file"] for i in images}, out_dir / "discovery.json", model)

    job_id, spec = build_job(images, seen, out_dir)

    if spec["images"]:

        sam = RemoteSam(job_id, spec, out_dir)

        with Spinner("Preparing the GPU laptop") as spinner:

            sam.install()
            sam.upload()
            sam.start()

            sam.wait(spinner)

            spinner.text = "Fetching masks"
            sam.fetch()

    else:
        (out_dir / "results.jsonl").write_text("", encoding="utf-8")

    # A caption for every cropped thing, so the details of what was shown are in words
    crops = {}

    for line in open(out_dir / "results.jsonl", encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        for d in row.get("detections", []) + row.get("unlabeled", []):
            if d.get("crop"):
                crops[d["crop"]] = out_dir / d["crop"]

    with Spinner(f"Describing {len(crops)} crops"):
        captions, uncaptioned = discover.caption(crops, out_dir / "captions.json", model)

    result = {
        "job": job_id, "prompts": prompts, "unseen_keyframes": unseen, "uncaptioned_crops": uncaptioned,
        **assemble(plan_result, spec, skipped, out_dir, seen, captions)
    }

    save_json(out_dir / "masks.json", result)

    return result


# -----------------------------------------
# ON DEMAND: an agent asks the GPU to look for something new in one segment
# -----------------------------------------

LOOK_FOR_WAIT_S = 600


class Quiet:

    text = ""


def look_for(masks_dir, plan_segment, prompts):

    masks_dir = Path(masks_dir)

    images = [
        {"id": f"x{plan_segment['id']:03d}_k{n}", "file": f"s{plan_segment['id']:03d}_k{n}.jpg", "segment": plan_segment["id"],
         "t": key["t"], "prompts": prompts, "zoom": True}
        for n, key in enumerate(plan_segment["keyframes"])
    ]

    spec = {"threshold": DETECTION_THRESHOLD, "everything": False, "images": images}

    job_id = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]

    out_dir = masks_dir / "extra" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(out_dir / "jobs.json", spec)

    sam = RemoteSam(job_id, spec, out_dir, key_dir=masks_dir / "keyframes")

    # The extra job's images are the main job's keyframes, sent under their own names
    sam.install()
    sam.upload()
    sam.start()
    sam.wait(Quiet(), max_wait=LOOK_FOR_WAIT_S)
    sam.fetch()

    return assemble({"segments": [plan_segment]}, spec, {}, out_dir)["segments"][0], out_dir
