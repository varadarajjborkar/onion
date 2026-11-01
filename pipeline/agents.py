
import json
import re
import time

from pathlib import Path
from typing import TypedDict

import cv2

from langgraph.graph import END, START, StateGraph

import llm_env
import masking

from ingest import load_json, save_json
from memory import COMPRESS_PROMPT, Memory
from video_processor import clock


# -----------------------------------------
# CONFIGURATION
# -----------------------------------------

# The LLM or the internet can be gone for a while; wait this long before giving up (the run resumes anyway)
PATIENCE_S = 2 * 3600
RETRY_S = 30

# Raw details are low weight: the agent may reach for a tool this many times per segment, no more
MAX_TOOL_CALLS = 2

# Images shown to the agent per segment: 1 for a calm segment, up to this for the most important
MAX_IMAGES = 6

# Things listed to the agent per segment (named, then text, then unnamed regions)
MAX_THINGS = 16

CHAPTER_SEGMENTS = 8
FINAL_FAN_IN = 8

OBSERVER_PROMPT = (
    "You are describing one segment of a video for a running summary.\n\n"
    "{core}\n\n"
    "THIS SEGMENT\n{evidence}\n\n"
    "THINGS SEEN IN THIS SEGMENT (use these ids)\n{things}\n\n"
    "Images: {images}\n\n"
    "Use the evidence above first; it is already checked. Tell only what is new in this segment and do not repeat "
    "the story so far. Something may be shown or pointed at in many ways: a hand gesture, an object held up, an animation, "
    "a meme, a document or text on screen. For each moment where the speech refers to something visible, say what it is "
    "in as much detail as the evidence allows (how it looks, what it says), and give its id from the list, or null if it is not listed. "
    "If the images and the speech are not enough to tell, say so. Do not use em dashes.\n"
    'Reply with only a JSON object: {{"summary": "two to four sentences", "events": ["short items"], '
    '"objects": [{{"name": "...", "detail": "..."}}], '
    '"moments": [{{"at": "m:ss", "said": "short quote", "target_id": "T1 or null", "what": "what is shown or pointed at, in detail"}}]}}\n'
    "{tools}"
)

TOOLS_DOC = (
    'Only if something important is missing you may reply with {{"tool": "name", "args": {{...}}}} instead, at most {left} more time(s). Tools:\n'
    '- get_transcript {{"start": seconds, "end": seconds}}: the speech in that range\n'
    '- search_memory {{"query": "words"}}: earlier segments about a topic\n'
    '- get_details {{"segment": id}}: raw numbers for a segment (cue scores, camera, cuts)\n'
    '- look_for {{"prompts": ["noun", ...]}}: ask the GPU to mask new things in this segment, slow, use rarely'
)

CHAPTER_PROMPT = (
    "Below are consecutive parts of one video, each with its time and how important it is (0 to 1). "
    "Write a short title and a summary of three to five sentences for all of them together. "
    "Spend more words on the more important parts and say what was shown or pointed at. Do not use em dashes. "
    'Reply with only a JSON object: {{"title": "...", "summary": "..."}}\n\n'
)

FINAL_PROMPT = (
    "Write the final summary of a video from the notes below. Use these sections: "
    "Overview (three or four sentences), Timeline (one line per part with its times), "
    "Key moments (each place where something is shown or pointed at, with the time and what it was in detail: how it looks, what it says, whether it is an animation, a meme, a document or an object), Objects and people. "
    "Do not invent anything that is not in the notes. Do not use em dashes.\n\n"
    "VIDEO: {video}\n\nPARTS:\n{parts}\n\nSHOWN OR POINTED AT:\n{pointing}\n\nOBJECTS AND PEOPLE:\n{objects}"
)


# -----------------------------------------
# HELPERS
# -----------------------------------------

def ask_patient(prompt, images=(), model=None):

    waited = 0

    while True:

        try:
            return llm_env.describe(list(images), prompt, model=model)

        except (RuntimeError, OSError) as err:

            # A rejected request (wrong key, bad input) will not get better by waiting
            code = re.search(r"\((\d{3})\)", str(err))

            if code and 400 <= int(code.group(1)) < 500 and int(code.group(1)) not in (408, 429):
                raise

            if waited >= PATIENCE_S:
                raise

            time.sleep(RETRY_S)
            waited += RETRY_S


def parse_object(text):

    match = re.search(r"\{.*\}", text, re.S)

    try:
        return json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None


def clean_observation(obj, reply):

    if not isinstance(obj, dict) or "summary" not in obj:
        return {"summary": reply.strip()[:600], "events": [], "objects": [], "moments": []}

    moments = []

    for m in obj.get("moments", []) if isinstance(obj.get("moments"), list) else []:

        if isinstance(m, dict) and str(m.get("what", "")).strip():
            moments.append({
                "at": str(m.get("at", "")),
                "said": str(m.get("said", "")),
                "target_id": str(m["target_id"]) if m.get("target_id") else None,
                "what": str(m["what"]).strip()
            })

    return {
        "summary": str(obj["summary"]).strip(),
        "events": [str(e) for e in obj.get("events", []) if e],
        "objects": [o for o in obj.get("objects", []) if isinstance(o, dict)],
        "moments": moments
    }


# -----------------------------------------
# THE RUN: agents, tools and memory over one video
# -----------------------------------------

class State(TypedDict):

    i: int


class Run:

    def __init__(self, manifest_path="processed/ingest.json", model=None, use_gpu=True):

        self.dir = Path(manifest_path).parent
        self.out = self.dir / "agent"
        self.out.mkdir(exist_ok=True)

        self.model = model
        self.use_gpu = use_gpu

        self.manifest = load_json(Path(manifest_path))
        self.onion = {s["id"]: s for s in load_json(self.dir / "onion.json")["segments"]}
        self.plan = {s["id"]: s for s in load_json(self.dir / "plan.json")["segments"]}
        self.masks = {s["id"]: s for s in load_json(self.dir / "masks" / "masks.json")["segments"]}

        self.order = sorted(self.onion)

        self.stamp = [(self.dir / n).stat().st_mtime_ns for n in ("ingest.json", "onion.json", "plan.json", "masks/masks.json")]

        self.memory = Memory(self.manifest["context"].get("summary", ""))
        self.results = []
        self.chapters = []
        self.final = None

        self.extra_images = []

    # -- checkpoint: every finished segment is on disk ------------------

    def load(self):

        path = self.out / "state.json"

        if not path.exists():
            return

        saved = load_json(path)

        # Different inputs than the saved run: start clean
        if saved["stamp"] != self.stamp:
            return

        self.memory = Memory.from_dict(saved["memory"])
        self.results = saved["results"]
        self.chapters = saved["chapters"]
        self.final = saved["final"]

    def save(self):

        save_json(self.out / "state.json", {
            "stamp": self.stamp, "memory": self.memory.to_dict(),
            "results": self.results, "chapters": self.chapters, "final": self.final
        })

    # -- tools -----------------------------------------------------------

    def run_tool(self, name, args, seg_id):

        try:

            if name == "get_transcript":

                lo, hi = float(args["start"]), float(args["end"])
                lines = [f"[{clock(t['start'])}] {t['text']}" for t in self.manifest["transcript"] if t["start"] < hi and t["end"] > lo]

                return "\n".join(lines) or "no speech in that range"

            if name == "search_memory":

                hits = self.memory.search(str(args["query"]))

                return "\n".join(f"[{clock(h['start'])}] {h['summary']}" for h in hits) or "nothing found"

            if name == "get_details":

                sid = int(args.get("segment", seg_id))

                return json.dumps({"onion": {k: self.onion[sid][k] for k in ("weight", "pointing", "cues")}, "plan": self.plan[sid]}, default=str)[:1500]

            if name == "look_for":

                if not self.use_gpu:
                    return "the GPU is not available in this run"

                prompts = [str(p) for p in args["prompts"]][:4]

                found, out_dir = masking.look_for(self.dir / "masks", self.plan[seg_id], prompts)

                for k in found["keyframes"]:
                    if k["overlay"]:
                        self.extra_images.append(out_dir / k["overlay"])

                lines = [
                    f"{d['prompt']} at {clock(k['t'])}: score {d['score']:.2f}, covers {d['area_frac'] * 100:.1f}% of the frame"
                    for k in found["keyframes"] for d in k["detections"][:1]
                ]

                return "\n".join(lines) + f"\nnot found: {sorted({m for k in found['keyframes'] for m in k['missed']})}" if lines else f"nothing found for {prompts}"

            return f"unknown tool {name}"

        except (masking.LinkDown, TimeoutError):
            return "the GPU laptop is not reachable right now"

        except (KeyError, ValueError, TypeError, RuntimeError) as err:
            return f"the tool failed: {type(err).__name__}: {err}"

    # -- evidence for one segment ------------------------------------------

    def things_seen(self, seg_id):

        mk = self.masks[seg_id]

        masked_items, other_items, texts, unlabeled = [], [], [], []

        keys = set()

        def fresh(key):
            if key in keys:
                return False
            keys.add(key)
            return True

        for k in mk["keyframes"]:

            best = {}

            for d in k["detections"]:
                best.setdefault(d["prompt"], d)

            for it in k["items"]:

                hit = best.get(it["name"])

                if fresh(("item", it["name"])):

                    detail = it["detail"] or (hit or {}).get("caption") or ""

                    row = {"kind": it["kind"], "name": it["name"], "detail": detail, "t": k["t"],
                           "crop": (hit or {}).get("crop"), "masked": f"masked, score {hit['score']:.2f}" if hit else "not masked"}

                    (masked_items if hit else other_items).append(row)

            for d in k["detections"]:
                if fresh(("item", d["prompt"])):
                    masked_items.append({"kind": "object", "name": d["prompt"], "detail": d.get("caption") or "", "t": k["t"],
                                         "crop": d.get("crop"), "masked": f"masked, score {d['score']:.2f}"})

            for text in k["text"]:
                if fresh(("text", text.lower())):
                    texts.append({"kind": "text", "name": f'"{text}"', "detail": "", "t": k["t"], "crop": None, "masked": "read from the picture"})

            for u in k["unlabeled"]:
                if u.get("caption") and fresh(("unlabeled", u["caption"][:40].lower())):
                    unlabeled.append({"kind": "unnamed region", "name": "region without a name", "detail": u["caption"], "t": k["t"],
                                      "crop": u["crop"], "masked": f"segmented, score {u['score']:.2f}"})

        things = (masked_items + other_items + texts + unlabeled)[:MAX_THINGS]

        for n, t in enumerate(things, 1):
            t["id"] = f"T{n}"

        return things

    def evidence(self, seg_id):

        on, pl, mk = self.onion[seg_id], self.plan[seg_id], self.masks[seg_id]

        cues = "; ".join(f'"{c["text"]}" at {clock(c["start"])} ({c["score"]:.2f})' for c in on["cues"] if c["score"] >= 0.3) or "none"

        mostly = max(pl["situation"], key=pl["situation"].get)

        if mk.get("skipped"):
            seen = f"not run ({mk['skipped']})"
        else:
            seen = "; ".join(
                f"{p} in {o['keyframes_found']}/{len(mk['keyframes'])} keyframes, score {o['best_score']:.2f}, covers {o['mean_area_frac'] * 100:.1f}% of the frame"
                for p, o in mk["objects"].items()
            ) or "nothing found"

        return (
            f"Segment {seg_id}, {clock(on['start'])} to {clock(on['end'])}, importance {on['weight']:.2f}\n"
            f"Speech: {on['text']}\n"
            f"Possible pointing or showing in the speech: {cues}\n"
            f"Picture: mostly {mostly}, {len(pl['cuts'])} cut(s), processing mode {pl['mode']}\n"
            f"Objects found by segmentation: {seen}"
        )

    def images_for(self, seg_id):

        pl, mk = self.plan[seg_id], self.masks[seg_id]

        budget = 1 + round(pl["focus"] * (MAX_IMAGES - 1))

        paths, notes = [], []

        keyframes = sorted(mk["keyframes"], key=lambda k: -len(k["detections"]))

        for k in keyframes:
            if k["overlay"]:
                paths.append(self.dir / "masks" / k["overlay"])
                notes.append(f"keyframe at {clock(k['t'])} with colored masks and labels drawn on it")

        for k in keyframes:
            for d in k["detections"]:
                if d.get("crop"):
                    paths.append(self.dir / "masks" / d["crop"])
                    notes.append(f"close-up of '{d['prompt']}' at {clock(k['t'])}")

        # Segments the GPU skipped still get plain keyframes
        if not paths:

            cap = cv2.VideoCapture(self.manifest["video"]["path"])
            key_dir = self.dir / "masks" / "keyframes"
            key_dir.mkdir(exist_ok=True)

            for n, key in enumerate(pl["keyframes"]):
                path = key_dir / f"s{seg_id:03d}_k{n}.jpg"
                masking.save_keyframe(cap, key["t"], path)
                paths.append(path)
                notes.append(f"plain keyframe at {clock(key['t'])}")

            cap.release()

        return paths[:budget], notes[:budget]

    # -- agents ------------------------------------------------------------

    def observe(self, seg_id):

        paths, notes = self.images_for(seg_id)

        things = self.things_seen(seg_id)

        things_text = "\n".join(
            f"{t['id']} [{t['kind']}] {t['name']}" + (f": {t['detail']}" if t["detail"] else "") + f" (seen at {clock(t['t'])}, {t['masked']})"
            for t in things
        ) or "nothing listed"

        self.extra_images = []

        tool_results = []

        reply, obj = "", None

        for call in range(MAX_TOOL_CALLS + 1):

            left = MAX_TOOL_CALLS - call

            prompt = OBSERVER_PROMPT.format(
                core=self.memory.core_text(),
                evidence=self.evidence(seg_id),
                things=things_text,
                images="; ".join(f"{n + 1}. {note}" for n, note in enumerate(notes)) or "none",
                tools=TOOLS_DOC.format(left=left) if left else "No tools are left, answer now."
            )

            if tool_results:
                prompt += "\n\nTOOL RESULTS SO FAR:\n" + "\n".join(tool_results)

            images = [p.read_bytes() for p in paths + self.extra_images[:2]]

            reply = ask_patient(prompt, images, self.model)

            obj = parse_object(reply)

            if left and isinstance(obj, dict) and "tool" in obj:
                tool_results.append(f"{obj['tool']} -> " + self.run_tool(str(obj["tool"]), obj.get("args") or {}, seg_id))
                continue

            break

        result = clean_observation(obj, reply)

        by_id = {t["id"]: t for t in things}

        # Tie each moment to the thing it points at, with that thing's own details
        for m in result["moments"]:
            m["target"] = by_id.get(m["target_id"])

        result["id"] = seg_id
        result["tools_used"] = len(tool_results)

        return result

    def condense(self, text, limit):

        return ask_patient(COMPRESS_PROMPT.format(limit=limit) + text, (), self.model).strip()

    def chapter(self, parts, label):

        reply = ask_patient(CHAPTER_PROMPT + "\n".join(parts), (), self.model)

        obj = parse_object(reply) or {}

        return {"title": str(obj.get("title", label)), "summary": str(obj.get("summary", reply.strip()))}

    # -- the graph -------------------------------------------------------

    def n_observe(self, state):

        seg_id = self.order[state["i"]]

        result = self.observe(seg_id)

        on = self.onion[seg_id]

        self.results.append(result)
        self.memory.add(seg_id, on["start"], on["end"], on["weight"], result)

        if self.memory.needs_compression():
            self.memory.compress(self.condense)

        self.save()

        return {"i": state["i"] + 1}

    def n_chapters(self, state):

        if self.chapters:
            return {}

        level = []

        for k in range(0, len(self.order), CHAPTER_SEGMENTS):

            ids = self.order[k:k + CHAPTER_SEGMENTS]

            parts = [
                f"[{clock(self.onion[i]['start'])}] (importance {self.onion[i]['weight']:.2f}) {r['summary']}"
                for i, r in zip(ids, self.results[k:k + CHAPTER_SEGMENTS])
            ]

            chap = self.chapter(parts, f"Part {len(level) + 1}")
            chap.update(start=self.onion[ids[0]]["start"], end=self.onion[ids[-1]]["end"])

            level.append(chap)

        # Very long videos: fold the parts again until few enough remain
        while len(level) > FINAL_FAN_IN:

            merged = []

            for k in range(0, len(level), FINAL_FAN_IN):

                group = level[k:k + FINAL_FAN_IN]

                parts = [f"[{clock(c['start'])}] (importance 0.5) {c['title']}: {c['summary']}" for c in group]

                chap = self.chapter(parts, f"Part {len(merged) + 1}")
                chap.update(start=group[0]["start"], end=group[-1]["end"])

                merged.append(chap)

            level = merged

        self.chapters = level
        self.save()

        return {}

    def n_final(self, state):

        if self.final:
            return {}

        parts = "\n".join(f"{clock(c['start'])} to {clock(c['end'])}  {c['title']}: {c['summary']}" for c in self.chapters)

        pointing = "\n".join(
            f"[{m['at'] or clock(self.onion[r['id']]['start'])}] said \"{m['said']}\": {m['what']}"
            + (f" (item {m['target']['id']}, {m['target']['kind']}: {m['target']['name']})" if m.get("target") else "")
            for r in self.results for m in r["moments"]
        ) or "none"

        objects = "\n".join(f"{n}: {e['detail']}" for n, e in sorted(self.memory.entities.items(), key=lambda x: -x[1]["weight"])) or "none"

        text = ask_patient(
            FINAL_PROMPT.format(video=self.memory.video, parts=parts, pointing=pointing, objects=objects), (), self.model
        )

        self.final = text.strip()
        self.save()

        return {}

    def route(self, state):

        return "observe" if state["i"] < len(self.order) else "chapters"

    def build(self):

        graph = StateGraph(State)

        graph.add_node("observe", self.n_observe)
        graph.add_node("chapters", self.n_chapters)
        graph.add_node("final", self.n_final)

        graph.add_conditional_edges(START, self.route, ["observe", "chapters"])
        graph.add_conditional_edges("observe", self.route, ["observe", "chapters"])
        graph.add_edge("chapters", "final")
        graph.add_edge("final", END)

        return graph.compile()

    def run(self, force=False, on_step=None):

        if not force:
            self.load()

        graph = self.build()

        # recursion_limit counts every node call, and there is one per segment
        config = {"recursion_limit": len(self.order) + 20}

        for event in graph.stream({"i": len(self.results)}, config, stream_mode="updates"):

            if on_step and "observe" in event:
                on_step(event["observe"]["i"], len(self.order))

        summary = {"final": self.final, "chapters": self.chapters, "segments": self.results, "entities": self.memory.entities}

        save_json(self.out / "summary.json", summary)
        (self.out / "summary.txt").write_text(self.final + "\n", encoding="utf-8")

        return summary
