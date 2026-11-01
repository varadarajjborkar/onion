
import re

from video_processor import clock


# -----------------------------------------
# CONFIGURATION
# -----------------------------------------

# Core memory is what every agent call sees, so it has a hard size limit
CORE_STORY_CHARS = 3000

# The old part of the story is squeezed down to about this size
ARCHIVE_CHARS = 1200

ENTITY_MAX = 40
ENTITY_DETAIL_CHARS = 240

RECALL_HITS = 3

COMPRESS_PROMPT = (
    "Below are the running notes of a video so far, an older summary and then the entries that follow it. "
    "Rewrite all of it as one shorter paragraph of at most {limit} characters. "
    "Keep names, numbers, what was shown or pointed at, and anything marked important in full. "
    "Drop repetition and small talk. Reply with only the paragraph.\n\n"
)


# -----------------------------------------
# MEMORY IN THREE TIERS
# core    : always in the prompt (video context, story so far, objects seen)
# recall  : every segment summary, searched on demand
# archival: the raw evidence on disk, reached through tools
# -----------------------------------------

class Memory:

    def __init__(self, video_context=""):

        self.video = video_context
        self.archive = ""
        self.entries = []
        self.entities = {}
        self.recall = []

    # -- save and load, so a run can resume after a cut ----------------

    def to_dict(self):

        return {"video": self.video, "archive": self.archive, "entries": self.entries, "entities": self.entities, "recall": self.recall}

    @classmethod
    def from_dict(cls, data):

        memory = cls(data["video"])
        memory.archive = data["archive"]
        memory.entries = data["entries"]
        memory.entities = data["entities"]
        memory.recall = data["recall"]

        return memory

    # -- what the agents read -----------------------------------------

    def story_text(self):

        lines = [f"[{clock(e['start'])}] {e['text']}" for e in self.entries]

        return "\n".join(([self.archive] if self.archive else []) + lines) or "nothing yet"

    def core_text(self):

        seen = "; ".join(
            f"{name}: {e['detail']} (last seen {clock(e['last_seen'])})"
            for name, e in sorted(self.entities.items(), key=lambda x: -x[1]["last_seen"])
        ) or "nothing yet"

        return f"VIDEO: {self.video or 'unknown'}\n\nSTORY SO FAR:\n{self.story_text()}\n\nOBJECTS AND PEOPLE SEEN: {seen}"

    def search(self, query, hits=RECALL_HITS):

        words = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 2}

        scored = []

        for item in self.recall:

            have = set(re.findall(r"\w+", item["summary"].lower()))
            score = len(words & have) * (1.0 + item["weight"])

            if score:
                scored.append((score, item))

        return [i for _, i in sorted(scored, key=lambda x: -x[0])[:hits]]

    # -- what the agents write ----------------------------------------

    def add(self, segment_id, start, end, weight, result):

        self.entries.append({"id": segment_id, "start": start, "end": end, "weight": weight, "text": result["summary"]})

        self.recall.append({"id": segment_id, "start": start, "end": end, "weight": weight, "summary": result["summary"]})

        for obj in result.get("objects", []):

            name = str(obj.get("name", "")).strip().lower()
            detail = str(obj.get("detail", "")).strip()

            if not name:
                continue

            known = self.entities.get(name)

            if known and detail and detail.lower() not in known["detail"].lower():
                known["detail"] = (known["detail"] + "; " + detail)[-ENTITY_DETAIL_CHARS:]
            elif not known:
                known = self.entities[name] = {"detail": detail[:ENTITY_DETAIL_CHARS], "weight": 0.0, "last_seen": start}

            known["last_seen"] = start
            known["weight"] = max(known["weight"], weight)

        # Forget the least important, longest unseen things first
        while len(self.entities) > ENTITY_MAX:
            drop = min(self.entities, key=lambda n: (self.entities[n]["weight"], self.entities[n]["last_seen"]))
            del self.entities[drop]

    def needs_compression(self):

        return len(self.story_text()) > CORE_STORY_CHARS and len(self.entries) > 2

    def compress(self, condense):

        # condense(text, limit) is an LLM call; the older half of the entries is folded into the archive
        total = sum(len(e["text"]) for e in self.entries)

        old, size = [], 0

        while self.entries[:-1] and size < total / 2:
            entry = self.entries.pop(0)
            old.append(entry)
            size += len(entry["text"])

        lines = [
            f"[{clock(e['start'])}]{' (important)' if e['weight'] >= 0.5 else ''} {e['text']}"
            for e in old
        ]

        self.archive = condense("\n".join(([self.archive] if self.archive else []) + lines), ARCHIVE_CHARS)
