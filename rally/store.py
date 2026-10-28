"""Review state: names for each person, and your verdict on each candidate touch.

review.json is the source of truth for reels and stats. Nothing counts until it's in here.
"""
from __future__ import annotations

import json
from pathlib import Path

ACTIONS = ["attack", "set", "pass", "dig", "block", "serve", "other"]
OUTCOMES = ["great", "good", "error"]


def load_review(work: Path) -> dict:
    p = work / "review.json"
    if p.exists():
        r = json.loads(p.read_text())
    else:
        r = {}
    r.setdefault("session", {"title": work.name, "date": "", "type": "practice", "youtube": ""})
    r.setdefault("players", {})
    r.setdefault("moments", {})
    r.setdefault("added", [])
    return r


def save_review(work: Path, review: dict) -> None:
    tmp = work / "review.json.tmp"
    tmp.write_text(json.dumps(review, indent=1))
    tmp.replace(work / "review.json")


def confirmed_touches(work: Path) -> list[dict]:
    """Every kept touch with a resolved player name, from AI candidates and manual additions."""
    review = load_review(work)
    moments = json.loads((work / "moments.json").read_text())
    names = {g: (v.get("name") or "").strip() for g, v in review["players"].items()}
    out = []
    for m in moments:
        r = review["moments"].get(m["id"], {})
        if r.get("status") != "keep":
            continue
        group = r.get("player") or m["group"]
        name = names.get(group, "")
        if not name:
            continue
        out.append({
            "id": m["id"], "t": m["t"], "tid": m["tid"], "group": group, "player": name,
            "action": r.get("action") or m["action"], "outcome": r.get("outcome"),
            "box": m["box"], "source": "ai",
        })
    for a in review["added"]:
        if a.get("status", "keep") != "keep":
            continue
        name = names.get(a.get("player"), "")
        if not name:
            continue
        out.append({
            "id": a["id"], "t": a["t"], "tid": None, "group": a["player"], "player": name,
            "action": a.get("action") or "other", "outcome": a.get("outcome"), "box": None, "source": "manual",
        })
    out.sort(key=lambda x: x["t"])
    return out
