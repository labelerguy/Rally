"""Stage 6: turn confirmed touches into stats and publish them to the team site (docs/)."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .store import ACTIONS, confirmed_touches, load_review

REPO = Path(__file__).resolve().parent.parent
SITE_DATA = REPO / "docs" / "data"


def youtube_id(url: str) -> str:
    if not url:
        return ""
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    return url if re.fullmatch(r"[A-Za-z0-9_-]{11}", url) else ""


def _blank():
    return {a: {"n": 0, "great": 0, "good": 0, "error": 0} for a in ACTIONS}


def session_stats(work: Path) -> dict:
    review = load_review(work)
    meta = json.loads((work / "meta.json").read_text())
    touches = confirmed_touches(work)
    players: dict[str, dict] = {}
    team = _blank()
    for t in touches:
        p = players.setdefault(t["player"], {"name": t["player"], "by_action": _blank(), "clips": []})
        a = t["action"] if t["action"] in ACTIONS else "other"
        for bucket in (p["by_action"][a], team[a]):
            bucket["n"] += 1
            if t["outcome"] in ("great", "good", "error"):
                bucket[t["outcome"]] += 1
        p["clips"].append({"t": t["t"], "action": a, "outcome": t["outcome"]})
    s = review["session"]
    return {
        "id": work.name,
        "title": s.get("title") or work.name,
        "date": s.get("date", ""),
        "type": s.get("type", "practice"),
        "youtube": youtube_id(s.get("youtube", "")),
        "duration": round(meta["duration"], 1),
        "team": team,
        "players": sorted(players.values(), key=lambda p: p["name"].lower()),
    }


def publish(work: Path, push: bool = False) -> Path:
    data = session_stats(work)
    (SITE_DATA / "sessions").mkdir(parents=True, exist_ok=True)
    out = SITE_DATA / "sessions" / f"{data['id']}.json"
    out.write_text(json.dumps(data, indent=1))
    idx_path = SITE_DATA / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {"team": "", "sessions": []}
    idx["sessions"] = [x for x in idx["sessions"] if x["id"] != data["id"]]
    idx["sessions"].append({k: data[k] for k in ("id", "title", "date", "type")})
    idx["sessions"].sort(key=lambda x: x.get("date") or "", reverse=True)
    idx_path.write_text(json.dumps(idx, indent=1))
    n = sum(sum(v["n"] for v in p["by_action"].values()) for p in data["players"])
    print(f"[publish] {data['title']}: {len(data['players'])} players, {n} confirmed touches -> {out.relative_to(REPO)}")
    if not data["youtube"]:
        print("[publish] tip: set the YouTube link in the review app so teammates can watch their touches on the site")
    if push:
        subprocess.run(["git", "add", "docs/data"], cwd=REPO, check=True)
        subprocess.run(["git", "commit", "-m", f"Publish stats: {data['title']}"], cwd=REPO, check=False)
        subprocess.run(["git", "push"], cwd=REPO, check=True)
    return out
