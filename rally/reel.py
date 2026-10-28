"""Stage 5: cut a player's confirmed touches into one highlight reel with ffmpeg."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .store import confirmed_touches

PRE, POST = 3.0, 2.0


def _windows(touches: list[dict], duration: float, pre: float, post: float) -> list[dict]:
    wins: list[dict] = []
    for t in touches:
        a, b = max(0.0, t["t"] - pre), min(duration, t["t"] + post)
        if wins and a <= wins[-1]["end"] + 0.5:
            wins[-1]["end"] = max(wins[-1]["end"], b)
            wins[-1]["touches"].append(t)
        else:
            wins.append({"start": a, "end": b, "touches": [t]})
    return wins


def build(
    work: Path,
    player: str,
    actions: list[str] | None = None,
    outcomes: list[str] | None = None,
    spotlight: bool = True,
    pre: float = PRE,
    post: float = POST,
    out: Path | None = None,
) -> Path:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install it: brew install ffmpeg")
    meta = json.loads((work / "meta.json").read_text())
    touches = [t for t in confirmed_touches(work) if t["player"].lower() == player.lower()]
    if actions:
        touches = [t for t in touches if t["action"] in actions]
    if outcomes:
        touches = [t for t in touches if t["outcome"] in outcomes]
    if not touches:
        raise RuntimeError(f"No kept touches for {player}. Mark some as Keep in the review app first.")

    wins = _windows(touches, meta["duration"], pre, post)
    W, H = meta["width"], meta["height"]
    slug = re.sub(r"[^a-z0-9]+", "-", player.lower()).strip("-")
    out = out or (work / "reels" / f"{slug}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for i, w in enumerate(wins):
            vf = []
            if spotlight:
                # A ring around the player for the moment of contact, so viewers know who to watch.
                for t in w["touches"]:
                    if not t.get("box"):
                        continue
                    x1, y1, x2, y2 = t["box"]
                    pad = 0.12 * (x2 - x1)
                    rel = t["t"] - w["start"]
                    vf.append(
                        f"drawbox=x={max(0, x1 - pad):.0f}:y={max(0, y1 - pad):.0f}:"
                        f"w={min(W, x2 - x1 + 2 * pad):.0f}:h={min(H, y2 - y1 + 2 * pad):.0f}:"
                        f"color=0xF5C542@0.95:t=3:enable='between(t,{max(0, rel - 0.9):.2f},{rel + 0.4:.2f})'"
                    )
            vf.append("format=yuv420p")
            part = Path(tmp) / f"{i:04d}.mp4"
            cmd = [
                "ffmpeg", "-v", "error", "-y", "-ss", f"{w['start']:.2f}", "-i", meta["video"],
                "-t", f"{w['end'] - w['start']:.2f}", "-vf", ",".join(vf),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2", part.name,
            ]
            subprocess.run(cmd, check=True, cwd=tmp)
            parts.append(part)
        listing = Path(tmp) / "list.txt"
        listing.write_text("".join(f"file '{p.name}'\n" for p in parts))
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt",
             "-c", "copy", "-movflags", "+faststart", str(out.resolve())],
            check=True, cwd=tmp,
        )
    total = sum(w["end"] - w["start"] for w in wins)
    print(f"[reel] {player}: {len(touches)} touches in {len(wins)} clips, {total:.0f}s -> {out}")
    return out
