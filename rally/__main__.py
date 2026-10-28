"""Rally command line.

  python -m rally process VIDEO --name 2026-10-27-practice     detect, group people, find touches
  python -m rally review 2026-10-27-practice                   name players, confirm touches (opens browser)
  python -m rally reel 2026-10-27-practice --player "Sir"      export a highlight reel
  python -m rally publish 2026-10-27-practice --push           update the team stats site
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "work"


def _work(name: str) -> Path:
    p = WORK / name
    if not (p / "meta.json").exists():
        raise SystemExit(f"No processed session '{name}' in {WORK}. Run: python -m rally process VIDEO --name {name}")
    return p


def _hms(s: str | None) -> float | None:
    if not s:
        return None
    parts = [float(x) for x in s.split(":")]
    sec = 0.0
    for x in parts:
        sec = sec * 60 + x
    return sec


def main():
    ap = argparse.ArgumentParser(prog="rally", description="Volleyball highlight reels and stats")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("process", help="run detection, grouping and touch finding on a video")
    p.add_argument("video")
    p.add_argument("--name", help="session name, e.g. 2026-10-27-practice (default: video filename)")
    p.add_argument("--model", default="yolo11n-pose.pt", help="pose model (nano tested best on small far-court players)")
    p.add_argument("--fps", type=float, default=10.0, help="frames per second to analyze (default 10)")
    p.add_argument("--imgsz", type=int, help="model input size (default picks by video resolution)")
    p.add_argument("--device", help="mps, cpu, or 0 for CUDA (default: auto)")
    p.add_argument("--start", help="start time, e.g. 12:30")
    p.add_argument("--end", help="end time, e.g. 1:05:00")
    p.add_argument("--threshold", type=float, default=0.42, help="clothing match strictness for grouping (lower = stricter)")
    p.add_argument("--skip-detect", action="store_true", help="reuse existing detections, only regroup and refind touches")

    r = sub.add_parser("review", help="open the review app")
    r.add_argument("name")
    r.add_argument("--port", type=int, default=8765)
    r.add_argument("--no-browser", action="store_true")

    e = sub.add_parser("reel", help="export a player's highlight reel")
    e.add_argument("name")
    e.add_argument("--player", required=True)
    e.add_argument("--actions", help="comma list, e.g. attack,block")
    e.add_argument("--outcomes", help="comma list, e.g. great,good")
    e.add_argument("--no-spotlight", action="store_true")
    e.add_argument("--out")

    s = sub.add_parser("publish", help="write stats for the team site")
    s.add_argument("name")
    s.add_argument("--push", action="store_true", help="git commit and push docs/data")

    a = ap.parse_args()

    if a.cmd == "process":
        from . import actions, detect, identify

        name = a.name or re.sub(r"[^A-Za-z0-9_-]+", "-", Path(a.video).stem)
        work = WORK / name
        if not a.skip_detect:
            detect.run(a.video, work, sample_fps=a.fps, model_name=a.model, device=a.device,
                       imgsz=a.imgsz, start=_hms(a.start) or 0.0, end=_hms(a.end))
        identify.run(work, threshold=a.threshold)
        actions.run(work)
        print(f"\nNext: python -m rally review {name}")
    elif a.cmd == "review":
        from .server import serve

        serve(_work(a.name), port=a.port, open_browser=not a.no_browser)
    elif a.cmd == "reel":
        from .reel import build

        build(_work(a.name), a.player,
              actions=a.actions.split(",") if a.actions else None,
              outcomes=a.outcomes.split(",") if a.outcomes else None,
              spotlight=not a.no_spotlight, out=Path(a.out) if a.out else None)
    elif a.cmd == "publish":
        from .stats import publish

        publish(_work(a.name), push=a.push)


if __name__ == "__main__":
    main()
