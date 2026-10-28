# Rally: notes for Claude

A volleyball film tool for Sir's club team. It produces player highlight reels and team stats from practice and match video. It's a personal team tool, not a commercial product. The detector is Ultralytics YOLO (AGPL), which is fine while this stays open source.

## Pipeline (python -m rally ...)
- `detect.py`: YOLO pose + BoT-SORT at ~10 samples/s → `work/<s>/dets.npz`, `tracks.json` (per-track clothing histogram + best crop), `cuts.json`, `meta.json`.
- `identify.py`: groups tracks into people by clothing (Hellinger distance on HSV histograms of shirt and shorts, average-linkage clustering, with a cannot-link for tracks that share frames) → `players.json`.
- `actions.py`: rule-based pose signatures → candidate touches in `moments.json`. No ball tracking (the ball is ~4px at 360p).
- `server.py` + `review/index.html`: local review app. Names live on groups. Groups sharing a name are the same person.
- `store.py`: `review.json` is the source of truth. Only `status: keep` touches with a named player count.
- `reel.py`: ffmpeg cuts touch t-3s to t+2s, merges overlapping windows, draws a gold box around the player at contact.
- `stats.py`: writes `docs/data/sessions/<id>.json` + `docs/data/index.json` for the GitHub Pages site in `docs/`.

## Rules
- Never commit video, `work/`, or model weights.
- Stats come only from human-confirmed touches. Don't add auto-counted stats to the site.
- The site is public (Sir chose anyone-with-link). Don't publish anything beyond names, counts, and clip timestamps.
- Test changes on a 5 to 10 minute slice (`--start/--end`) before running full sessions.

## Tuning knobs
- Grouping too eager (two people merged): lower `--threshold` (default 0.42). Too split: raise it.
- Too many junk candidates: raise the confidence floor in the review UI, or raise `min_score` in `actions.run`.
- Far players missed: `--model yolo11m-pose.pt` and higher `--imgsz` on 1080p+ footage.

## Roadmap
1. Jersey-number OCR for matches (team in identical jerseys breaks clothing grouping).
2. Ball detection on 1080p+ footage, to confirm contacts and cut false positives.
3. Court calibration per camera segment → rotation and zone stats.
4. Rally segmentation + point winner → side-out %, rally win rate.
