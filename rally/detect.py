"""Stage 1: detect and track every person, with pose keypoints and clothing features.

Reads the video once at a reduced sample rate, runs a YOLO pose model with
BoT-SORT tracking, and writes:
  dets.npz     per-detection frame, track id, box, confidence, 17 pose keypoints
  tracks.json  per-track summary: sample count, time span, clothing histogram, best crop
  cuts.json    timestamps where the camera moved or the shot changed
  meta.json    video info and run settings
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import features

TRACKER_CFG = str(Path(__file__).parent / "trackers" / "botsort.yaml")


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def default_imgsz(width: int) -> int:
    # Far-court players are small. Give the model more pixels when the source has them.
    if width <= 960:
        return 640
    if width <= 1920:
        return 960
    return 1280


def run(
    video: str,
    out_dir: Path,
    sample_fps: float = 10.0,
    model_name: str = "yolo11s-pose.pt",
    device: str | None = None,
    imgsz: int | None = None,
    start: float = 0.0,
    end: float | None = None,
    min_box_h: int = 28,
) -> dict:
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "crops").mkdir(exist_ok=True)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Can't open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_frames / fps
    stride = max(1, round(fps / sample_fps))
    imgsz = imgsz or default_imgsz(width)
    device = pick_device(device)

    first = int(start * fps)
    last = int(end * fps) if end else n_frames
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    model = YOLO(model_name)
    print(
        f"[detect] {Path(video).name}: {width}x{height} @ {fps:.1f}fps, {duration/60:.1f} min. "
        f"Sampling every {stride} frames ({fps/stride:.1f}/s), model {model_name}, imgsz {imgsz}, device {device}"
    )

    rec_f, rec_tid, rec_box, rec_conf, rec_kp = [], [], [], [], []
    tracks: dict[int, dict] = {}
    cuts: list[float] = []
    prev_small = None

    t0 = time.time()
    fi = first
    n_done = 0
    total_samples = max(1, (last - first) // stride)
    while fi < last:
        ok, frame = cap.read()
        if not ok:
            break
        # Skip ahead without full decode work where possible.
        for _ in range(stride - 1):
            if not cap.grab():
                break

        # Camera move / shot change detection: compare coarse grayscale thumbnails.
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32)
        if prev_small is not None:
            diff = float(np.mean(np.abs(small - prev_small)))
            if diff > 28:
                cuts.append(round(fi / fps, 2))
        prev_small = small

        res = model.track(
            frame,
            persist=True,
            tracker=TRACKER_CFG,
            imgsz=imgsz,
            device=device,
            classes=[0],
            conf=0.2,
            verbose=False,
        )[0]

        if res.boxes is not None and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            kps = res.keypoints.data.cpu().numpy() if res.keypoints is not None else np.zeros((len(ids), 17, 3))
            for box, tid, cf, kp in zip(boxes, ids, confs, kps):
                h = box[3] - box[1]
                if h < min_box_h:
                    continue
                rec_f.append(fi)
                rec_tid.append(tid)
                rec_box.append(box)
                rec_conf.append(cf)
                rec_kp.append(kp)

                tr = tracks.get(tid)
                if tr is None:
                    tr = tracks[tid] = {
                        "n": 0, "first": fi, "last": fi, "feat_sum": None, "feat_n": 0,
                        "best_score": -1.0, "h_sum": 0.0,
                    }
                tr["n"] += 1
                tr["last"] = fi
                tr["h_sum"] += float(h)
                feat = features.clothing_hist(frame, box, kp)
                if feat is not None:
                    tr["feat_sum"] = feat if tr["feat_sum"] is None else tr["feat_sum"] + feat
                    tr["feat_n"] += 1
                # Keep the clearest crop for the naming screen: big, confident, full body visible.
                score = float(h) * float(cf) * (1.0 + float(np.mean(kp[:, 2] > 0.5)))
                if score > tr["best_score"]:
                    crop = features.crop_person(frame, box)
                    if crop is not None:
                        cv2.imwrite(str(out_dir / "crops" / f"{tid}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
                        tr["best_score"] = score

        fi += stride
        n_done += 1
        if n_done % 500 == 0:
            el = time.time() - t0
            rate = n_done / el
            eta = (total_samples - n_done) / rate
            print(f"[detect] {fi/fps/60:6.1f} min  {rate:5.1f} samples/s  ETA {eta/60:5.1f} min", flush=True)

    cap.release()

    np.savez_compressed(
        out_dir / "dets.npz",
        f=np.asarray(rec_f, np.int32),
        tid=np.asarray(rec_tid, np.int32),
        box=np.asarray(rec_box, np.float32).reshape(-1, 4),
        conf=np.asarray(rec_conf, np.float16),
        kp=np.asarray(rec_kp, np.float32).reshape(-1, 17, 3),
    )
    out_tracks = {}
    for tid, tr in tracks.items():
        feat = (tr["feat_sum"] / tr["feat_n"]).round(5).tolist() if tr["feat_n"] else None
        out_tracks[str(tid)] = {
            "n": tr["n"],
            "first": tr["first"],
            "last": tr["last"],
            "mean_h": round(tr["h_sum"] / tr["n"], 1),
            "feat": feat,
            "feat_n": tr["feat_n"],
            "crop": f"crops/{tid}.jpg" if tr["best_score"] >= 0 else None,
        }
    (out_dir / "tracks.json").write_text(json.dumps(out_tracks))
    (out_dir / "cuts.json").write_text(json.dumps(cuts))
    meta = {
        "video": str(Path(video).resolve()),
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration,
        "stride": stride,
        "sample_fps": fps / stride,
        "model": model_name,
        "imgsz": imgsz,
        "start": start,
        "end": end,
        "processed_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[detect] done: {len(rec_f)} detections, {len(tracks)} tracks, {len(cuts)} camera cuts, "
          f"{meta['processed_seconds']/60:.1f} min")
    return meta
