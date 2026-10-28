"""Stage 3: find candidate touches from body pose.

At practice-camera resolution the ball is a few pixels, so we don't rely on it. Every
volleyball contact has a body signature instead:
  attack  jump + hitting arm above the head
  block   jump + both hands up, close together
  set     both hands up at the forehead, no big jump
  serve   one arm high while grounded (toss + swing)
  pass    forearms together low in front of the body (platform)
These are candidates. A person confirms each one in the review app before it counts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import L_AN, L_EL, L_HIP, L_SH, L_WR, NOSE, R_AN, R_EL, R_HIP, R_SH, R_WR

KP_OK = 0.35
WRIST_OK = 0.25  # wrists are small and blurry on far players; accept weaker evidence
BASE_SCORE = {"attack": 1.0, "block": 0.9, "set": 0.75, "serve": 0.6, "pass": 0.8}


def _rolling_median(x: np.ndarray, half: int) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(x[max(0, i - half): i + half + 1])
    return out


def _pose_states(box: np.ndarray, kp: np.ndarray, hb: np.ndarray) -> dict[str, np.ndarray]:
    def ok(i):
        return kp[:, i, 2] > KP_OK

    y = lambda i: kp[:, i, 1]  # noqa: E731
    x = lambda i: kp[:, i, 0]  # noqa: E731
    sh_ok = ok(L_SH) & ok(R_SH)
    sh_y = np.where(sh_ok, (y(L_SH) + y(R_SH)) / 2, box[:, 1] + 0.2 * hb)
    head_y = np.where(ok(NOSE), y(NOSE), sh_y - 0.12 * hb)
    sh_w = np.where(sh_ok, np.abs(x(L_SH) - x(R_SH)), 0.25 * (box[:, 2] - box[:, 0]))
    hip_ok = ok(L_HIP) & ok(R_HIP)
    hip_y = np.where(hip_ok, (y(L_HIP) + y(R_HIP)) / 2, box[:, 1] + 0.52 * hb)

    lw_ok, rw_ok = kp[:, L_WR, 2] > WRIST_OK, kp[:, R_WR, 2] > WRIST_OK
    lw_up = lw_ok & (y(L_WR) < head_y - 0.04 * hb)
    rw_up = rw_ok & (y(R_WR) < head_y - 0.04 * hb)
    both_up = lw_up & rw_up
    one_up = lw_up ^ rw_up
    # The other arm in a hitting swing is usually lower than the shoulder line or across the body.
    hit_arm = one_up | (both_up & (np.abs(y(L_WR) - y(R_WR)) > 0.15 * hb))
    wr_dist = np.hypot(x(L_WR) - x(R_WR), y(L_WR) - y(R_WR))
    hands_close = lw_ok & rw_ok & (wr_dist < np.maximum(1.3 * sh_w, 0.12 * hb))
    set_shape = both_up & hands_close & (np.minimum(y(L_WR), y(R_WR)) > head_y - 0.35 * hb)
    block_shape = both_up & (np.abs(y(L_WR) - y(R_WR)) < 0.1 * hb)

    # Platform: wrists together between chest and knees, arms reaching out (not hanging at the sides,
    # which in a side view also puts both wrists in the same spot), and legs bent.
    wr_mid_x, wr_mid_y = (x(L_WR) + x(R_WR)) / 2, (y(L_WR) + y(R_WR)) / 2
    sh_mid_x = (x(L_SH) + x(R_SH)) / 2
    foot_y = box[:, 3]
    leg = (foot_y - hip_y) / np.maximum(hb, 1)
    leg_ref = np.median(leg[hip_ok]) if hip_ok.any() else 0.5
    crouch = hip_ok & (leg < 0.9 * leg_ref)
    reach = np.hypot(wr_mid_x - sh_mid_x, wr_mid_y - sh_y)
    frontal = sh_w > 0.14 * hb
    arms_out = (np.abs(wr_mid_x - sh_mid_x) > 0.09 * hb) | frontal
    platform = (
        lw_ok & rw_ok & sh_ok & (crouch | frontal) & arms_out
        & (wr_dist < 0.12 * hb)
        & (wr_mid_y > sh_y + 0.2 * hb)
        & (wr_mid_y < hip_y + 0.2 * hb)
        & (reach > 0.28 * hb)
    )
    conf = np.nanmean(kp[:, [NOSE, L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP], 2], axis=1)
    return {
        "hit_arm": hit_arm, "both_up": both_up, "one_up": one_up, "set_shape": set_shape,
        "block_shape": block_shape, "platform": platform, "conf": conf,
    }


def _jump(box: np.ndarray, hb: np.ndarray, sps: float) -> np.ndarray:
    """The whole body clearly higher than both shortly before and shortly after.

    Two look-alikes are rejected: walking away from the camera (feet rise slowly and don't come back
    down) and someone passing in front (the box bottom jumps up because the legs are hidden, but the
    head doesn't move). In a real jump, the top of the box rises along with the feet."""
    foot, top = box[:, 3], box[:, 1]
    n = len(foot)
    w_far, w_near = int(round(1.4 * sps)), int(round(0.45 * sps))
    jump = np.zeros(n, bool)
    hmed = _rolling_median(hb, int(sps))
    for i in range(n):
        lo_b, hi_b = max(0, i - w_far), max(0, i - w_near)
        lo_a, hi_a = i + w_near + 1, i + w_far + 1
        if hi_b - lo_b < 2 or len(foot[lo_a:hi_a]) < 2:
            continue
        foot_rise = min(np.median(foot[lo_b:hi_b]), np.median(foot[lo_a:hi_a])) - foot[i]
        top_rise = min(np.median(top[lo_b:hi_b]), np.median(top[lo_a:hi_a])) - top[i]
        if foot_rise > 0.08 * hmed[i] and top_rise > 0.3 * foot_rise:
            jump[i] = True
    return jump


def _events(states: dict, jump: np.ndarray) -> list[tuple[int, str, float]]:
    """Return (index, action, strength) per detected contact signature."""
    n = len(jump)
    labels = [None] * n
    strength = np.zeros(n)
    jump_near = np.convolve(jump.astype(int), np.ones(5, int), mode="same") > 0
    for i in range(n):
        s = states
        if jump_near[i] and s["block_shape"][i]:
            labels[i] = "block"
        elif jump_near[i] and (s["hit_arm"][i] or s["both_up"][i]):
            labels[i] = "attack"
        elif jump[i]:
            labels[i] = "attack"
        elif s["set_shape"][i]:
            labels[i] = "set"
        elif s["platform"][i]:
            labels[i] = "pass"
        elif s["one_up"][i]:
            labels[i] = "serve"
        if labels[i]:
            strength[i] = float(s["conf"][i]) * (1.25 if jump[i] else 1.0)

    out = []
    i = 0
    while i < n:
        if labels[i] is None:
            i += 1
            continue
        j = i
        while j + 1 < n and labels[j + 1] is not None:
            j += 1
        seg = range(i, j + 1)
        # A run can mix labels (a jump starts as "serve", peaks as "attack"); keep the strongest kind.
        votes: dict[str, float] = {}
        for k in seg:
            votes[labels[k]] = votes.get(labels[k], 0) + strength[k] * BASE_SCORE[labels[k]]
        act = max(votes, key=votes.get)
        peak = max((k for k in seg if labels[k] == act), key=lambda k: strength[k])
        dur = j - i + 1
        # One-frame flickers are mostly noise, except jumps (fast at 10 samples/s).
        # Contacts are fast: at 10 samples/s a set or pass may show in a single frame. Keep those,
        # with a lower score, so they sort below multi-frame evidence in the review app.
        out.append((peak, act, votes[act] / dur * min(1.0, 0.6 + 0.2 * dur)))
        i = j + 1
    return out


def _activity(f: np.ndarray, tid: np.ndarray, box: np.ndarray, fps: float, stride: int) -> dict[int, float]:
    """How much everyone is moving at each sampled frame. Drills and rallies move; water breaks don't."""
    cx = (box[:, 0] + box[:, 2]) / 2
    cy = box[:, 3]
    h = box[:, 3] - box[:, 1]
    speed_by_frame: dict[int, list[float]] = {}
    order = np.lexsort((f, tid))
    for a, b in zip(order[:-1], order[1:]):
        if tid[a] == tid[b] and f[b] - f[a] == stride:
            v = np.hypot(cx[b] - cx[a], cy[b] - cy[a]) / max(h[b], 1) * (fps / stride)
            speed_by_frame.setdefault(int(f[b]), []).append(v)
    frames = sorted(speed_by_frame)
    raw = np.array([np.percentile(speed_by_frame[k], 75) for k in frames]) if frames else np.array([])
    smooth = np.convolve(raw, np.ones(20) / 20, mode="same") if len(raw) else raw
    return {k: float(v) for k, v in zip(frames, smooth)}


def run(work: Path, min_score: float = 0.25) -> list[dict]:
    meta = json.loads((work / "meta.json").read_text())
    players = json.loads((work / "players.json").read_text())
    t2g = players["track_to_group"]
    fps, stride = meta["fps"], meta["stride"]
    sps = fps / stride
    d = np.load(work / "dets.npz")
    f, tid, box, kp = d["f"], d["tid"], d["box"], d["kp"]
    activity = _activity(f, tid, box, fps, stride)
    act_vals = np.array(list(activity.values())) if activity else np.array([0.0])
    act_ref = float(np.percentile(act_vals, 70)) or 1.0

    moments = []
    for t in np.unique(tid):
        idx = np.where(tid == t)[0]
        if len(idx) < 6:
            continue
        idx = idx[np.argsort(f[idx])]
        b, k = box[idx], kp[idx]
        hb = b[:, 3] - b[:, 1]
        states = _pose_states(b, k, hb)
        jump = _jump(b, hb, sps)
        for (i, action, strength) in _events(states, jump):
            fr = int(f[idx[i]])
            a = activity.get(fr, 0.0)
            score = strength * BASE_SCORE[action] * (0.45 + 0.55 * min(1.0, a / act_ref))
            if hb[i] < 45:
                score *= 0.7  # small, far players: pose is less reliable
            if score < min_score:
                continue
            moments.append({
                "t": round(fr / fps, 2),
                "f": fr,
                "tid": int(t),
                "group": t2g.get(str(int(t))),
                "action": action,
                "score": round(float(score), 3),
                "box": [round(float(v), 1) for v in b[i]],
            })

    # Same person, same action within a second: one contact seen twice. Keep the stronger.
    moments.sort(key=lambda m: (m["group"] or f"t{m['tid']}", m["t"]))
    merged: list[dict] = []
    for m in moments:
        key = m["group"] or f"t{m['tid']}"
        if merged:
            p = merged[-1]
            pkey = p["group"] or f"t{p['tid']}"
            if pkey == key and m["t"] - p["t"] < 1.0:
                if m["score"] > p["score"]:
                    merged[-1] = m
                continue
        merged.append(m)
    merged.sort(key=lambda m: m["t"])
    for i, m in enumerate(merged):
        m["id"] = f"m{i + 1}"
    (work / "moments.json").write_text(json.dumps(merged))
    counts: dict[str, int] = {}
    for m in merged:
        counts[m["action"]] = counts.get(m["action"], 0) + 1
    print(f"[actions] {len(merged)} candidate touches: {counts}")
    return merged
