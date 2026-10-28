"""Stage 2: group tracks into people by clothing.

A tracker loses people constantly (crossing at the net, walking off frame, camera moves),
so one person ends up as dozens of short tracks. We merge tracks whose clothing matches,
and never merge two tracks that appear in the same frame (one person can't be in two places).
You then name each group once per video in the review app.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import hellinger, pairwise_hellinger


def run(work: Path, min_samples: int = 15, threshold: float = 0.42, attach_threshold: float = 0.36) -> dict:
    tracks = json.loads((work / "tracks.json").read_text())
    dets = np.load(work / "dets.npz")
    f_all, tid_all = dets["f"], dets["tid"]

    # Frames each track appears in, for the "can't be two places at once" rule.
    order = np.argsort(tid_all, kind="stable")
    tid_sorted, f_sorted = tid_all[order], f_all[order]
    bounds = np.searchsorted(tid_sorted, np.unique(tid_sorted))
    uniq = np.unique(tid_sorted)
    frames_of = {int(t): f_sorted[b:e] for t, b, e in zip(uniq, bounds, list(bounds[1:]) + [len(tid_sorted)])}

    core = [t for t, v in tracks.items() if v["feat"] is not None and v["feat_n"] >= min_samples]
    minor = [t for t, v in tracks.items() if v["feat"] is not None and 3 <= v["feat_n"] < min_samples]
    print(f"[identify] {len(core)} solid tracks to cluster, {len(minor)} short tracks to attach")

    groups: list[list[str]] = []
    if core:
        feats = {t: np.asarray(tracks[t]["feat"], np.float32) for t in core + minor}
        n = len(core)
        D = pairwise_hellinger(np.stack([feats[t] for t in core]))
        firsts = np.array([tracks[t]["first"] for t in core])
        lasts = np.array([tracks[t]["last"] for t in core])
        overlap = (firsts[:, None] <= lasts[None, :]) & (firsts[None, :] <= lasts[:, None])
        for i, j in zip(*np.nonzero(np.triu(overlap, 1))):
            shared = np.intersect1d(frames_of[int(core[i])], frames_of[int(core[j])], assume_unique=True)
            if len(shared) >= 2:
                D[i, j] = D[j, i] = 5.0  # cannot-link: one person can't be in two places
        np.fill_diagonal(D, 0)

        if n == 1:
            labels = np.array([0])
        else:
            from sklearn.cluster import AgglomerativeClustering

            labels = AgglomerativeClustering(
                n_clusters=None, metric="precomputed", linkage="average", distance_threshold=threshold
            ).fit_predict(D)
        by_label: dict[int, list[str]] = {}
        for t, lab in zip(core, labels):
            by_label.setdefault(int(lab), []).append(t)
        groups = list(by_label.values())

        # Attach short tracks to the closest group if the clothes clearly match and there's no overlap.
        centroids = [np.mean([feats[t] for t in g], axis=0) for g in groups]
        for t in minor:
            best, best_d = None, attach_threshold
            for gi, c in enumerate(centroids):
                d = hellinger(feats[t], c)
                if d < best_d:
                    clash = False
                    ft = frames_of.get(int(t), np.array([], np.int32))
                    for other in groups[gi]:
                        o = tracks[other]
                        if o["first"] <= tracks[t]["last"] and tracks[t]["first"] <= o["last"]:
                            if len(np.intersect1d(ft, frames_of[int(other)], assume_unique=True)) >= 1:
                                clash = True
                                break
                    if not clash:
                        best, best_d = gi, d
            if best is not None:
                groups[best].append(t)

    # Biggest groups first: these are the people on court the most.
    groups.sort(key=lambda g: -sum(tracks[t]["n"] for t in g))
    out = []
    for gi, g in enumerate(groups):
        g_sorted = sorted(g, key=lambda t: -tracks[t]["n"] * tracks[t]["mean_h"])
        crops = [tracks[t]["crop"] for t in g_sorted if tracks[t]["crop"]][:8]
        out.append({
            "id": f"p{gi + 1}",
            "tracks": [int(t) for t in g_sorted],
            "samples": int(sum(tracks[t]["n"] for t in g)),
            "crops": crops,
        })
    track_to_group = {t: grp["id"] for grp in out for t in grp["tracks"]}
    result = {"groups": out, "track_to_group": {str(k): v for k, v in track_to_group.items()}}
    (work / "players.json").write_text(json.dumps(result))
    print(f"[identify] {len(out)} people found (biggest: {[g['samples'] for g in out[:12]]})")
    return result
