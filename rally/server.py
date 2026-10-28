"""Local review app server. Serves the video with seeking support, the crops, and saves your review."""
from __future__ import annotations

import json
import mimetypes
import re
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import reel, store

STATIC = Path(__file__).parent / "review"


class _State:
    def __init__(self, work: Path):
        self.work = work
        self.meta = json.loads((work / "meta.json").read_text())
        d = np.load(work / "dets.npz")
        self.f, self.tid, self.box = d["f"], d["tid"], d["box"]
        order = np.argsort(self.tid, kind="stable")
        self.f, self.tid, self.box = self.f[order], self.tid[order], self.box[order]
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def boxes(self, tids: list[int], t0: float, t1: float) -> list:
        fps = self.meta["fps"]
        out = []
        for t in tids:
            lo, hi = np.searchsorted(self.tid, t, "left"), np.searchsorted(self.tid, t, "right")
            fr, bx = self.f[lo:hi], self.box[lo:hi]
            m = (fr >= t0 * fps) & (fr <= t1 * fps)
            out += [[round(float(a) / fps, 3), *[round(float(v), 1) for v in b]] for a, b in zip(fr[m], bx[m])]
        out.sort(key=lambda r: r[0])
        return out


def _handler(state: _State):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path: Path, ctype: str | None = None):
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            size = path.stat().st_size
            ctype = ctype or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            if rng:
                m = re.match(r"bytes=(\d*)-(\d*)", rng)
                if m:
                    if m.group(1):
                        start = int(m.group(1))
                        end = int(m.group(2)) if m.group(2) else size - 1
                    else:
                        start = size - int(m.group(2))
                end = min(end, size - 1)
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(HTTPStatus.OK)
            length = end - start + 1
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            with open(path, "rb") as fh:
                fh.seek(start)
                left = length
                try:
                    while left > 0:
                        chunk = fh.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_GET(self):
            u = urlparse(self.path)
            p, q = u.path, parse_qs(u.query)
            w = state.work
            if p in ("/", "/index.html"):
                return self._file(STATIC / "index.html", "text/html; charset=utf-8")
            if p.startswith("/static/"):
                return self._file(STATIC / p[len("/static/"):])
            if p == "/video":
                return self._file(Path(state.meta["video"]), "video/mp4")
            if p.startswith("/crops/"):
                return self._file(w / "crops" / Path(p).name)
            if p.startswith("/reels/"):
                return self._file(w / "reels" / Path(p).name, "video/mp4")
            if p == "/api/session":
                return self._json({
                    "meta": state.meta,
                    "players": json.loads((w / "players.json").read_text()),
                    "moments": json.loads((w / "moments.json").read_text()),
                    "review": store.load_review(w),
                    "cuts": json.loads((w / "cuts.json").read_text()) if (w / "cuts.json").exists() else [],
                })
            if p == "/api/boxes":
                tids = [int(x) for x in q.get("tids", [""])[0].split(",") if x]
                return self._json(state.boxes(tids, float(q["from"][0]), float(q["to"][0])))
            if p == "/api/job":
                return self._json(state.jobs.get(q.get("id", [""])[0], {"status": "unknown"}))
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            p = urlparse(self.path).path
            if p == "/api/review":
                with state.lock:
                    store.save_review(state.work, body)
                return self._json({"ok": True})
            if p == "/api/reel":
                name = body["player"]
                job_id = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "player"
                state.jobs[job_id] = {"status": "running"}

                def work():
                    try:
                        out = reel.build(state.work, name, spotlight=body.get("spotlight", True))
                        state.jobs[job_id] = {"status": "done", "url": f"/reels/{out.name}"}
                    except Exception as e:  # surface the error in the UI
                        state.jobs[job_id] = {"status": "error", "error": str(e)}

                threading.Thread(target=work, daemon=True).start()
                return self._json({"id": job_id})
            self.send_error(HTTPStatus.NOT_FOUND)

    return H


def serve(work: Path, port: int = 8765, open_browser: bool = True):
    state = _State(work)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _handler(state))
    url = f"http://127.0.0.1:{port}/"
    print(f"[review] {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
