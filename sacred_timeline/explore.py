"""A local web page for looking at the moment embeddings: a 2D map (UMAP), automatic clusters with
plain-language labels, a search box that lights up matching moments, and a strip of stills that
play the moment from the source file when clicked.

  python -m sacred_timeline explore      -> http://127.0.0.1:8777

Everything stays on this Mac: the server listens on 127.0.0.1 only and reads the index directly.
"""
from __future__ import annotations

import json
import mimetypes
import re
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .index import HOME, vectors

PAGE = Path(__file__).with_name("explore.html")

# Short scene descriptions used to name clusters: each cluster takes the one closest to its centre.
VOCAB = [
    "people talking at a front desk", "a group of friends together", "a man giving a speech", "a crowd outdoors",
    "two people arguing", "someone laughing", "an awkward silence", "a tense confrontation", "a close-up of a face",
    "a hotel lobby", "an office", "a car interior", "a street at night", "a desert", "a military base",
    "an explosion", "a fight", "a chase", "a party", "a bar or restaurant", "a bedroom", "a kitchen",
    "a phone call", "someone walking", "a romantic moment", "someone crying", "a celebration",
    "a title card or text on screen", "a channel logo or subscribe screen", "an aerial view of a city",
    "a man in a suit", "soldiers", "a woman talking", "a man talking to the camera", "a meeting around a table",
]


class Explorer:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.embedder = None
        self.model = None
        self.lock = threading.Lock()
        self.map = None

    def db(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def emb(self):
        with self.lock:
            if self.embedder is None:
                from .embed import Embedder
                self.embedder = Embedder()
            return self.embedder

    def build_map(self) -> dict:
        db = self.db()
        ids, mat = vectors(db)
        n = len(ids)
        cache = HOME / "map.json"
        if cache.exists():
            m = json.loads(cache.read_text())
            if m.get("n") == n and m.get("ids") == ids.tolist():
                self.map = m
                self._refresh_tags(m, db)
                return m
        t = time.time()
        import umap
        from sklearn.cluster import HDBSCAN
        xy = umap.UMAP(n_neighbors=max(2, min(15, n - 1)), min_dist=0.08, metric="cosine",
                       random_state=7).fit_transform(mat)
        xy = (xy - xy.min(0)) / np.maximum(xy.max(0) - xy.min(0), 1e-9)
        labels = HDBSCAN(min_cluster_size=max(3, n // 60), metric="euclidean").fit_predict(xy) if n >= 6 \
            else np.zeros(n, int)
        vocab = self.emb().queries(VOCAB)
        cids = sorted(set(labels.tolist()) - {-1})
        centres = np.stack([mat[labels == c].mean(0) for c in cids]) if cids else np.zeros((0, mat.shape[1]))
        centres /= np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), 1e-9)
        # Each cluster gets a different description: what sets it apart from the other clusters,
        # assigned greedily from the most distinctive match down.
        sims = centres @ vocab.T
        distinct = sims - sims.mean(0, keepdims=True)
        names, used = {}, set()
        for flat in np.argsort(-distinct, axis=None):
            ci, vi = divmod(int(flat), len(VOCAB))
            if ci in names or vi in used:
                continue
            names[ci], _ = VOCAB[vi], used.add(vi)
        clusters = [{"id": c, "label": names.get(i, f"group {c}"), "n": int((labels == c).sum()),
                     "x": float(xy[labels == c, 0].mean()), "y": float(xy[labels == c, 1].mean())}
                    for i, c in enumerate(cids)]
        rows = {r["id"]: r for r in db.execute(
            "select m.id, m.t0, m.t1, m.thumb, m.tags, c.path, c.id as clip_id from moments m "
            "join clips c on c.id = m.clip_id")}
        clips = {}
        points = []
        for i, mid in enumerate(ids.tolist()):
            r = rows[mid]
            clips.setdefault(r["clip_id"], {"id": r["clip_id"], "name": clip_name(r["path"]), "path": r["path"]})
            points.append({"id": mid, "x": float(xy[i, 0]), "y": float(xy[i, 1]), "cluster": int(labels[i]),
                           "clip": r["clip_id"], "t0": r["t0"], "t1": r["t1"],
                           "tags": json.loads(r["tags"]) if r["tags"] else {}})
        m = {"n": n, "ids": ids.tolist(), "points": points, "clusters": clusters, "clips": list(clips.values()),
             "built_seconds": round(time.time() - t, 1)}
        cache.write_text(json.dumps(m))
        self.map = m
        return m

    def _refresh_tags(self, m: dict, db: sqlite3.Connection):
        tags = {r[0]: r[1] for r in db.execute("select id, tags from moments where tags is not null")}
        for p in m["points"]:
            if p["id"] in tags:
                p["tags"] = json.loads(tags[p["id"]])

    def vecs(self):
        n = self.db().execute("select count(*) from moments where vec is not null").fetchone()[0]
        if getattr(self, "_vecs", None) is None or len(self._vecs[0]) != n:
            self._vecs = vectors(self.db())
        return self._vecs

    def search(self, q: str, k: int = 24) -> dict:
        ids, mat = self.vecs()
        t = time.time()
        qv = self.emb().queries([q])[0]
        sims = mat @ qv
        order = np.argsort(-sims)
        ms = (time.time() - t) * 1000
        top = order[:k]
        return {"query": q, "ms": round(ms, 1), "hits": [{"id": int(ids[j]), "sim": float(sims[j])} for j in top],
                "sims": {int(ids[j]): round(float(sims[j]), 4) for j in range(len(ids))}}

    def judge(self, q: str, ids: list[int]) -> dict:
        from .index import ClipCache, ask
        from .models import load_model
        from .search import question
        from .tags import TAGS, flatten
        with self.lock:
            if self.model is None:
                self.model = load_model("clef-omni")
        db = self.db()
        t = time.time()
        ans = ask(db, self.model, ids, {"q": question(q)} | TAGS, ClipCache(self.model), log=lambda s: None)
        out = {}
        for mid in ids:
            tags = flatten({k: ans[mid][k] for k in TAGS})
            db.execute("update moments set tags = ? where id = ?", (json.dumps(tags), mid))
            out[mid] = {"score": ans[mid]["q"]["noul"], "tags": tags}
        db.commit()
        return {"seconds": round(time.time() - t, 1), "results": out}


def clip_name(path: str) -> str:
    s = re.sub(r"\(.*?\)|#\w+|^(YTDown\.com_YouTube_|ssvid\.net-+)|_\d+p$|_Media_\w+", "", Path(path).stem)
    s = re.sub(r"[-_]+", " ", s).strip()
    return s[:60]


class Handler(BaseHTTPRequestHandler):
    ex: Explorer

    def log_message(self, *a):
        pass

    def send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj):
        self.send(200, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/map":
                return self.json(self.ex.map or self.ex.build_map())
            if u.path == "/api/search":
                return self.json(self.ex.search(q["q"][0]))
            if u.path == "/api/judge":
                ids = [int(x) for x in q["ids"][0].split(",") if x]
                return self.json(self.ex.judge(q["q"][0], ids))
            m = re.match(r"^/(thumb|video)/(\d+)$", u.path)
            if m:
                row = self.ex.db().execute("select m.thumb, c.path from moments m join clips c on c.id = m.clip_id "
                                           "where m.id = ?", (int(m.group(2)),)).fetchone()
                if not row:
                    return self.send(404, b"", "text/plain")
                if m.group(1) == "thumb":
                    return self.send(200, Path(row["thumb"]).read_bytes(), "image/jpeg",
                                     {"Cache-Control": "max-age=86400"})
                return self.stream(Path(row["path"]))
            self.send(404, b"not found", "text/plain")
        except Exception as e:  # keep the server up; show the error in the page
            self.send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")

    def stream(self, path: Path):
        """Serve the source file with HTTP range support so the browser can seek to the moment."""
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "video/mp4"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and (m := re.match(r"bytes=(\d*)-(\d*)", rng)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = size - int(m.group(2))
        end = min(end, size - 1, start + 8 * 1024 * 1024 - 1)
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)


def serve(db: sqlite3.Connection, port: int = 8777):
    db.close()
    ex = Explorer(HOME / "index.db")
    print("building the map (UMAP + clusters)...", flush=True)
    m = ex.build_map()
    print(f"  {m['n']} moments, {len(m['clusters'])} clusters")
    Handler.ex = ex
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"explorer: http://127.0.0.1:{port}", flush=True)
    srv.serve_forever()
