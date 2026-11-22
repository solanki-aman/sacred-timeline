"""A local web page for looking at the moment embeddings.

  python -m sacred_timeline explore      -> http://127.0.0.1:8777

- Automatic map: UMAP layout and clusters found in the footage, named from a fixed set of scene
  descriptions. Good for a cold start: what is in this library?
- Project lenses: pick a project and the map is re-arranged around that project's own clusters,
  which come from its Clusters.md note (see lenses.py) and can be added or removed on the page.
- Search lights up matching moments; Clef-Omni can judge the shortlist; clicking a still plays the
  moment from the source file.

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

from . import lenses
from .index import HOME, vectors

PAGE = Path(__file__).with_name("explore.html")  # the embedding map, at /map
WALL = Path(__file__).with_name("wall.html")  # the wall of stills and the string-out, at /
FRAMES_PER_MOMENT = 5  # stills per moment for skimming

# Short scene descriptions used to name automatic clusters.
VOCAB = [
    "people talking at a front desk", "a group of friends together", "a man giving a speech", "a crowd outdoors",
    "two people arguing", "someone laughing", "an awkward silence", "a tense confrontation", "a close-up of a face",
    "a hotel lobby", "an office", "a car interior", "a street at night", "a desert", "a military base",
    "an explosion", "a fight", "a chase", "a party", "a bar or restaurant", "a bedroom", "a kitchen",
    "a phone call", "someone walking", "a romantic moment", "someone crying", "a celebration",
    "a title card or text on screen", "a channel logo or subscribe screen", "an aerial view of a city",
    "a man in a suit", "soldiers", "a woman talking", "a man talking to the camera", "a meeting around a table",
]


def normalise(xy: np.ndarray) -> np.ndarray:
    return (xy - xy.min(0)) / np.maximum(xy.max(0) - xy.min(0), 1e-9)


def umap_layout(mat: np.ndarray, y=None, init=None, min_dist: float = 0.08, target_weight: float = 0.55) -> np.ndarray:
    import umap
    n = len(mat)
    kw = dict(n_neighbors=max(2, min(15, n - 1)), min_dist=min_dist, metric="cosine", random_state=7)
    if y is not None:
        kw.update(target_metric="categorical", target_weight=target_weight)
    if init is not None:
        kw["init"] = init
    return normalise(umap.UMAP(**kw).fit_transform(mat, y=y))


def samples(mat: np.ndarray, members: np.ndarray, centre: np.ndarray, k: int = 4) -> list[int]:
    """Indexes of the k members closest to a centre vector."""
    idx = np.where(members)[0]
    return idx[np.argsort(-(mat[idx] @ centre))[:k]].tolist()


class Explorer:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.embedder = None
        self.model = None
        self.lock = threading.Lock()
        self.base = None

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

    def vecs(self):
        n = self.db().execute("select count(*) from moments where vec is not null").fetchone()[0]
        if getattr(self, "_vecs", None) is None or len(self._vecs[0]) != n:
            self._vecs = vectors(self.db())
        return self._vecs

    # ------------------------------------------------------------------ automatic map

    def build_base(self) -> dict:
        ids, mat = self.vecs()
        n = len(ids)
        cache = HOME / "map.json"
        if cache.exists():
            m = json.loads(cache.read_text())
            if m.get("version") == 2 and m.get("ids") == ids.tolist():
                self.base = m
                return m
        t = time.time()
        from sklearn.cluster import HDBSCAN
        xy = umap_layout(mat)
        labels = HDBSCAN(min_cluster_size=max(3, n // 60)).fit_predict(xy) if n >= 6 else np.zeros(n, int)
        vocab = self.emb().queries(VOCAB)
        cids = sorted(set(labels.tolist()) - {-1})
        centres = np.stack([mat[labels == c].mean(0) for c in cids]) if cids else np.zeros((0, mat.shape[1]))
        centres /= np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), 1e-9)
        # Each cluster takes the description that sets it apart most from the other clusters.
        sims = centres @ vocab.T
        distinct = sims - sims.mean(0, keepdims=True)
        names, used = {}, set()
        for flat in np.argsort(-distinct, axis=None):
            ci, vi = divmod(int(flat), len(VOCAB))
            if ci not in names and vi not in used:
                names[ci] = VOCAB[vi]
                used.add(vi)
        clusters = [{"id": i, "label": names.get(i, f"group {i + 1}"), "description": "",
                     "n": int((labels == c).sum()),
                     "samples": [int(ids[j]) for j in samples(mat, labels == c, centres[i])]}
                    for i, c in enumerate(cids)]
        remap = {c: i for i, c in enumerate(cids)}
        m = {"version": 2, "ids": ids.tolist(), "xy": xy.round(5).tolist(),
             "cluster": [remap.get(int(c), -1) for c in labels], "clusters": clusters,
             "built_seconds": round(time.time() - t, 1)}
        cache.write_text(json.dumps(m))
        self.base = m
        return m

    # ------------------------------------------------------------------ project lens

    def build_lens(self, project: str) -> dict:
        base = self.base or self.build_base()
        ids, mat = self.vecs()
        lens = lenses.read(project)
        k = lenses.key(lens, ids.tolist())
        cache = HOME / "lenses" / f"{k}.json"
        if cache.exists():
            return json.loads(cache.read_text())
        t = time.time()
        texts = [f"{c['label']}: {c['description']}" if c["description"] else c["label"] for c in lens["clusters"]]
        cvecs = self.emb().queries(texts) if texts else np.zeros((0, mat.shape[1]), np.float32)
        skip = lenses.cards(mat, self.emb().queries([lenses.CARD_TEXT])[0], self.footage(ids))
        labels, csim = lenses.assign(mat, cvecs, skip)
        intent = np.zeros(len(ids))
        if lens["intent"]:
            r = mat @ self.emb().queries([lens["intent"]])[0]
            intent = (r - r.min()) / max(r.max() - r.min(), 1e-9)
        xy = lenses.layout(mat, cvecs, labels, ids.tolist()) if len(cvecs) else np.asarray(base["xy"])
        clusters = [{"id": i, "label": c["label"], "description": c["description"], "n": int((labels == i).sum()),
                     "samples": [int(ids[j]) for j in samples(mat, labels == i, cvecs[i])]}
                    for i, c in enumerate(lens["clusters"])]
        m = {"project": project, "intent": lens["intent"], "ids": ids.tolist(), "xy": xy.round(5).tolist(),
             "cluster": labels.tolist(), "cluster_sim": csim.round(4).tolist(),
             "intent_score": intent.round(4).tolist(), "clusters": clusters, "built_seconds": round(time.time() - t, 1)}
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(m))
        return m

    def edit_lens(self, body: dict) -> dict:
        lens = lenses.read(body["project"])
        if "intent" in body:
            lens["intent"] = body["intent"].strip()
        if body.get("add"):
            label = body["add"]["label"].strip()
            if not label:
                raise ValueError("a cluster needs a name")
            lens["clusters"] = [c for c in lens["clusters"] if c["label"].lower() != label.lower()]
            lens["clusters"].append({"label": label, "description": body["add"].get("description", "").strip()})
        if body.get("remove"):
            lens["clusters"] = [c for c in lens["clusters"] if c["label"] != body["remove"]]
        lenses.write(lens)
        return self.build_lens(body["project"])

    # ------------------------------------------------------------------ the wall

    def wall(self, project: str) -> dict:
        """Islands of moments for the board: a project's clusters, or the automatic ones ("dailies").

        Also returns `links`: pairs of islands whose moments resemble each other most (a minimum
        spanning tree over island centres), so the board can tie related islands together."""
        ids, mat = self.vecs()
        row = {int(i): n for n, i in enumerate(ids)}
        if project:
            m = self.build_lens(project)
            sim = np.asarray(m["cluster_sim"])
            cols = []
            for c in m["clusters"]:
                members = [i for i, lab in enumerate(m["cluster"]) if lab == c["id"]]
                members.sort(key=lambda i: -sim[i])
                cols.append({"id": c["id"], "label": c["label"], "description": c["description"],
                             "ids": [m["ids"][i] for i in members]})
            loose = [m["ids"][i] for i, lab in enumerate(m["cluster"]) if lab == -1]
            cards = [m["ids"][i] for i, lab in enumerate(m["cluster"]) if lab == -2]
            intent = m["intent"]
        else:
            b = self.base or self.build_base()
            cols = [{"id": c["id"], "label": c["label"], "description": "",
                     "ids": [b["ids"][i] for i, lab in enumerate(b["cluster"]) if lab == c["id"]]} for c in b["clusters"]]
            loose = [b["ids"][i] for i, lab in enumerate(b["cluster"]) if lab == -1]
            cards, intent = [], ""
        links = []
        live = [c for c in cols if c["ids"]]
        if len(live) > 1:
            cen = np.stack([mat[[row[i] for i in c["ids"]]].mean(0) for c in live])
            cen /= np.linalg.norm(cen, axis=1, keepdims=True)
            d = 1 - cen @ cen.T
            inside, todo = {0}, set(range(1, len(live)))
            while todo:  # Prim's minimum spanning tree
                a, b = min(((a, b) for a in inside for b in todo), key=lambda e: d[e])
                links.append({"a": live[a]["id"], "b": live[b]["id"], "sim": round(float(1 - d[a, b]), 3)})
                inside.add(b)
                todo.remove(b)
        return {"project": project, "intent": intent, "columns": cols, "loose": loose, "cards": cards, "links": links}

    def frame(self, mid: int, k: int) -> bytes:
        """Still k of FRAMES_PER_MOMENT from a moment, cut from the source file once and cached."""
        import av
        from PIL import Image
        cache = HOME / "frames" / f"{mid}-{k}.jpg"
        if cache.exists():
            return cache.read_bytes()
        row = self.db().execute("select m.t0, m.t1, c.path from moments m join clips c on c.id = m.clip_id "
                                "where m.id = ?", (mid,)).fetchone()
        t = row["t0"] + (k + 0.5) * (row["t1"] - row["t0"]) / FRAMES_PER_MOMENT
        with av.open(row["path"]) as c:
            vs = c.streams.video[0]
            c.seek(int(max(0, t - 1) / vs.time_base), stream=vs, backward=True)
            img = None
            for f in c.decode(vs):
                img = f
                if f.time is not None and f.time >= t:
                    break
        im = img.to_image()
        im.thumbnail((480, 480))
        cache.parent.mkdir(exist_ok=True)
        im.save(cache, quality=84)
        return cache.read_bytes()

    # ------------------------------------------------------------------ shared

    def footage(self, ids: np.ndarray) -> np.ndarray:
        """Clef-Omni's footage probability per moment, nan where it has not judged the moment."""
        tags = {r[0]: json.loads(r[1]).get("footage") for r in
                self.db().execute("select id, tags from moments where tags is not null")}
        return np.array([np.nan if tags.get(int(i)) is None else tags[int(i)] for i in ids], dtype=float)

    def aspect(self, path: str) -> float:
        """Display aspect ratio (width / height) of a clip, read once from the file."""
        cache = self.__dict__.setdefault("_aspects", {})
        if path not in cache:
            import av
            try:
                with av.open(path) as c:
                    v = c.streams.video[0].codec_context
                    sar = v.sample_aspect_ratio or 1
                    cache[path] = round(float(v.width * sar / v.height), 4)
            except Exception:
                cache[path] = 16 / 9
        return cache[path]

    def moments(self) -> dict:
        db = self.db()
        clips, points = {}, []
        for r in db.execute("select m.id, m.t0, m.t1, m.tags, c.path, c.id as clip_id from moments m "
                            "join clips c on c.id = m.clip_id order by m.id"):
            if r["clip_id"] not in clips:
                clips[r["clip_id"]] = {"id": r["clip_id"], "name": clip_name(r["path"]), "ar": self.aspect(r["path"])}
            points.append({"id": r["id"], "clip": r["clip_id"], "t0": r["t0"], "t1": r["t1"],
                           "tags": json.loads(r["tags"]) if r["tags"] else {}})
        return {"points": points, "clips": list(clips.values())}

    def search(self, q: str, k: int = 24) -> dict:
        ids, mat = self.vecs()
        t = time.time()
        sims = mat @ self.emb().queries([q])[0]
        ms = (time.time() - t) * 1000
        card = lenses.cards(mat, self.emb().queries([lenses.CARD_TEXT])[0], self.footage(ids))
        top = [j for j in np.argsort(-sims) if not card[j]][:k]
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


def stringout_read(project: str) -> dict:
    note = lenses.project_dir(project) / "String-out.md"
    ids = [int(x) for x in re.findall(r"\[moment (\d+)\]", note.read_text())] if note.exists() else []
    return {"project": project, "ids": ids}


def stringout_write(project: str, ids: list[int], ex: "Explorer") -> dict:
    """Save the string-out as an Obsidian note: one numbered line per select, in order."""
    from .vault import tc
    rows = {r["id"]: r for r in ex.db().execute(
        f"select m.id, m.t0, m.t1, c.path from moments m join clips c on c.id = m.clip_id "
        f"where m.id in ({','.join('?' * len(ids)) or 'null'})", ids)}
    total = sum(rows[i]["t1"] - rows[i]["t0"] for i in ids if i in rows)
    lines = ["---", "type: string-out", "driver: human", "status: Draft", f"running_time: {tc(total)}", "---",
             "# String-out", "", "Selects in order, built on the wall. Each line keeps its moment number so the wall "
             "can read it back.", ""]
    for n, i in enumerate(ids, 1):
        r = rows.get(i)
        if r:
            lines.append(f"{n}. [moment {i}] {clip_name(r['path'])} · {tc(r['t0'])}–{tc(r['t1'])}")
    (lenses.project_dir(project) / "String-out.md").write_text("\n".join(lines) + "\n")
    return {"project": project, "ids": ids, "running_time": total}


def clip_name(path: str) -> str:
    s = re.sub(r"\(.*?\)|#\w+|^(YTDown\.com_YouTube_|ssvid\.net-+)|_\d+p$|_Media_\w+", "", Path(path).stem)
    s = re.sub(r"[-_]+", " ", s).strip()
    s = re.sub(r"\s+\d{3,4}p?$|\s+\d{3}$", "", s)
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

    def fail(self, e: Exception):
        self.send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if urlparse(self.path).path == "/api/lens":
                return self.json(self.ex.edit_lens(body))
            if urlparse(self.path).path == "/api/stringout":
                return self.json(stringout_write(body["project"], body["ids"], self.ex))
            self.send(404, b"not found", "text/plain")
        except Exception as e:
            self.fail(e)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                return self.send(200, WALL.read_bytes(), "text/html; charset=utf-8")
            if u.path == "/map":
                return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/wall":
                return self.json(self.ex.wall(q.get("project", [""])[0]))
            if u.path == "/api/stringout":
                return self.json(stringout_read(q["project"][0]))
            if m := re.match(r"^/frame/(\d+)/(\d+)$", u.path):
                return self.send(200, self.ex.frame(int(m.group(1)), int(m.group(2))), "image/jpeg",
                                 {"Cache-Control": "max-age=86400"})
            if u.path == "/api/moments":
                return self.json(self.ex.moments())
            if u.path == "/api/projects":
                return self.json(lenses.projects())
            if u.path == "/api/map":
                p = q.get("project", [""])[0]
                return self.json(self.ex.build_lens(p) if p else (self.ex.base or self.ex.build_base()))
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
            self.fail(e)

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
    m = ex.build_base()
    print(f"  {len(m['ids'])} moments, {len(m['clusters'])} clusters")
    Handler.ex = ex
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"explorer: http://127.0.0.1:{port}", flush=True)
    srv.serve_forever()
