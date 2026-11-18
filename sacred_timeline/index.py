"""The scene index.

Indexing is fast and model-light: each clip is decoded once (about 175x real time), split into
moments at shot cuts (long shots split into 4 s pieces), and each moment gets one EmbeddingGemma 2
vector from its frames and sound (about 100x real time on an M5 Max).

The decision model runs only at search time, on a shortlist, and every answer it gives is cached
in SQLite keyed by model and by the exact question. Asking a question again is free; asking ten new
questions at once costs about the same as one, because they share one pass over the moment.

Layout under SACRED_TIMELINE_HOME (default ~/Movies/sacred-timeline/index):
  index.db       clips, moments (with vectors), answers
  thumbs/        one JPEG per moment
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .models import FPS, SR, DecisionModel, Moment

HOME = Path(os.environ.get("SACRED_TIMELINE_HOME", Path.home() / "Movies" / "sacred-timeline" / "index"))
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv"}
MAX_MOMENT = 4.0  # seconds; longer shots are split
MIN_MOMENT = 1.0  # seconds; shorter shots are merged into the previous one
CUT = 0.45  # colour-histogram distance (0-2) between sampled frames that counts as a cut

SCHEMA = """
create table if not exists clips (id integer primary key, path text unique, mtime real, duration real,
                                  embed_model text, index_seconds real);
create table if not exists moments (id integer primary key, clip_id integer, t0 real, t1 real, thumb text,
                                    vec blob, tags text, unique (clip_id, t0));
create table if not exists answers (moment_id integer, model text, qkey text, question text, answer text,
                                    primary key (moment_id, model, qkey));
create index if not exists moments_clip on moments (clip_id);
"""


def connect(home: Path = HOME) -> sqlite3.Connection:
    home.mkdir(parents=True, exist_ok=True)
    (home / "thumbs").mkdir(exist_ok=True)
    db = sqlite3.connect(home / "index.db")
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def qkey(question: dict) -> str:
    return hashlib.sha1(json.dumps(question, sort_keys=True).encode()).hexdigest()[:16]


def find_clips(roots: list[Path]) -> list[Path]:
    out = []
    for root in roots:
        files = [root] if root.is_file() else sorted(root.rglob("*"))
        for p in files:
            if p.suffix.lower() in VIDEO_EXT and ".annotated." not in p.name and "sacred-timeline" not in p.parts \
                    and not any(part.endswith((".tvlibrary", ".photoslibrary")) for part in p.parts):
                out.append(p)
    return out


def slug(clip: Path) -> str:
    s = re.sub(r"\(.*?\)|#\w+|^(YTDown\.com_YouTube_|ssvid\.net-+)", "", clip.stem)
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:48].strip("-") or "clip"


# --------------------------------------------------------------------------- shots


def shots(frames: np.ndarray, fps: float) -> list[tuple[int, int]]:
    """Frame ranges [a, b), one per shot, with long shots split and tiny ones merged."""
    hists = []
    for f in frames:
        small = np.asarray(Image.fromarray(f).resize((64, 36))) // 32  # 8 levels per channel
        idx = small[..., 0].astype(int) * 64 + small[..., 1] * 8 + small[..., 2]
        h = np.bincount(idx.ravel(), minlength=512).astype(float)
        hists.append(h / h.sum())
    cuts = [0] + [i for i in range(1, len(hists)) if np.abs(hists[i] - hists[i - 1]).sum() > CUT] + [len(frames)]
    ranges = []
    for a, b in zip(cuts, cuts[1:]):
        if ranges and (b - a) < MIN_MOMENT * fps:
            ranges[-1] = (ranges[-1][0], b)
            continue
        ranges.append((a, b))
    out, step = [], MAX_MOMENT * fps
    for a, b in ranges:
        n = max(1, round((b - a) / step))
        edges = np.linspace(a, b, n + 1).round().astype(int)
        out += [(int(x), int(y)) for x, y in zip(edges, edges[1:]) if y > x]
    return out


# --------------------------------------------------------------------------- indexing


def needs_index(db: sqlite3.Connection, clip: Path, embed_id: str) -> bool:
    row = db.execute("select mtime, embed_model from clips where path = ?", (str(clip.resolve()),)).fetchone()
    return not (row and row["mtime"] == clip.resolve().stat().st_mtime and row["embed_model"] == embed_id)


def index_clips(db: sqlite3.Connection, embedder, clips: list[Path], home: Path = HOME, log=print) -> int:
    """Index many clips, decoding the next clip on a CPU thread while the GPU embeds the current one."""
    from concurrent.futures import ThreadPoolExecutor
    from .embed import decode
    todo = [c for c in clips if needs_index(db, c, embedder.id)]
    log(f"{len(clips)} clips, {len(clips) - len(todo)} already indexed, {len(todo)} to index")
    n = 0
    with ThreadPoolExecutor(1) as pool:
        nxt = pool.submit(decode, todo[0]) if todo else None
        for i, clip in enumerate(todo):
            t = time.time()
            decoded = nxt.result()
            nxt = pool.submit(decode, todo[i + 1]) if i + 1 < len(todo) else None
            log(f"[{i + 1}/{len(todo)}] {clip.name}")
            n += index_clip(db, embedder, clip, home, log, decoded=decoded, started=t)
    return n


def index_clip(db: sqlite3.Connection, embedder, clip: Path, home: Path = HOME, log=print,
               decoded=None, started: float | None = None) -> int:
    """Decode, cut, thumbnail and embed one clip. Skips clips already indexed and unchanged."""
    from .embed import DECODE_FPS, EMBED_STEP, decode
    clip = clip.resolve()
    mtime = clip.stat().st_mtime
    if not needs_index(db, clip, embedder.id):
        return 0
    row = db.execute("select id from clips where path = ?", (str(clip),)).fetchone()
    if row:
        db.execute("delete from answers where moment_id in (select id from moments where clip_id = ?)", (row["id"],))
        db.execute("delete from moments where clip_id = ?", (row["id"],))
        db.execute("delete from clips where id = ?", (row["id"],))
    t = started or time.time()
    frames, audio = decoded or decode(clip)
    ranges = shots(frames, DECODE_FPS)
    items = []
    for a, b in ranges:
        sub = frames[a:b][::EMBED_STEP] if b - a > 1 else frames[a:b]
        aud = audio[int(a / DECODE_FPS * SR):int(b / DECODE_FPS * SR)] if audio is not None else None
        items.append((sub, aud))
    vecs = embedder.moments(items)
    dur = len(frames) / DECODE_FPS
    cur = db.execute("insert into clips (path, mtime, duration, embed_model, index_seconds) values (?, ?, ?, ?, ?)",
                     (str(clip), mtime, dur, embedder.id, round(time.time() - t, 2)))
    for (a, b), v in zip(ranges, vecs):
        thumb = home / "thumbs" / f"{slug(clip)}-{cur.lastrowid}-{a / DECODE_FPS:07.1f}.jpg"
        Image.fromarray(frames[(a + b) // 2]).save(thumb, quality=82)
        db.execute("insert into moments (clip_id, t0, t1, thumb, vec) values (?, ?, ?, ?, ?)",
                   (cur.lastrowid, a / DECODE_FPS, b / DECODE_FPS, str(thumb), v.astype(np.float16).tobytes()))
    db.commit()
    dt = time.time() - t
    log(f"  {dur:.0f} s -> {len(ranges)} moments in {dt:.1f} s ({dur / dt:.0f}x real time)")
    return len(ranges)


def vectors(db: sqlite3.Connection) -> tuple[np.ndarray, np.ndarray]:
    """(moment ids, (n, 768) float32 matrix) for every embedded moment."""
    rows = db.execute("select id, vec from moments where vec is not null order by id").fetchall()
    if not rows:
        return np.zeros(0, int), np.zeros((0, 768), np.float32)
    ids = np.array([r[0] for r in rows])
    mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float16).reshape(len(rows), -1).astype(np.float32)
    return ids, mat


# --------------------------------------------------------------------------- decision model, cached


class ClipCache:
    """Keeps the most recently decoded clip, since answering walks moments clip by clip."""

    def __init__(self, model: DecisionModel):
        self.model, self.path, self.data = model, None, None

    def moment(self, row: sqlite3.Row) -> Moment:
        path = Path(row["path"])
        if path != self.path:
            self.path, self.data = path, self.model.load_clip(path)
        frames, audio = self.data
        a, b = round(row["t0"] * FPS), max(round(row["t1"] * FPS), round(row["t0"] * FPS) + 1)
        aud = audio[int(row["t0"] * SR):int(row["t1"] * SR)] if audio is not None else None
        return Moment(path, row["t0"], row["t1"], frames[a:b], aud)


def ask(db: sqlite3.Connection, model: DecisionModel, moment_ids: list[int], questions: dict[str, dict],
        cache: ClipCache, log=print) -> dict[int, dict[str, dict]]:
    """Answers to every question for every moment, from the cache where possible.

    Moments missing any answer get one model call carrying all their missing questions."""
    keys = {name: qkey(q) for name, q in questions.items()}
    out: dict[int, dict[str, dict]] = {m: {} for m in moment_ids}
    by_key = {v: k for k, v in keys.items()}
    for mid in moment_ids:
        for r in db.execute("select qkey, answer from answers where moment_id = ? and model = ?", (mid, model.id)):
            if r["qkey"] in by_key:
                out[mid][by_key[r["qkey"]]] = json.loads(r["answer"])
    todo = [m for m in moment_ids if len(out[m]) < len(questions)]
    if not todo:
        return out
    rows = db.execute(
        f"select m.id, m.t0, m.t1, c.path from moments m join clips c on c.id = m.clip_id "
        f"where m.id in ({','.join('?' * len(todo))}) order by c.path, m.t0", todo).fetchall()
    t = time.time()
    for n, r in enumerate(rows, 1):
        missing = {k: questions[k] for k in questions if k not in out[r["id"]]}
        ans = model.ask(cache.moment(r), missing)
        for name, a in ans.items():
            out[r["id"]][name] = a
            db.execute("insert or replace into answers values (?, ?, ?, ?, ?)",
                       (r["id"], model.id, keys[name], json.dumps(questions[name]), json.dumps(a)))
        if n % 10 == 0 or n == len(rows):
            db.commit()
            log(f"    judged {n}/{len(rows)} moments, {(time.time() - t) / n:.2f} s each")
    db.commit()
    return out
