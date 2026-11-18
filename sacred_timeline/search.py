"""Find moments that match free-form descriptions.

Two stages, so search time does not grow with the size of the library:

  1. recall   each description is embedded (EmbeddingGemma 2) and compared with every moment's
              vector; one matrix product, milliseconds even for a million moments.
  2. judge    the top `pool` moments per description go to the decision model (Clef-Omni) as a
              yes/no question, together with the standard tags. All descriptions in one search
              share that pass, and every answer is cached, so repeats are instant.

Tag filters (`--where emotion=angry`) apply to the judged shortlist; with filters the shortlist
is widened so enough moments survive them.
"""
from __future__ import annotations

import json
import operator
import re
import sqlite3
import time
from dataclasses import dataclass, field

import numpy as np

from .index import ClipCache, ask, vectors
from .models import DecisionModel
from .tags import TAGS, flatten

OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt, "!=": operator.ne, "=": operator.eq}


def question(text: str) -> dict:
    return {"type": "noul", "instructions": f"Does this moment of the film match this description: {text.strip()}?"}


def parse_where(exprs: list[str]):
    """`emotion=angry`, `tension>0.6`, `people!=none` -> list of predicates on a tags dict."""
    preds = []
    for e in exprs:
        m = re.match(r"\s*(\w+)\s*(>=|<=|!=|=|>|<)\s*(.+?)\s*$", e)
        if not m:
            raise ValueError(f"cannot read filter {e!r}; use e.g. emotion=angry or tension>0.6")
        key, op, val = m.groups()
        try:
            val = float(val)
        except ValueError:
            pass
        preds.append((key, OPS[op], val))
    return preds


@dataclass
class Hit:
    moment_id: int
    clip: str
    t0: float
    t1: float
    thumb: str
    score: float  # decision model's yes probability (or the embedding similarity if not judged)
    similarity: float  # embedding cosine similarity
    tags: dict = field(default_factory=dict)


def recall(db: sqlite3.Connection, embedder, queries: list[str], pool: int,
           include_cards: bool = False) -> tuple[np.ndarray, dict[str, list]]:
    ids, mat = vectors(db)
    qv = embedder.queries(queries)
    sims = qv @ mat.T
    if not include_cards:  # end cards and promo graphics never make the shortlist
        from .lenses import CARD_TEXT, cards
        tags = {r[0]: json.loads(r[1]).get("footage") for r in db.execute("select id, tags from moments where tags is not null")}
        footage = np.array([np.nan if tags.get(int(i)) is None else tags[int(i)] for i in ids], dtype=float)
        sims[:, cards(mat, embedder.queries([CARD_TEXT])[0], footage)] = -np.inf
    out = {}
    for i, q in enumerate(queries):
        order = np.argsort(-sims[i])[:pool]
        out[q] = [(int(ids[j]), float(sims[i, j])) for j in order if np.isfinite(sims[i, j])]
    return qv, out


def search(db: sqlite3.Connection, embedder, model: DecisionModel | None, queries: list[str], where=(),
           top: int = 5, pool: int = 24, include_cards: bool = False, log=print) -> dict[str, list[Hit]]:
    where = list(where)
    if where:
        pool *= 3
    t = time.time()
    _, shortlist = recall(db, embedder, queries, pool, include_cards)
    log(f"  recall: {sum(len(v) for v in shortlist.values())} candidates in {(time.time() - t) * 1000:.0f} ms")
    cand = sorted({m for v in shortlist.values() for m, _ in v})
    rows = {r["id"]: r for r in db.execute(
        f"select m.id, m.t0, m.t1, m.thumb, m.tags, c.path from moments m join clips c on c.id = m.clip_id "
        f"where m.id in ({','.join('?' * len(cand))})", cand)}

    judged: dict[int, dict] = {}
    tags_of = {mid: json.loads(r["tags"]) if r["tags"] else {} for mid, r in rows.items()}
    if model is not None:
        qs = {f"q{i}": question(q) for i, q in enumerate(queries)} | TAGS
        t = time.time()
        judged = ask(db, model, cand, qs, ClipCache(model), log=log)
        for mid in cand:
            tags = tags_of[mid] = flatten({k: judged[mid][k] for k in TAGS})
            db.execute("update moments set tags = ? where id = ?", (json.dumps(tags), mid))
        db.commit()
        log(f"  judge: {len(cand)} moments in {time.time() - t:.1f} s")
    preds = parse_where(where)
    if model is not None and not include_cards:
        preds.append(("footage", operator.ge, 0.5))

    results = {}
    for i, q in enumerate(queries):
        hits = []
        for mid, sim in shortlist[q]:
            r = rows[mid]
            tags = tags_of[mid]
            if preds and not all(k in tags and f(tags[k], v) for k, f, v in preds):
                continue
            score = judged[mid][f"q{i}"]["noul"] if judged else sim
            hits.append(Hit(mid, r["path"], r["t0"], r["t1"], r["thumb"], score, sim, tags))
        hits.sort(key=lambda h: (-h.score, -h.similarity))
        picked: list[Hit] = []
        for h in hits:
            # Skip a moment directly next to one already picked from the same clip: same scene twice.
            if any(p.clip == h.clip and (abs(p.t0 - h.t1) < 0.01 or abs(p.t1 - h.t0) < 0.01) for p in picked):
                continue
            picked.append(h)
            if len(picked) == top:
                break
        results[q] = picked
    return results
