"""Project lenses: clusters defined by a project's intent instead of found automatically.

The automatic clusters on the moment map answer "what does this footage look like?". A lens answers
"how does this footage serve *this* video?". Each project keeps its lens as a note in the Obsidian
vault, so it can be edited there or from the map:

  creator-vault/Projects/<project>/Clusters.md

  ---
  type: clusters
  intent: Confident men who are less in control than they think.
  ---
  # Clusters
  - **Bravado**: a man boasting or showing off with total confidence
  - **Cracks**: someone embarrassed, flustered or caught out

Each cluster's name and description is embedded in the same space as the moments. A moment joins
the cluster it stands out most for (its similarity compared with how close every other moment is
to that cluster), or stays unassigned if it stands out for none. The map is then laid out around
those clusters, starting from the automatic layout so points move rather than jump.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .vault import VAULT

PROJECTS = VAULT / "Projects"
NOTE = "Clusters.md"
STANDOUT = 0.6  # standard deviations above a cluster's average similarity needed to join it
CARD_TEXT = ("a promotional end card, channel logo or subscribe screen with text and graphics, "
             "not a scene from the film")
CARD_Z = 1.0  # moments this far above average similarity to CARD_TEXT are treated as end cards


def project_dir(name: str) -> Path:
    d = (PROJECTS / name).resolve()
    if PROJECTS.resolve() not in d.parents:
        raise ValueError("unknown project")
    return d


def angle(project: Path) -> str:
    """The one-line angle from the project's Script.md, used as the default intent."""
    script = project / "Script.md"
    if script.exists():
        m = re.search(r"\*\*Angle:\*\*\s*(.+)", script.read_text())
        if m:
            return m.group(1).strip()
    return ""


def read(name: str) -> dict:
    d = project_dir(name)
    note = d / NOTE
    lens = {"project": name, "intent": angle(d), "clusters": [], "exists": note.exists()}
    if not note.exists():
        return lens
    text = note.read_text()
    fm = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if fm:
        m = re.search(r"^intent:\s*(.+)$", fm.group(1), re.M)
        if m:
            lens["intent"] = m.group(1).strip().strip('"')
    for line in text.splitlines():
        m = re.match(r"^\s*[-*]\s+(?:\*\*(.+?)\*\*|([^:]+?))\s*(?::\s*(.+))?\s*$", line)
        if m:
            label = (m.group(1) or m.group(2)).strip()
            lens["clusters"].append({"label": label, "description": (m.group(3) or "").strip()})
    return lens


def write(lens: dict) -> None:
    d = project_dir(lens["project"])
    intent = lens.get("intent", "").replace("\n", " ").strip()
    lines = ["---", "type: clusters", f"intent: {intent}", "---", "# Clusters", "",
             "Each line is one cluster for this project: **name**: what belongs in it. "
             "Edit here or on the moment map.", ""]
    for c in lens["clusters"]:
        desc = c.get("description", "").replace("\n", " ").strip()
        lines.append(f"- **{c['label']}**" + (f": {desc}" if desc else ""))
    (d / NOTE).write_text("\n".join(lines) + "\n")


def projects() -> list[dict]:
    out = []
    for d in sorted(p for p in PROJECTS.iterdir() if p.is_dir()) if PROJECTS.exists() else []:
        lens = read(d.name)
        out.append({"project": d.name, "intent": lens["intent"], "n": len(lens["clusters"]),
                    "exists": lens["exists"]})
    return out


def key(lens: dict, ids: list[int]) -> str:
    blob = json.dumps([lens["intent"], lens["clusters"], ids], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def cards(mat: np.ndarray, card_vec: np.ndarray, footage: np.ndarray | None = None) -> np.ndarray:
    """True for moments that are end cards or promo graphics rather than film.

    Uses the decision model's `footage` tag where a moment has one (nan where not), and otherwise
    how much more a moment resembles CARD_TEXT than the library does on average."""
    s = mat @ card_vec
    z = (s - s.mean()) / max(s.std(), 1e-6)
    out = z >= CARD_Z
    if footage is not None:
        known = ~np.isnan(footage)
        out[known] = footage[known] < 0.5
    return out


def assign(mat: np.ndarray, cvecs: np.ndarray, skip: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(cluster index per moment, -1 for none, -2 for skipped; similarity to that cluster).

    Each cluster's similarities are compared only across the moments not skipped, so a block of
    end cards cannot make film moments look ordinary, or vice versa."""
    n = len(mat)
    skip = np.zeros(n, bool) if skip is None else skip
    labels, sims = np.full(n, -1), np.zeros(n)
    if len(cvecs) and (~skip).any():
        s = mat @ cvecs.T
        keep = ~skip
        mu, sd = s[keep].mean(0, keepdims=True), np.maximum(s[keep].std(0, keepdims=True), 1e-6)
        z = (s - mu) / sd
        best = z.argmax(1)
        labels = np.where(z[np.arange(n), best] >= STANDOUT, best, -1)
        sims = s[np.arange(n), best]
    labels[skip] = -2
    return labels, sims


def layout(mat: np.ndarray, cvecs: np.ndarray, labels: np.ndarray, ids: list[int]) -> np.ndarray:
    """A designed layout for a lens, in [0, 1] x [0, 1].

    Each cluster is an island on a ring; inside an island, similar moments sit near each other (the
    island's own first two principal components). Moments that fit no cluster sit inside the ring,
    pulled toward the clusters they most resemble. End cards get a small island in a corner.
    """
    n, k = len(mat), len(cvecs)
    xy = np.full((n, 2), 0.5)
    rng = np.random.default_rng(7)
    if k == 0:
        return xy
    angles = -np.pi / 2 + 2 * np.pi * np.arange(k) / k
    rx, ry = (0.36, 0.34) if k > 2 else (0.3, 0.2)
    centres = np.stack([0.5 + rx * np.cos(angles), 0.5 - ry * np.sin(angles)], 1)

    def island(members: np.ndarray, centre: np.ndarray, radius: float):
        pts = mat[members]
        if len(pts) == 1:
            xy[members] = centre
            return
        c = pts - pts.mean(0)
        u, sv, _ = np.linalg.svd(c, full_matrices=False)
        p2 = u[:, :2] * sv[:2]
        if p2.shape[1] < 2:
            p2 = np.c_[p2, np.zeros(len(p2))]
        p2 /= max(np.linalg.norm(p2, axis=1).max(), 1e-9)
        # spread to fill the disc evenly (rank-based radius keeps tight groups readable)
        r = np.linalg.norm(p2, axis=1)
        order = np.argsort(np.argsort(r))
        r_new = np.sqrt((order + 0.5) / len(r))
        p2 = p2 / np.maximum(r[:, None], 1e-9) * r_new[:, None]
        xy[members] = centre + p2 * radius

    for c in range(k):
        m = np.where(labels == c)[0]
        if len(m):
            island(m, centres[c], 0.035 + 0.016 * np.sqrt(len(m)))
    cards = np.where(labels == -2)[0]
    if len(cards):
        island(cards, np.array([0.95, 0.05]), 0.03 + 0.008 * np.sqrt(len(cards)))
    loose = np.where(labels == -1)[0]
    if len(loose):
        s = mat[loose] @ cvecs.T
        w = np.exp((s - s.max(1, keepdims=True)) * 40)
        w /= w.sum(1, keepdims=True)
        pull = w @ centres
        xy[loose] = 0.5 + (pull - 0.5) * 0.45 + rng.normal(0, 0.025, (len(loose), 2))
    return xy
