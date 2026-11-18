"""Write search results into the Obsidian vault as notes with stills, so picks can be reviewed there.

Stills are copied into a `selects/` folder next to the note. Those folders hold frames from films,
so they are git-ignored.
"""
from __future__ import annotations

import re
import shutil
from datetime import date
from pathlib import Path
from urllib.parse import quote

from .search import Hit

VAULT = Path(__file__).resolve().parent.parent / "creator-vault"


def tc(t: float) -> str:
    return f"{int(t) // 60:02d}:{t % 60:04.1f}"


def tag_line(tags: dict) -> str:
    bits = [tags.get("emotion", ""), {"none": "no people", "one": "1 person", "two": "2 people", "group": "group"}.get(tags.get("people"), ""),
            tags.get("shot", "") + " shot" if tags.get("shot") else "", tags.get("setting", "")]
    if tags.get("speaking", 0) >= 0.5 and tags.get("voice"):
        bits.append(f"{tags['voice']} voice")
    if tags.get("tension", 0) >= 0.5:
        bits.append("tense")
    if tags.get("funny", 0) >= 0.5:
        bits.append("funny")
    return " · ".join(b for b in bits if b)


def clip_beats(script: Path) -> list[tuple[str, str]]:
    """(beat, description) pairs from a script note: a table with a 'Clip wanted' column, or bullets."""
    lines = script.read_text().splitlines()
    beats, col, label_col = [], None, None
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")] if line.strip().startswith("|") else None
        if cells and col is None:
            low = [c.lower() for c in cells]
            if "clip wanted" in low:
                col = low.index("clip wanted")
                label_col = next((i for i, c in enumerate(low) if c not in ("#", "clip wanted")), None)
            continue
        if cells and col is not None:
            if set("".join(cells)) <= set("-: "):
                continue
            if col < len(cells) and cells[col]:
                label = cells[label_col] if label_col is not None and label_col < len(cells) else cells[col]
                beats.append((label, cells[col]))
            continue
        if col is None and re.match(r"\s*[-*]\s+\S", line):
            text = re.sub(r"^\s*[-*]\s+", "", line)
            beats.append((text, text))
    return beats


def write_note(note: Path, title: str, sections: list[tuple[str, str, list[Hit]]], model_id: str) -> Path:
    """sections: (heading, description, hits)."""
    folder = note.parent / "selects"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.jpg"):  # stills from an earlier run of this note
        old.unlink()
    out = ["---", "type: selects", "driver: ai", "status: Review", f"model: {model_id}",
           f"created: {date.today().isoformat()}", "---", f"# {title}", ""]
    for heading, desc, hits in sections:
        out += [f"## {heading}", f"*Looking for:* {desc}", ""]
        if not hits:
            out += ["No match in the index.", ""]
            continue
        out += ["| Still | Match | Where |", "|---|---|---|"]
        for h in hits:
            src = Path(h.thumb)
            shutil.copy2(src, folder / src.name)
            link = "file://" + quote(h.clip)
            out.append(f"| ![[selects/{src.name}\\|240]] | **{h.score:.0%}**<br>{tag_line(h.tags)} | "
                       f"[{Path(h.clip).stem[:40]}]({link})<br>{tc(h.t0)}–{tc(h.t1)} |")
        out.append("")
    note.write_text("\n".join(out))
    return note
