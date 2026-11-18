"""sacred-timeline: find scenes in your own movie and TV files with a local decision model.

  index    PATH...               cut clips into moments and embed each one (once per clip)
  find     "description" ...     best moments for each description
  story    NOTE.md               best moments for every 'Clip wanted' beat in a script note
  explore                        open the embedding map in your browser
  tags                           the standard tags you can filter on

Search recalls a shortlist with EmbeddingGemma 2 (milliseconds), then Clef-Omni judges only that
shortlist. --no-judge skips Clef-Omni for an instant, rougher answer.

Examples:
  python -m sacred_timeline index ~/Movies
  python -m sacred_timeline find "a man showing off weapons" --where people!=none
  python -m sacred_timeline find "an awkward hello" --where emotion=awkward --note
  python -m sacred_timeline story "creator-vault/Projects/Sample video/Script.md"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(prog="sacred_timeline", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="clef-omni")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index")
    p.add_argument("paths", nargs="+", type=Path)
    for name in ("find", "story"):
        p = sub.add_parser(name)
        if name == "find":
            p.add_argument("queries", nargs="+")
            p.add_argument("--note", action="store_true", help="also write a note to creator-vault/Searches")
        else:
            p.add_argument("script", type=Path)
        p.add_argument("--where", action="append", default=[], help="tag filter, e.g. emotion=angry, tension>0.6")
        p.add_argument("--top", type=int, default=3)
        p.add_argument("--pool", type=int, default=24, help="moments per description sent to the judge")
        p.add_argument("--no-judge", action="store_true", help="embedding similarity only; skips Clef-Omni")
        p.add_argument("--include-cards", action="store_true", help="also search end cards and promo graphics")
        p.add_argument("--json", action="store_true")
    p = sub.add_parser("explore")
    p.add_argument("--port", type=int, default=8777)
    sub.add_parser("tags")
    args = ap.parse_args()

    if args.cmd == "tags":
        from .tags import TAGS
        for k, q in TAGS.items():
            vals = ", ".join(q["criteria"]) if q["type"] == "choice" else ("0-2" if q["type"] == "score" else "0-1")
            print(f"{k:10} {q['type']:6} {vals}")
        return

    from .embed import Embedder
    from .index import connect, find_clips, index_clips
    from .models import load_model
    db = connect()
    if args.cmd == "explore":
        from .explore import serve
        serve(db, args.port)
        return
    t = time.time()
    embedder = Embedder()
    print(f"loaded {embedder.id} in {time.time() - t:.1f} s", file=sys.stderr)

    if args.cmd == "index":
        clips = find_clips([p.expanduser() for p in args.paths])
        t = time.time()
        index_clips(db, embedder, clips)
        dur = db.execute("select sum(duration) from clips").fetchone()[0] or 0
        n = db.execute("select count(*) from moments").fetchone()[0]
        print(f"index: {dur / 60:.1f} min of footage, {n} moments; this run took {time.time() - t:.1f} s")
        return

    model = None
    if not args.no_judge:
        t = time.time()
        model = load_model(args.model)
        print(f"loaded {model.id} in {time.time() - t:.1f} s", file=sys.stderr)

    from .search import search
    from .vault import VAULT, clip_beats, tc, tag_line, write_note
    if args.cmd == "find":
        beats = [(q, q) for q in args.queries]
    else:
        beats = clip_beats(args.script)
        if not beats:
            sys.exit(f"no 'Clip wanted' column or bullet list found in {args.script}")
    t = time.time()
    res = search(db, embedder, model, [d for _, d in beats], args.where, args.top, args.pool, args.include_cards,
                 log=lambda s: print(s, file=sys.stderr))
    print(f"searched in {time.time() - t:.1f} s", file=sys.stderr)
    if args.json:
        print(json.dumps({d: [h.__dict__ for h in hits] for d, hits in res.items()}, indent=1))
    else:
        for label, d in beats:
            print(f"\n{label}" + (f"  ({d})" if d != label else ""))
            for h in res[d] or []:
                print(f"  {h.score:4.0%} (sim {h.similarity:.2f})  {Path(h.clip).name[:50]}  {tc(h.t0)}–{tc(h.t1)}  {tag_line(h.tags)}")
            if not res[d]:
                print("  no match")
    sections = [(label, d, res[d]) for label, d in beats]
    note = None
    if args.cmd == "story":
        note = write_note(args.script.with_name("Selects.md"), f"Selects for {args.script.stem}", sections, (model or embedder).id)
    elif args.note:
        name = re.sub(r"[^\w ,'-]+", "", args.queries[0])[:60].strip()
        note = write_note(VAULT / "Searches" / f"{name}.md", f"Search: {', '.join(args.queries)}", sections, (model or embedder).id)
    if note:
        print(f"\nnote: {note}", file=sys.stderr)


if __name__ == "__main__":
    main()
