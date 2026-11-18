# Scene search

Find moments in your own movie and TV files by describing them, then review the picks in Obsidian
or on the moment map.

```
python -m sacred_timeline index ~/Movies                    # once per clip, about 70x real time
python -m sacred_timeline find "two friends arguing" --where tension>0.5
python -m sacred_timeline story "creator-vault/Projects/<project>/Script.md"
python -m sacred_timeline explore                           # the wall at http://127.0.0.1:8777
```

## The board

Modelled on the evidence walls in film and TV, and on Walter Murch's walls of stills. Pick the wall
from the top bar: light table (default), midnight leather, black foam core, concrete, green felt, kraft paper or cork.
Photos keep their source's full frame and original colours; only their size varies.

- **Board**: one per project. "Dailies" shows everything, grouped automatically.
- **The angle**: a front-page clipping on the left with the project's intent. Edit the headline and the
  groups re-sort.
- **Groups**: one dense column each, colour-coded like a Homeland wall: every printout in a group is
  marked up in that group's colour (coloured tape and marker scribbles on bare prints of every size). The title is a ransom note cut from magazines; a
  newspaper clipping reports the facts (how many shots, how long, which sources, the mood once checked).
  Groups come from `Projects/<project>/Clusters.md`.
- **Threads** leave the angle, run along the top of the cork wound round pins, and drop to each title,
  so they never cross the text.
- **Pull**: describe a moment and it gets its own orange column at the front. "Pin it up" keeps it.
  The sticky pad adds a group by hand.
- **Skim** a photo by moving across it; click to play. Drag anything anywhere; "Tidy up" resets.
- **Ask Clef**: Clef-Omni circles a fit score on each printout in red marker and crosses out weak ones.
- **Telling them apart**: a pinned **Key** lists every group with its colour. Hover a group in the key
  (or its title, clipping or notes) to light it up and dim the rest; click to keep it lit, Esc or a click
  on empty cork to let go. The real matches carry numbered evidence markers in the group's colour,
  best first; anything without a number is extra material. Hovering any photo or sheet shows a tag with
  its group, its rank (or what kind of extra it is), the source and the timecode.
- **Filled wall**: every gap is filled from the footage itself (other frames of a group's moments,
  contact sheets, typed shot logs with highlighted and redacted lines, stray notes), so almost no cork shows.

The older embedding map is still at `/map`.

## How it stays fast

| Stage | Model | When it runs | Cost |
|---|---|---|---|
| Cut | colour histogram | index | about 175x real time decode, cuts at 1 fps |
| Recall | EmbeddingGemma 2 (frames at 0.5 fps + sound, 70 tokens a frame) | index, then one matrix product per search | about 70x real time to index; about 70 ms per search over 900,000 moments |
| Judge | Clef-Omni 8-bit | search, top 24 per description only | about 0.75 s per moment, cached forever |

Indexing never runs the decision model. A search only runs it on its shortlist, so search time does
not grow with the size of the library. Each answer is cached by model and by the exact question, so
asking again is instant and editing a question re-asks only that question.

Measured on an M5 Max with 30 minutes of footage: 720 moments indexed in 27 s, including model load.
That projects to about 15 hours of indexing for 1,000 hours of footage, done once, in the background.

## Plugging in another model

Everything talks to two small interfaces in `sacred_timeline/`:

- **Decision model** (`models.py`): a class with an `id`, `load_clip(path)` and
  `ask(moment, questions) -> answers`, using the Decisions question format (choice, score, noul).
  Register it in `MODELS`, then pass `--model <name>`.
- **Embedder** (`embed.py`): `moments([(frames, audio), ...])` and `queries([text, ...])`, both
  returning unit vectors in one shared space.

Answers are stored with the model's `id`, so results from different models never mix.

## Standard tags

`python -m sacred_timeline tags` lists them: footage (vs end card), people, shot, setting, speaking,
emotion, intensity, voice, tension, funny, action. Clef-Omni fills them in for each moment it
judges, and `--where` filters on them.

## Where things live

- `~/Movies/sacred-timeline/index/`: `index.db` (clips, moments, vectors, cached answers),
  `thumbs/`, `map.json`. Outside the repo, because it holds frames from your films.
- `creator-vault/**/selects/`: stills copied in for Obsidian notes. Git-ignored for the same reason.
