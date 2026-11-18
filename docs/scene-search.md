# Scene search

Find moments in your own movie and TV files by describing them, then review the picks in Obsidian
or on the moment map.

```
python -m sacred_timeline index ~/Movies                    # once per clip, about 70x real time
python -m sacred_timeline find "two friends arguing" --where tension>0.5
python -m sacred_timeline story "creator-vault/Projects/<project>/Script.md"
python -m sacred_timeline explore                           # moment map at http://127.0.0.1:8777
```

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
