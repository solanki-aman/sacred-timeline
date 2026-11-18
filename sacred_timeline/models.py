"""The plug-in point: any decision model that can answer questions about a moment.

A moment is a short piece of a clip (frames sampled at 2 fps, plus its sound). A model gets a
moment and a dict of questions in the Decisions format (choice, score or noul) and returns one
answer per question. Everything else in sacred-timeline (index, search, overlays) talks only to
this interface, so another model can be added by writing one class and registering it in MODELS.

Answer shapes (same as Clef-Omni's systemone):
  noul    {"noul": p}                         probability of yes, 0-1
  choice  {"choice": key, "confidence": p, "probabilities": {key: p}}
  score   {"score": x, ...}                   0 .. len(criteria)-1
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

FPS = 2  # frames per second given to models
SR = 16000  # audio sample rate given to models


@dataclass
class Moment:
    clip: Path
    t0: float
    t1: float
    frames: np.ndarray  # (n, h, w, 3) uint8 at FPS
    audio: np.ndarray | None  # mono float32 at SR, or None


class DecisionModel(Protocol):
    id: str  # stored with every cached answer, so answers from different models never mix

    def ask(self, moment: Moment, questions: dict[str, dict]) -> dict[str, dict]: ...

    def load_clip(self, path: Path) -> tuple[np.ndarray, np.ndarray | None]:
        """All frames at FPS and the full audio track at SR."""
        ...


class ClefOmni:
    """mlx-community/clef-omni-8bit, run locally on Apple silicon (about 37 GB peak memory)."""

    id = "mlx-community/clef-omni-8bit"
    STATE = "A short moment from a film or TV scene, with its sound."

    def __init__(self):
        from huggingface_hub import snapshot_download
        path = snapshot_download(self.id)
        sys.path.insert(0, path)
        import clef_mlx
        self._clef = clef_mlx
        self._model = clef_mlx.load(path)

    def load_clip(self, path: Path):
        return self._clef.load_video(str(path))

    def ask(self, moment: Moment, questions: dict[str, dict]) -> dict[str, dict]:
        req = {"model": "clef-omni", "state": self.STATE, "videos": [moment.frames], "questions": questions}
        if moment.audio is not None and moment.audio.size >= SR // 4:
            req["audio"] = [moment.audio]
        return self._model.systemone(req)["answers"]


MODELS = {"clef-omni": ClefOmni}


def load_model(name: str = "clef-omni") -> DecisionModel:
    return MODELS[name]()
