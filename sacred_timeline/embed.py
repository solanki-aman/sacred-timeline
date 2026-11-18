"""Fast recall: one EmbeddingGemma 2 vector per moment (its frames and its sound together).

Text, video and audio share one 768-d space, so a typed description is compared directly against
every moment with a single matrix product. This is what makes thousands of hours searchable;
the decision model (Clef-Omni) then only judges the few dozen moments this shortlists.
"""
from __future__ import annotations

import os
from pathlib import Path

import av
import numpy as np

EMBED_ID = "google/embeddinggemma-2"
DECODE_FPS = 1  # frames decoded per second (cuts are found at this rate)
EMBED_STEP = 2  # every 2nd decoded frame goes to the embedding model, so 0.5 fps
TOKENS_PER_FRAME = 70  # model default is 140; 70 is twice as fast and agrees at cosine 0.97
SHORT_SIDE = 256  # frames are scaled so their short side is this many pixels
SR = 16000


def decode(path: Path, fps: float = DECODE_FPS, short: int = SHORT_SIDE) -> tuple[np.ndarray, np.ndarray | None]:
    """Frames at `fps` (n, h, w, 3) uint8 and mono 16 kHz audio, decoded in one streaming pass each."""
    c = av.open(str(path))
    vs = c.streams.video[0]
    vs.thread_type = "AUTO"
    w, h = vs.codec_context.width, vs.codec_context.height
    scale = short / min(w, h)
    ow, oh = int(round(w * scale / 2) * 2), int(round(h * scale / 2) * 2)
    frames, nxt = [], 0.0
    for f in c.decode(vs):
        if f.time is not None and f.time + 1e-6 >= nxt:
            frames.append(f.reformat(width=ow, height=oh, format="rgb24").to_ndarray())
            nxt += 1 / fps
    c.close()
    audio = None
    c = av.open(str(path))
    if c.streams.audio:
        rs = av.AudioResampler("fltp", "mono", SR)
        chunks = [g.to_ndarray()[0] for f in c.decode(c.streams.audio[0]) for g in rs.resample(f)]
        audio = np.concatenate(chunks).astype(np.float32) if chunks else None
    c.close()
    return np.stack(frames), audio


class Embedder:
    def __init__(self, device: str | None = None):
        import torch
        from sentence_transformers import SentenceTransformer
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        # bfloat16 on Apple GPUs; never float16 (the model card warns it silently degrades).
        dtype = torch.bfloat16 if device == "mps" else torch.float32
        self.model = SentenceTransformer(EMBED_ID, device=device, model_kwargs={"torch_dtype": dtype})
        self.model[0].processor.video_processor.max_soft_tokens = TOKENS_PER_FRAME
        self.id = EMBED_ID

    def moments(self, items: list[tuple[np.ndarray, np.ndarray | None]], batch: int = 8) -> np.ndarray:
        """items: (frames already thinned to 0.5 fps, audio at 16 kHz or None) -> (n, 768) unit vectors."""
        inputs = []
        for frames, audio in items:
            d = {"video": frames}
            if audio is not None and audio.size >= SR // 2:
                d["audio"] = audio
            inputs.append(d)
        return np.asarray(self.model.encode(inputs, batch_size=batch, normalize_embeddings=True), dtype=np.float32)

    def queries(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, prompt_name="SearchQuery", normalize_embeddings=True),
                          dtype=np.float32)
