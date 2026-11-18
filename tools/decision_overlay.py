"""Draw Clef-Omni's read of each moment's emotion onto a video.

The clip is cut into short segments. Each segment (frames plus sound) goes to Clef-Omni with a
set of questions about emotion, and the answers are drawn as a small stack of callouts:

  emotion    what the main character feels (choice), linked by a line to their face
  intensity  how strongly it shows (score, 0 to 2)
  voice      the tone of the voice (choice)
  tension    whether there is tension between characters (yes/no probability)

Faces are found with Apple's Vision framework, on device. The stack is placed in whichever
corner overlaps faces least, and the line ends beside the face rather than on it. Clef-Omni
itself returns no coordinates; it only picks which face is speaking, via a 3 x 3 grid question.

Writes, outside the repo by default:
  <out>/<clip>.annotated.mp4   the clip with callouts and its original sound
  <out>/<clip>.decisions.json  every answer and face box, per segment
  <out>/<clip>.still-N.jpg     a few annotated frames

Usage:
  ~/.venvs/clef-omni/bin/python tools/decision_overlay.py CLIP [CLIP ...] [--out DIR] [--segment 2.0]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

MODEL_ID = "mlx-community/clef-omni-8bit"
SR = 16000  # Clef-Omni audio sample rate
FPS = 2  # Clef-Omni video sampling rate

# --------------------------------------------------------------------------- questions

GRID = {
    "top_left": (1 / 6, 1 / 6), "top": (1 / 2, 1 / 6), "top_right": (5 / 6, 1 / 6),
    "left": (1 / 6, 1 / 2), "center": (1 / 2, 1 / 2), "right": (5 / 6, 1 / 2),
    "bottom_left": (1 / 6, 5 / 6), "bottom": (1 / 2, 5 / 6), "bottom_right": (5 / 6, 5 / 6),
}

EMOTIONS = {
    "happy": "Happy or delighted",
    "amused": "Amused, finding something funny",
    "sad": "Sad or hurt",
    "angry": "Angry or irritated",
    "anxious": "Afraid, anxious or worried",
    "surprised": "Surprised or shocked",
    "awkward": "Awkward or embarrassed",
    "confident": "Confident or self-assured",
}
EMOTION_LABEL = {"happy": "Happy", "amused": "Amused", "sad": "Sad", "angry": "Angry", "anxious": "Anxious",
                 "surprised": "Surprised", "awkward": "Awkward", "confident": "Confident"}
VOICES = {
    "calm": "Calm and even",
    "serious": "Serious or matter-of-fact",
    "warm": "Warm or friendly",
    "nervous": "Nervous or hesitant",
    "sarcastic": "Sarcastic or mocking",
    "deadpan": "Deadpan or flat",
    "excited": "Excited or eager",
    "pleading": "Pleading or desperate",
    "angry": "Angry or raised",
}
VOICE_LABEL = {k: v.split(" or ")[0].split(" and ")[0] for k, v in VOICES.items()}
INTENSITY = ["Subtle", "Clear", "Strong"]

QUESTIONS = {
    "footage": {"type": "noul", "instructions":
                "Is this live-action footage from the film or show, rather than a promotional end card, "
                "subscribe screen or channel graphic?"},
    "person_visible": {"type": "noul", "instructions": "Is a person's face visible in the frame?"},
    "speaking": {"type": "noul", "instructions": "Is someone speaking in this moment?"},
    "focus_at": {"type": "choice",
                 "instructions": "Where in the frame is the face of the character this moment is about: "
                                 "the one speaking, or else the one the camera focuses on?",
                 "criteria": {k: f"The {k.replace('_', ' ')} of the frame" for k in GRID}},
    "emotion": {"type": "choice", "instructions":
                "Which emotion does that character show most, from their face, body and voice? Pick the closest even if it is subtle.", "criteria": EMOTIONS},
    "intensity": {"type": "score", "instructions": "How strongly does that character show the emotion?",
                  "criteria": INTENSITY},
    "voice": {"type": "choice", "instructions": "What is the tone of the voice that is speaking?",
              "criteria": VOICES},
    "tension": {"type": "noul", "instructions":
                "Is there tension, conflict or discomfort between the characters in this moment?"},
}

# --------------------------------------------------------------------------- faces (Apple Vision)


class FaceFinder:
    def __init__(self):
        import Vision
        from Foundation import NSData
        self.Vision, self.NSData = Vision, NSData

    def __call__(self, img: Image.Image) -> list[tuple[float, float, float, float]]:
        """Face boxes as (x0, y0, x1, y1), normalised 0-1 with the origin at top left."""
        im = img.convert("RGB")
        im.thumbnail((640, 640))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        data = self.NSData.dataWithBytes_length_(buf.getvalue(), len(buf.getvalue()))
        req = self.Vision.VNDetectFaceRectanglesRequest.alloc().init()
        handler = self.Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
        handler.performRequests_error_([req], None)
        out = []
        for ob in req.results() or []:
            b = ob.boundingBox()
            x, y, w, h = b.origin.x, b.origin.y, b.size.width, b.size.height
            out.append((x, 1 - y - h, x + w, 1 - y))
        return out


# --------------------------------------------------------------------------- drawing

FONT = "/System/Library/Fonts/SFNS.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"
DOT = {"noul": (52, 199, 120), "choice": (64, 120, 242), "score": (244, 162, 54)}
INK, MUTED = (29, 29, 27), (112, 111, 106)


def font(size: float, weight: str = "Regular", mono: bool = False) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(MONO if mono else FONT, max(8, round(size)))
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f


def pill(kind: str, key: str, label: str, value: str, unit: float, width: float, suffix: str = "",
         score: float | None = None, note: str = "") -> Image.Image:
    """One compact callout, drawn at 2x and scaled down for smooth edges."""
    s = 2 * unit
    f_head, f_label = font(15 * s, mono=True), font(23 * s, "Medium")
    f_value, f_suffix = font(36 * s), font(17 * s)
    w, h = int(width * s), int(76 * s)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle((0, 0, w - 1, h - 1), radius=h / 2, fill=(250, 249, 246, 228))
    d = ImageDraw.Draw(img)
    cy, x, r = h / 2, 30 * s, 6 * s
    d.ellipse((x - r, cy - r, x + r, cy + r), fill=DOT[kind] + (255,))
    tx = x + 18 * s
    d.text((tx, cy - 27 * s), f"{kind} · {key}", font=f_head, fill=MUTED)
    d.text((tx, cy - 6 * s), label, font=f_label, fill=INK)
    if note:
        d.text((tx + f_label.getlength(label) + 10 * s, cy - 6 * s + 5 * s), note, font=font(16 * s), fill=MUTED)
    value_w = f_value.getlength(value) + (f_suffix.getlength(suffix) + 3 * s if suffix else 0)
    vx = w - 32 * s - value_w
    d.text((vx, cy), value, font=f_value, fill=INK, anchor="lm")
    if suffix:
        d.text((vx + f_value.getlength(value) + 3 * s, cy + 9 * s), suffix, font=f_suffix, fill=MUTED, anchor="lm")
    if score is not None:
        by, bw = cy + 23 * s, 120 * s
        lx = tx + f_label.getlength(label) + 16 * s
        by, bx = cy + 2 * s, min(lx, vx - bw - 18 * s)
        d.rounded_rectangle((bx, by - 2 * s, bx + bw, by + 2 * s), radius=2 * s, fill=(218, 216, 209, 255))
        d.rounded_rectangle((bx, by - 2 * s, bx + max(4 * s, bw * score), by + 2 * s), radius=2 * s,
                            fill=DOT["score"] + (255,))
        kx, kr = bx + bw * score, 6 * s
        d.ellipse((kx - kr, by - kr, kx + kr, by + kr), fill=(255, 255, 255, 255), outline=(196, 194, 186, 255),
                  width=int(1.5 * s))
    return img.resize((round(w / 2), round(h / 2)), Image.LANCZOS)


def dot_sprite(unit: float) -> Image.Image:
    s = 4
    r = 13 * unit * s
    img = Image.new("RGBA", (int(2 * r), int(2 * r)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((r * 0.15, r * 0.15, r * 1.85, r * 1.85), fill=(255, 255, 255, 70))
    d.ellipse((r * 0.5, r * 0.5, r * 1.5, r * 1.5), fill=(255, 255, 255, 255))
    return img.resize((int(img.width / s), int(img.height / s)), Image.LANCZOS)


def fmt_tc(t: float) -> str:
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def overlap(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


class Stack:
    """The callout stack for one segment: built once, positioned away from faces."""

    def __init__(self, seg: dict, W: int, H: int, unit: float):
        self.W, self.H, self.u = W, H, unit
        a = seg["answers"]
        self.footage = a["footage"]["noul"] >= 0.5
        self.person = a["person_visible"]["noul"] >= 0.5
        self.speaking = a["speaking"]["noul"] >= 0.5
        self.focus = grid_point(a["focus_at"])
        width = 440
        pills = []
        if not self.footage:
            pills.append(pill("noul", "is_footage", "Promo or end card", f"{a['footage']['noul']:.2f}", unit, width))
        else:
            if self.person:
                e = a["emotion"]
                ranked = sorted(e["probabilities"].items(), key=lambda kv: -kv[1])
                second = f"then {EMOTION_LABEL[ranked[1][0]].lower()} {ranked[1][1] * 100:.0f}%"
                pills.append(pill("choice", "emotion", EMOTION_LABEL[e["choice"]], f"{e['confidence'] * 100:.0f}%",
                                  unit, width, note=second))
                sc = a["intensity"]["score"]
                pills.append(pill("score", "intensity", INTENSITY[min(2, max(0, round(sc)))], f"{sc:.1f}", unit,
                                  width, suffix="/2", score=sc / 2))
            if self.speaking:
                v = a["voice"]
                pills.append(pill("choice", "voice", VOICE_LABEL[v["choice"]], f"{v['confidence'] * 100:.0f}%",
                                  unit, width))
            t = a["tension"]["noul"]
            pills.append(pill("noul", "tension", "Tension" if t >= 0.5 else "At ease", f"{t:.2f}", unit, width))
        gap = int(10 * unit)
        head_f = font(17 * unit, mono=True)
        head = f"Clef-Omni · {fmt_tc(seg['t0'])} · {seg['seconds']:.2f} s"
        head_h = int(30 * unit)
        w = max(p.width for p in pills)
        h = head_h + sum(p.height for p in pills) + gap * (len(pills) - 1)
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.text((int(14 * unit) + 1, 1), head, font=head_f, fill=(0, 0, 0, 120))
        d.text((int(14 * unit), 0), head, font=head_f, fill=(255, 255, 255, 235))
        y = head_h
        for p in pills:
            img.alpha_composite(p, (0, y))
            y += p.height + gap
        self.img = img
        self.first_pill_y = head_h + pills[0].height / 2 if pills else 0
        self.has_emotion = self.footage and self.person
        self.pos = None

    def place(self, faces: list[tuple], prev: str | None) -> None:
        W, H, m = self.W, self.H, int(0.04 * min(self.W, self.H))
        w, h = self.img.size
        spots = {"top_left": (m, m), "top_right": (W - m - w, m),
                 "bottom_left": (m, H - m - h), "bottom_right": (W - m - w, H - m - h)}
        if H > W:  # portrait: also the middle band, left and right
            spots.update({"mid_left": (m, int(H * 0.5 - h / 2)), "mid_right": (W - m - w, int(H * 0.5 - h / 2))})
        pad = 0.04
        boxes = [((f[0] - pad) * W, (f[1] - pad) * H, (f[2] + pad) * W, (f[3] + pad) * H) for f in faces]

        def cost(name):
            x, y = spots[name]
            r = (x, y, x + w, y + h)
            c = sum(overlap(r, b) for b in boxes)
            return (round(c / (w * h), 2), 0 if name == prev else 1, list(spots).index(name))
        self.slot = min(spots, key=cost)
        self.pos = spots[self.slot]

    def side(self) -> str:
        return "left" if self.pos[0] < self.W / 2 else "right"


def grid_point(answer: dict) -> tuple[float, float]:
    probs = {k: v ** 2 for k, v in answer["probabilities"].items()}
    total = sum(probs.values()) or 1.0
    return (sum(GRID[k][0] * p for k, p in probs.items()) / total,
            sum(GRID[k][1] * p for k, p in probs.items()) / total)


def pick_face(faces, target):
    """The detected face nearest the grid cell Clef-Omni chose."""
    if not faces:
        return None
    return min(faces, key=lambda f: ((f[0] + f[2]) / 2 - target[0]) ** 2 + ((f[1] + f[3]) / 2 - target[1]) ** 2)


# --------------------------------------------------------------------------- pipeline

def analyse(model, clef_mlx, path: Path, seg: float, finder: FaceFinder):
    frames, audio = clef_mlx.load_video(str(path))
    per = max(2, round(seg * FPS))
    per += per % 2
    out = []
    for i in range(0, len(frames), per):
        fr = frames[i:i + per]
        if len(fr) % 2:
            fr = np.concatenate([fr, fr[-1:]])
        t0, t1 = i / FPS, (i + per) / FPS
        req = {"model": "clef-omni", "state": "A short moment from a film or TV scene, with its sound.",
               "videos": [fr], "questions": QUESTIONS}
        if audio is not None:
            a = audio[int(t0 * SR):int(t1 * SR)]
            if a.size >= SR // 4:
                req["audio"] = [a]
        start = time.time()
        ans = model.systemone(req)["answers"]
        secs = round(time.time() - start, 3)
        faces = [finder(Image.fromarray(f)) for f in frames[i:i + per]]
        out.append({"t0": t0, "t1": t1, "seconds": secs, "answers": ans, "faces": faces})
        print(f"  {fmt_tc(t0)} emotion={ans['emotion']['choice']}({ans['emotion']['confidence']:.2f})"
              f" intensity={ans['intensity']['score']:.1f} voice={ans['voice']['choice']}"
              f" tension={ans['tension']['noul']:.2f} faces={len(faces[0])}", flush=True)
    return out


def render(path: Path, segments: list[dict], dest: Path, finder: FaceFinder, stills: int = 4) -> list[Path]:
    src = av.open(str(path))
    vin = src.streams.video[0]
    ain = src.streams.audio[0] if src.streams.audio else None
    vin.thread_type = "AUTO"
    out = av.open(str(dest), "w")
    vout = out.add_stream("libx264", rate=vin.average_rate)
    W, H = vin.width, vin.height
    vout.width, vout.height, vout.pix_fmt = W, H, "yuv420p"
    vout.options = {"crf": "19", "preset": "medium"}
    aout = out.add_stream_from_template(ain) if ain else None
    unit = (W / 1080) if H > W else H / 1080
    dot = dot_sprite(unit)

    stacks: dict[int, Stack] = {}
    prev_slot = None
    dur = float(src.duration / 1e6) if src.duration else segments[-1]["t1"]
    want = [dur * (k + 0.5) / stills for k in range(stills)]
    saved, n, faces, anchor = [], 0, [], None
    for packet in src.demux(*[s for s in (vin, ain) if s is not None]):
        if packet.stream is ain:
            if packet.dts is not None:
                packet.stream = aout
                out.mux(packet)
            continue
        for frame in packet.decode():
            t = frame.time if frame.time is not None else n / float(vin.average_rate)
            k = max(0, min(len(segments) - 1, int(t // (segments[0]["t1"] - segments[0]["t0"]))))
            if k not in stacks:
                stacks.clear()
                sg = segments[k]
                st = Stack(sg, W, H, unit)
                st.place([f for fs in sg["faces"] for f in fs], prev_slot)
                prev_slot = st.slot
                stacks[k] = st
            st = stacks[k]
            img = frame.to_image()
            if n % 3 == 0:
                faces = finder(img)
            rgba = img.convert("RGBA")
            if st.has_emotion and faces:
                f = pick_face(faces, st.focus)
                # End beside the face, on the side facing the stack, at eye height.
                if st.side() == "left":
                    fx = f[0] * W - 14 * unit
                else:
                    fx = f[2] * W + 14 * unit
                fy = (f[1] + 0.42 * (f[3] - f[1])) * H
                target = (fx, fy)
                anchor = target if anchor is None else (0.6 * anchor[0] + 0.4 * fx, 0.6 * anchor[1] + 0.4 * fy)
                sx = st.pos[0] + (st.img.width if st.side() == "left" else 0)
                sy = st.pos[1] + st.first_pill_y
                lines = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
                ImageDraw.Draw(lines).line([(sx, sy), anchor], fill=(255, 255, 255, 200), width=max(2, round(2 * unit)))
                lines.alpha_composite(dot, (int(anchor[0] - dot.width / 2), int(anchor[1] - dot.height / 2)))
                rgba.alpha_composite(lines)
            else:
                anchor = None
            rgba.alpha_composite(st.img, st.pos)
            rgb = rgba.convert("RGB")
            if want and t >= want[0]:
                p = dest.with_name(dest.name.replace(".annotated.mp4", f".still-{len(saved) + 1}.jpg"))
                rgb.save(p, quality=90)
                saved.append(p)
                want.pop(0)
            nf = av.VideoFrame.from_image(rgb)
            nf.pts = n
            n += 1
            for pk in vout.encode(nf):
                out.mux(pk)
    for pk in vout.encode():
        out.mux(pk)
    out.close()
    src.close()
    return saved


def slug(clip: Path) -> str:
    s = re.sub(r"\(.*?\)|#\w+|^(YTDown\.com_YouTube_|ssvid\.net-+)", "", clip.stem)
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:48].strip("-") or "clip"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path.home() / "Movies" / "sacred-timeline" / "annotated")
    ap.add_argument("--segment", type=float, default=2.0, help="seconds per decision (default 2.0)")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL_ID)
    sys.path.insert(0, path)
    import clef_mlx

    model = clef_mlx.load(path)
    finder = FaceFinder()
    args.out.mkdir(parents=True, exist_ok=True)
    for clip in args.clips:
        stem = slug(clip)
        print(f"{clip.name}\n analysing in {args.segment:g} s segments")
        t = time.time()
        segs = analyse(model, clef_mlx, clip, args.segment, finder)
        analyse_s = time.time() - t
        (args.out / f"{stem}.decisions.json").write_text(json.dumps(
            {"clip": str(clip), "model": MODEL_ID, "segment_seconds": args.segment, "questions": QUESTIONS,
             "analyse_seconds": round(analyse_s, 1), "segments": segs}, indent=1))
        print(f" analysed {len(segs)} segments in {analyse_s:.1f} s; rendering")
        t = time.time()
        stills = render(clip, segs, args.out / f"{stem}.annotated.mp4", finder)
        print(f" rendered in {time.time() - t:.1f} s -> {args.out / (stem + '.annotated.mp4')}")
        for p in stills:
            print(f"  still {p}")


if __name__ == "__main__":
    main()
