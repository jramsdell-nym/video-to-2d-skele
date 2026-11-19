"""Package existing synchronized overlays as standalone, shareable MP4s."""
from fractions import Fraction
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

import av
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exports" / "shareable-comparisons"
OUT.mkdir(parents=True, exist_ok=True)
REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT = ImageFont.truetype(REGULAR, 23)
TITLE = ImageFont.truetype(BOLD, 30)
MODEL = ImageFont.truetype(BOLD, 34)
BADGE = ImageFont.truetype(BOLD, 26)
BG = (13, 19, 29)
INK = (239, 244, 252)
MUTED = (166, 181, 202)
CYAN = (55, 225, 255)
CORAL = (255, 125, 110)
MODELS = ["Apple Vision", "YOLO11s-pose", "MediaPipe Full", "MediaPipe Heavy"]
CLIPS = [
    ("elian", "01-elian-left-handed.mp4", "Elian · left-handed swing"),
    ("trey", "02-trey-right-handed.mp4", "Trey · right-handed swing"),
    ("occlusion", "03-trey-background-people.mp4", "Trey · background people"),
    ("seed", "04-november-6-left-handed.mp4", "November 6 · left-handed swing"),
]


def compose_grid(source):
    grid = source.copy()
    d = ImageDraw.Draw(grid)
    for i, name in enumerate(MODELS):
        x, y = (i % 2) * 960, (i // 2) * 540
        # Replace only the old technical header; leave all image/overlay pixels.
        d.rectangle((x, y, x + 959, y + 57), fill=(21, 31, 45))
        d.text((x + 20, y + 8), name, font=MODEL, fill=INK)
    grid = grid.resize((1706, 960), Image.Resampling.LANCZOS)
    return grid


def compose_frame(grid, label, source_index, slow):
    image = Image.new("RGB", (1920, 1080), BG)
    image.paste(grid, (107, 64))
    d = ImageDraw.Draw(image)
    d.text((107, 15), "2D POSE COMPARISON  |  " + label, font=TITLE, fill=INK)
    speed = "QUARTER-SPEED REPLAY · 0.25×" if slow else "ORIGINAL SPEED · 1×"
    d.text((1813, 17), speed, font=BADGE, fill=CYAN, anchor="ra")
    d.line((960, 64, 960, 1024), fill=BG, width=4)
    d.line((107, 544, 1813, 544), fill=BG, width=4)
    d.text((107, 1034), f"Source {source_index / 60:.3f}s  ·  Same frame in every panel", font=FONT, fill=MUTED)
    d.text((935, 1034), "Left", font=FONT, fill=CYAN)
    d.text((989, 1034), "/", font=FONT, fill=MUTED)
    d.text((1010, 1034), "Right", font=FONT, fill=CORAL)
    d.text((1080, 1034), "· anatomical sides", font=FONT, fill=MUTED)
    d.text((1813, 1034), "Model-specific score ≥0.30", font=FONT, fill=MUTED, anchor="ra")
    return image


def full_body_frames(clip, source_path):
    """Render the uncropped source with the existing synchronized predictions."""
    import cv2
    from render_mediapipe import panel

    old = json.loads((ROOT / "gallery-data.js").read_text().removeprefix("window.REVIEW=").removesuffix(";"))
    data = old["clips"][clip]
    for variant in ("full", "heavy"):
        data[variant] = json.loads((ROOT / "outputs" / clip / f"mediapipe-{variant}.json").read_text())["frames"]
    keys = ("selected", "pose", "full", "heavy")
    frames = []
    with av.open(str(source_path)) as source:
        for i, frame in enumerate(source.decode(video=0)):
            src = frame.reformat(width=960, height=540, format="bgr24").to_ndarray()
            grid = Image.new("RGB", (1920, 1080))
            for j, key in enumerate(keys):
                row = data[key][i]
                assert abs(float(frame.pts * frame.time_base) - row["time"]) < 1e-6
                # No crop: head, legs, feet, and surrounding floor stay in view.
                rendered = panel(src, row, MODELS[j], rect=None)
                image = Image.fromarray(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
                grid.paste(image, ((j % 2) * 960, (j // 2) * 540))
            frames.append(compose_grid(grid))
    return frames


def export(clip, filename, label, full_body=False):
    source_path = ROOT / "data" / clip / "source.mp4" if full_body else ROOT / "outputs" / clip / "mediapipe-detail.mp4"
    if full_body:
        filename = filename.replace(".mp4", "-full-body.mp4")
        frames = full_body_frames(clip, source_path)
    else:
        with av.open(str(source_path)) as source:
            frames = [compose_grid(frame.to_image()) for frame in source.decode(video=0)]
    assert len(frames) == 120, (clip, len(frames))
    path = OUT / filename
    with av.open(str(path), "w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=60)
        stream.width, stream.height = 1920, 1080
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 15360)
        stream.codec_context.time_base = stream.time_base
        stream.options = {"crf": "21", "preset": "fast", "bf": "0"}
        n = 0
        for slow, repetitions in [(False, 1), (True, 4)]:
            for i, grid in enumerate(frames):
                image = compose_frame(grid, label, i, slow)
                if not slow and i == 60:
                    image.save(OUT / filename.replace(".mp4", "-preview.jpg"), quality=92)
                for _ in range(repetitions):
                    frame = av.VideoFrame.from_image(image)
                    frame.pts, frame.time_base = n * 256, Fraction(1, 15360)
                    for packet in stream.encode(frame):
                        container.mux(packet)
                    n += 1
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        timestamps = [float(f.pts * f.time_base) for f in container.decode(video=0)]
        assert len(timestamps) == 600
        assert all(abs(t - i / 60) < 1e-7 for i, t in enumerate(timestamps))
        assert stream.width == 1920 and stream.height == 1080
        assert stream.codec_context.name == "h264"
        assert stream.codec_context.format.name == "yuv420p"
        assert abs(float(stream.duration * stream.time_base) - 10) < 1e-7
    result = {"clip": clip, "file": filename, "bytes": path.stat().st_size,
              "frames": 600, "seconds": 10, "fps": 60, "resolution": [1920, 1080],
              "codec": "h264", "pixelFormat": "yuv420p", "sourceFrames": 120,
              "source": str(source_path.relative_to(ROOT)),
              "sourceSHA256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
              "view": "Uncropped original frame; full body and feet" if full_body else "Shared upper-body crop",
              "models": ["Apple Vision with selected-hitter tracking", *MODELS[1:]],
              "sequence": "Original-speed 2s, then quarter-speed replay 8s. No interpolation.",
              "validation": "Full decode and every frame timestamp passed."}
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-body", action="store_true", help="Show the original frame with legs and feet visible")
    args = parser.parse_args()
    if args.full_body:
        OUT = ROOT / "exports" / "full-body-comparisons"
        OUT.mkdir(parents=True, exist_ok=True)
    results = [export(*clip, full_body=args.full_body) for clip in CLIPS]
    (OUT / "export-manifest.json").write_text(json.dumps(results, indent=2) + "\n")
    zip_name = "four-model-full-body-comparisons.zip" if args.full_body else "four-model-comparisons.zip"
    with zipfile.ZipFile(OUT / zip_name, "w", compression=zipfile.ZIP_STORED) as archive:
        for result in results:
            archive.write(OUT / result["file"], arcname=result["file"])
    print("Finished all four shareable videos and ZIP.", flush=True)
