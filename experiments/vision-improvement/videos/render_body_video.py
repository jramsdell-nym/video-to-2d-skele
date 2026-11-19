"""Export frozen baseline/candidate predictions, with every held-out frame."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/body-experiment/session-holdout-20261111'
RUN = ROOT / 'training/body-conservative-20261111'
OUT = Path(__file__).resolve().parent
EDGES = ((0,1),(0,2),(2,4),(1,3),(3,5),(0,6),(1,7),(6,7),(6,8),(8,10),(7,9),(9,11))
FONT = '/System/Library/Fonts/Helvetica.ttc'
TITLE = ImageFont.truetype(FONT, 32)
SMALL = ImageFont.truetype(FONT, 22)
COLOR = '#ffac55'
SIZE = (1920, 680)
RATE = 15


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return row['videoSHA256'], row['frameIndex']


def load(path):
    records = json.loads(path.read_text())
    result = {key(row): row for row in records}
    assert len(records) == len(result), 'Duplicate frame identity'
    return result


def panel(source, prediction):
    image = source.resize((960, 540), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    points = [None] * 12
    if prediction.get('detected'):
        assert len(prediction['prediction']) == len(prediction['keypointConfidences']) == 12
        for i, (p, conf) in enumerate(zip(prediction['prediction'], prediction['keypointConfidences'])):
            if (len(p) == 2 and all(math.isfinite(v) for v in [*p, conf]) and conf >= .3
                    and 0 <= p[0] <= source.width and 0 <= p[1] <= source.height):
                points[i] = (p[0] * 960 / source.width, p[1] * 540 / source.height)
    for a, b in EDGES:
        if points[a] is not None and points[b] is not None:
            draw.line([points[a], points[b]], fill='#141b22', width=7)
            draw.line([points[a], points[b]], fill=COLOR, width=3)
    for p in points:
        if p is not None:
            x, y = p
            draw.ellipse((x-4, y-4, x+4, y+4), fill=COLOR, outline='#141b22', width=1)
    return image


def main():
    inputs = {'referenceFrames': RUN/'val-rows.json',
              'baseline': BASE/'generic-val-predictions.json',
              'candidate': RUN/'candidate-val-predictions.json'}
    rows, baseline, candidate = [load(path) for path in inputs.values()]
    assert rows.keys() == baseline.keys() == candidate.keys()
    assert len(rows) == 240
    videos = sorted({identity[0] for identity in rows})
    assert len(videos) == 2
    for video in videos:
        frames = sorted((r for r in rows.values() if r['videoSHA256'] == video), key=lambda r: r['frameIndex'])
        assert [r['frameIndex'] for r in frames] == list(range(120))
        assert all(abs(r['frame']['time'] - i/60) < 1e-6 for i, r in enumerate(frames))
    target = OUT/'body-baseline-vs-candidate-quarter-speed.mp4'
    assert not target.exists(), 'Refusing to overwrite export'
    provenance = []
    with av.open(str(target), 'w', options={'movflags': '+faststart'}) as container:
        stream = container.add_stream('libx264', rate=RATE)
        stream.width, stream.height = SIZE
        stream.pix_fmt = 'yuv420p'
        stream.time_base = Fraction(1, RATE)
        stream.codec_context.time_base = stream.time_base
        stream.options = {'crf': '20', 'preset': 'fast', 'bf': '0'}
        for n, identity in enumerate(sorted(rows)):
            row = rows[identity]
            image_path = RUN/'dataset'/row['image']
            with Image.open(image_path) as opened:
                source = opened.convert('RGB')
            assert source.size == (row['width'], row['height'])
            image = Image.new('RGB', SIZE, '#10171e')
            image.paste(panel(source, baseline[identity]), (0, 88))
            image.paste(panel(source, candidate[identity]), (960, 88))
            draw = ImageDraw.Draw(image)
            draw.text((24, 10), 'BASELINE  |  Generic YOLO11n', font=TITLE, fill='white')
            draw.text((984, 10), 'CANDIDATE  |  Fine-tuned YOLO11n', font=TITLE, fill='white')
            caption = f'Swing {videos.index(identity[0])+1}/2  |  Frame {identity[1]+1}/120  |  Source {row["frame"]["time"]:.3f}s  |  Quarter speed'
            draw.text((24, 53), caption, font=SMALL, fill='#bdd0df')
            draw.text((984, 53), 'Experimental model  |  20 epochs  |  Same source frame', font=SMALL, fill='#bdd0df')
            draw.line((960, 88, 960, 628), fill='#10171e', width=3)
            draw.text((24, 642), 'Orange: predicted body joints  |  No smoothing  |  Better overall score; some right-arm errors remain', font=SMALL, fill='#bdd0df')
            frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = n, Fraction(1, RATE)
            for packet in stream.encode(frame):
                container.mux(packet)
            provenance.append({'videoSHA256': identity[0], 'frameIndex': identity[1], 'imageSHA256': sha(image_path)})
        for packet in stream.encode():
            container.mux(packet)
    timestamps = []
    with av.open(str(target)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == SIZE
        assert stream.codec_context.name == 'h264'
        assert stream.codec_context.format.name == 'yuv420p'
        for i, frame in enumerate(container.decode(video=0)):
            timestamps.append(float(frame.pts * frame.time_base))
            if i in (60, 180):
                frame.to_image().save(OUT/f'body-comparison-encoded-{i}.jpg', quality=92)
        duration = float(stream.duration * stream.time_base)
    assert len(timestamps) == 240
    assert all(abs(t-i/RATE) < 1e-7 for i,t in enumerate(timestamps))
    assert abs(duration-16) < 1e-7
    receipt = {'output': str(target), 'outputSHA256': sha(target), 'sourceCodeSHA256': sha(Path(__file__)),
               'inputs': {name: {'path': str(path), 'sha256': sha(path)} for name,path in inputs.items()},
               'frames': provenance, 'durationSeconds': duration, 'fps': RATE, 'size': SIZE,
               'validation': 'Full 240-frame decode and all timestamps passed',
               'gate': 'Detected; finite, in-image keypoints with confidence >= 0.30',
               'selection': 'All validation frames, ordered by video hash and source frame index',
               'inferencePerformed': False, 'smoothingOrInterpolationPerformed': False}
    (OUT/'body-video-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: receipt[k] for k in ['output','durationSeconds','validation']}))


if __name__ == '__main__':
    main()
