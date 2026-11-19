"""Render immutable, aligned 2D predictions; never run inference or fill gaps.

The input JSON contract is illustrated by input-contract.json. All paths are
resolved relative to the input JSON. Predictions use original-image pixels.
Every frame of each selected source video must have one prediction record for
each model, including records whose detected field is false.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re

import av
from PIL import Image, ImageDraw, ImageFont

SIZE = (1920, 1280)
PANEL = (960, 540)
ROW_LABEL_Y = (68, 654)
ROW_IMAGE_Y = (114, 700)
SPEED_FACTOR = 4
EDGES = ((0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6),
         (1, 7), (6, 7), (6, 8), (8, 10), (7, 9), (9, 11))
MODEL_KEYS = ('bodyBaseline', 'bodyCandidate', 'batBaseline', 'batCandidate')
BODY_ORDER = ('left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
              'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
              'left_knee', 'right_knee', 'left_ankle', 'right_ankle')
BAT_ORDER = ('tip', 'knob')
BG, TEXT, MUTED = '#10171e', '#f4f7fb', '#bdcedb'
BODY, TIP, KNOB = '#ffac55', '#ffce69', '#74cefa'
FONT_PATH = '/System/Library/Fonts/Helvetica.ttc'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def resolve(root, path):
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def verified_file(root, specification):
    path = resolve(root, specification['path'])
    observed = sha256(path)
    if observed != specification['sha256']:
        raise ValueError(f'Hash mismatch: {path}')
    return path


def load_predictions(path):
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError('Predictions must be a JSON array')
    indexed = {}
    for row in rows:
        key = (row['videoSHA256'], row['frameIndex'])
        if key in indexed:
            raise ValueError(f'Duplicate prediction identity: {key}')
        indexed[key] = row
    return indexed


def visible_points(row, count, size, threshold, confidence_field):
    if not row.get('detected'):
        return [None] * count
    coordinates = row['prediction']
    confidences = row[confidence_field]
    if len(coordinates) != count or len(confidences) != count:
        raise ValueError(f'Expected {count} ordered keypoints')
    points = []
    for point, confidence in zip(coordinates, confidences):
        valid = (len(point) == 2 and all(math.isfinite(v) for v in (*point, confidence))
                 and confidence >= threshold
                 and 0 <= point[0] < size[0] and 0 <= point[1] < size[1])
        points.append(tuple(point) if valid else None)
    return points


def fit_source(source):
    scale = min(PANEL[0] / source.width, PANEL[1] / source.height)
    target = (round(source.width * scale), round(source.height * scale))
    offset = ((PANEL[0] - target[0]) // 2, (PANEL[1] - target[1]) // 2)
    image = Image.new('RGB', PANEL, '#050a10')
    image.paste(source.resize(target, Image.Resampling.LANCZOS), offset)
    # Independent axis scales respect the rounded resize dimensions exactly.
    return image, (target[0] / source.width, target[1] / source.height), offset


def annotated_panel(base, points, scale, offset, kind):
    image = base.copy()
    draw = ImageDraw.Draw(image)
    points = [(p[0] * scale[0] + offset[0], p[1] * scale[1] + offset[1])
              if p is not None else None for p in points]
    for a, b in EDGES if kind == 'body' else ((0, 1),):
        if points[a] is not None and points[b] is not None:
            draw.line((points[a], points[b]), fill=BG, width=7)
            draw.line((points[a], points[b]), fill=BODY if kind == 'body' else TEXT, width=3)
    for index, point in enumerate(points):
        if point is None:
            continue
        x, y = point
        color = BODY if kind == 'body' else (TIP, KNOB)[index]
        radius = 4 if kind == 'body' else 7
        draw.ellipse((x-radius-1, y-radius-1, x+radius+1, y+radius+1), fill=BG)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                     fill=color if kind == 'body' else None, outline=color, width=2)
        if kind == 'bat':
            draw.line((x-11, y, x+11, y), fill=color, width=2)
            draw.line((x, y-11, x, y+11), fill=color, width=2)
    return image


def render_frame(source, rows, models, clip, number, total, frame_index, frame_count, source_time):
    image = Image.new('RGB', SIZE, BG)
    draw = ImageDraw.Draw(image)
    large = ImageFont.truetype(FONT_PATH, 28)
    medium = ImageFont.truetype(FONT_PATH, 25)
    small = ImageFont.truetype(FONT_PATH, 21)
    draw.text((22, 5), f'2D detector comparison  |  {clip["title"]}  |  {number}/{total}', font=large, fill=TEXT)
    draw.text((22, 39), f'Complete clip  |  Frame {frame_index+1}/{frame_count}  |  Source {source_time:.3f}s  |  Quarter speed', font=small, fill=MUTED)
    base, scale, offset = fit_source(source)
    coverage = {}
    for index, name in enumerate(MODEL_KEYS):
        kind = 'body' if index < 2 else 'bat'
        count = 12 if kind == 'body' else 2
        model = models[name]
        confidence_field = 'keypointConfidences' if kind == 'body' else 'endpointConfidences'
        prediction = rows[name]
        if prediction.get('score', 0) < model['detectionThreshold']:
            prediction = {'detected': False}
        points = visible_points(prediction, count, source.size, model['keypointThreshold'], confidence_field)
        coverage[name] = sum(point is not None for point in points)
        x, row = (index % 2) * PANEL[0], index // 2
        image.paste(annotated_panel(base, points, scale, offset, kind), (x, ROW_IMAGE_Y[row]))
        label = f'{kind.upper()}  |  {model["label"]}'
        draw.text((x+22, ROW_LABEL_Y[row]+7), label, font=medium, fill=TEXT)
        status = f'{coverage[name]}/{count} points'
        if coverage[name] == 0:
            status = 'NO USABLE POINTS'
        draw.text((x+PANEL[0]-22, ROW_LABEL_Y[row]+10), status, font=small,
                  fill=MUTED if coverage[name] else '#ffb28c', anchor='ra')
    draw.line((960, 68, 960, 1240), fill=BG, width=4)
    draw.text((22, 1248), 'Orange: body  |  Yellow: bat tip  |  Blue: knob  |  No smoothing or gap filling  |  2D only; no new 3D model', font=small, fill=MUTED)
    return image, coverage


def encode_frame(container, stream, image, pts, time_base):
    frame = av.VideoFrame.from_image(image)
    frame.pts, frame.time_base = pts, time_base
    for packet in stream.encode(frame):
        container.mux(packet)


def validate_video(path, expected_times, poster_indices, output):
    times, posters = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if (stream.width, stream.height) != SIZE:
            raise ValueError('Encoded dimensions differ from requested dimensions')
        if stream.codec_context.name != 'h264' or stream.codec_context.format.name != 'yuv420p':
            raise ValueError('Expected broadly playable H.264 / yuv420p')
        for index, frame in enumerate(container.decode(video=0)):
            times.append(Fraction(frame.pts) * frame.time_base)
            if index in poster_indices:
                target = output / f'{path.stem}-frame-{index:04d}.jpg'
                frame.to_image().save(target, quality=94)
                posters.append({'frameIndex': index, 'path': str(target), 'sha256': sha256(target)})
        duration = float(stream.duration * stream.time_base)
    if len(times) != len(expected_times):
        raise ValueError('Encoded frame count changed')
    for index, (actual, expected) in enumerate(zip(times, expected_times)):
        if abs(actual - expected) > Fraction(1, 100000):
            raise ValueError(f'Encoded timestamp differs at frame {index}')
    return {'allFramesDecoded': True, 'allTimestampsVerified': True, 'frameCount': len(times),
            'durationSeconds': duration, 'size': list(SIZE), 'codec': 'h264', 'pixelFormat': 'yuv420p',
            'firstTimestamp': float(times[0]), 'lastTimestamp': float(times[-1]), 'posters': posters}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    input_path = args.input.resolve()
    root = input_path.parent
    config = json.loads(input_path.read_text())
    output = args.output.resolve() if args.output else root / 'exports'
    output.mkdir(parents=True, exist_ok=True)
    if config['coordinateSystem'] != 'original-image-pixels-top-left':
        raise ValueError('Unsupported coordinate system')
    models, predictions, model_audit = config['models'], {}, {}
    for name in MODEL_KEYS:
        model = models[name]
        order = BODY_ORDER if name.startswith('body') else BAT_ORDER
        if tuple(model['keypointOrder']) != order:
            raise ValueError(f'Incorrect keypoint order: {name}')
        weights = verified_file(root, model['weights'])
        path = verified_file(root, model['predictions'])
        predictions[name] = load_predictions(path)
        if len(model['label']) > 29:
            raise ValueError('Model label too long for legible display')
        model_audit[name] = {**model, 'weights': str(weights), 'predictions': str(path)}
        if 'predictionManifest' in model:
            manifest_path = verified_file(root, model['predictionManifest'])
            manifest = json.loads(manifest_path.read_text())
            claimed_sha = manifest.get('checkpointSHA256', manifest.get('weightsSHA256'))
            if claimed_sha != model['weights']['sha256']:
                raise ValueError(f'Prediction manifest references other weights: {name}')
    clips = config['clips']
    if len({clip['source']['sha256'] for clip in clips}) != len(clips):
        raise ValueError('Duplicate selected source clip')
    receipt = {'input': str(input_path), 'inputSHA256': sha256(input_path),
               'rendererSHA256': sha256(Path(__file__)), 'models': model_audit,
               'selectionPolicy': config['selectionPolicy'], 'speedFactor': SPEED_FACTOR,
               'inferencePerformed': False, 'smoothingOrInterpolationPerformed': False,
               'layout': 'body baseline / candidate above bat baseline / candidate, full source in each panel',
               'clips': []}
    for number, clip in enumerate(clips, 1):
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', clip['slug']):
            raise ValueError('Unsafe output slug')
        source_path = verified_file(root, clip['source'])
        source_sha = clip['source']['sha256']
        frames_path = verified_file(root, clip['frames'])
        expected_frames = json.loads(frames_path.read_text())
        if [row['index'] for row in expected_frames] != list(range(len(expected_frames))):
            raise ValueError('Selection must include every source frame in original order')
        for index in range(len(expected_frames)):
            identity = (source_sha, index)
            if not all(identity in predictions[name] for name in MODEL_KEYS):
                raise ValueError(f'Missing aligned prediction: {identity}')
        target = output / f'{number:02d}-{clip["slug"]}-comparison-quarter-speed.mp4'
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite an existing export: {target}')
        times, coverage = [], []
        with av.open(str(source_path)) as decoder:
            input_stream = decoder.streams.video[0]
            time_base, rate = input_stream.time_base, input_stream.average_rate
            with av.open(str(target), 'w', options={'movflags': '+faststart'}) as encoder:
                stream = encoder.add_stream('libx264', rate=rate / SPEED_FACTOR)
                stream.width, stream.height = SIZE
                stream.pix_fmt = 'yuv420p'
                stream.time_base = time_base
                stream.codec_context.time_base = time_base
                stream.options = {'crf': '19', 'preset': 'fast', 'bf': '0'}
                for index, frame in enumerate(decoder.decode(video=0)):
                    if index >= len(expected_frames):
                        raise ValueError('Source has more frames than selection manifest')
                    expected = expected_frames[index]
                    if frame.pts != expected['pts'] or frame.time_base != Fraction(expected['timeBase']):
                        raise ValueError(f'Source identity mismatch at frame {index}')
                    if (frame.width, frame.height) != (clip['width'], clip['height']):
                        raise ValueError('Source dimensions differ from selection')
                    if index == 0:
                        first_pts = frame.pts
                    pts = (frame.pts-first_pts) * SPEED_FACTOR
                    times.append(Fraction(pts) * frame.time_base)
                    rows = {name: predictions[name][(source_sha, index)] for name in MODEL_KEYS}
                    source_time = float(frame.pts * frame.time_base)
                    image, shown = render_frame(frame.to_image(), rows, models, clip, number, len(clips),
                                                index, len(expected_frames), source_time)
                    coverage.append({'frameIndex': index, 'sourcePTS': frame.pts,
                                     'sourceTimeBase': str(frame.time_base), 'visiblePoints': shown})
                    encode_frame(encoder, stream, image, pts, frame.time_base)
                for packet in stream.encode():
                    encoder.mux(packet)
        if len(times) != len(expected_frames):
            raise ValueError('Source has fewer frames than selection manifest')
        poster_indices = sorted({0, (len(times)-1)//2, len(times)-1})
        validation = validate_video(target, times, poster_indices, output)
        entry = {**clip, 'output': str(target), 'outputSHA256': sha256(target),
                 'outputBytes': target.stat().st_size, 'validation': validation,
                 'coverage': coverage}
        receipt['clips'].append(entry)
        dump(output / 'export-manifest.json', receipt)
        print(json.dumps({'output': str(target), 'frames': len(times), 'validation': 'all decoded frames and timestamps pass'}), flush=True)
    print(json.dumps({'manifest': str(output/'export-manifest.json'), 'clips': len(clips)}), flush=True)


if __name__ == '__main__':
    main()
