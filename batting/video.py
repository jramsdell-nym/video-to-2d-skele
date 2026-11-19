"""Convert local batting video to timestamped 2D observations."""
import argparse
from contextlib import ExitStack
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from .tracking import HitterSelector

ROOT = Path(__file__).resolve().parents[1]
BODY_NAMES = ('left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
              'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
              'left_knee', 'right_knee', 'left_ankle', 'right_ankle')
BONES = ((0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6),
         (1, 7), (6, 7), (6, 8), (8, 10), (7, 9), (9, 11))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def point(x, y, confidence, width, height, threshold):
    finite = all(math.isfinite(float(v)) for v in (x, y, confidence))
    usable = finite and confidence >= threshold and 0 <= x < width and 0 <= y < height
    return {'x': float(x) if finite else None, 'y': float(y) if finite else None,
            'confidence': float(confidence) if finite else None,
            'normalized': [float(x / width), float(y / height)] if finite else None,
            'usable': bool(usable)}


def unpack(result, names, width, height, threshold):
    if result.keypoints is None or len(result.boxes) == 0:
        return []
    data = result.keypoints.data.cpu().numpy()
    if data.shape[1:] != (len(names), 3):
        raise ValueError(f'Expected {len(names)} keypoints with confidence, got {data.shape}')
    return [{name: point(*values, width, height, threshold)
             for name, values in zip(names, candidate)} for candidate in data]


def rotate_upright(image, rotation):
    import numpy as np
    if not math.isfinite(rotation) or abs(rotation / 90 - round(rotation / 90)) > .001:
        raise ValueError(f'Unsupported video rotation: {rotation}; export an upright video first')
    return np.ascontiguousarray(np.rot90(image, round(rotation / 90) % 4))


def draw_overlay(image, body, bat):
    import cv2
    canvas = image.copy()
    points = body.get('points') or {}
    for first, second in BONES:
        a, b = points.get(BODY_NAMES[first]), points.get(BODY_NAMES[second])
        if a and b and a['usable'] and b['usable']:
            cv2.line(canvas, (round(a['x']), round(a['y'])), (round(b['x']), round(b['y'])), (80, 190, 255), 2)
    for p in points.values():
        if p['usable']:
            cv2.circle(canvas, (round(p['x']), round(p['y'])), 4, (80, 190, 255), -1)
    if bat['accepted']:
        a, b = bat['points']['tip'], bat['points']['knob']
        cv2.line(canvas, (round(a['x']), round(a['y'])), (round(b['x']), round(b['y'])), (255, 255, 255), 3)
        for name, color in [('tip', (80, 210, 255)), ('knob', (255, 200, 100))]:
            p = bat['points'][name]
            xy = (round(p['x']), round(p['y']))
            cv2.circle(canvas, xy, 6, color, 2)
            cv2.putText(canvas, name, (xy[0] + 8, xy[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
    cv2.putText(canvas, f"body: {body['status']} | bat: {'observed' if bat['accepted'] else 'missing'}",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    return canvas


def validate_paths(video, output, overlay, model_paths):
    inputs = [Path(p).resolve() for p in (video, *model_paths)]
    for path in inputs:
        if not path.is_file():
            raise ValueError(f'Missing input: {path}. For model files run git lfs pull.')
    destinations = [Path(p).resolve() for p in (output, overlay) if p is not None]
    if len(set(destinations)) != len(destinations):
        raise ValueError('JSONL and overlay must use different output paths')
    for path in destinations:
        if path in inputs or path.exists():
            raise ValueError(f'Refusing to overwrite: {path}; choose a new output path')


def model_specs(manifest, release):
    if release is None:
        return manifest['defaults']
    releases = manifest.get('releases', {manifest['release']: manifest['defaults']})
    if release not in releases:
        raise ValueError('Unknown model release. Available releases: ' + ', '.join(sorted(releases)))
    return releases[release]


def run(args):
    manifest = json.loads((ROOT / 'models/manifest.json').read_text())
    specs = model_specs(manifest, args.model_release)
    paths = {kind: ROOT / spec['checkpoint'] for kind, spec in specs.items()}
    validate_paths(args.video, args.output, args.overlay, paths.values())
    for kind, path in paths.items():
        if sha256(path) != specs[kind]['sha256']:
            raise ValueError(f'{kind} model checksum mismatch. Run git lfs pull, then python verify_models.py.')
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.cache/ultralytics'))
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    import av
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    settings.update({'sync': False})
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(min(args.threads, 2))
    models = {kind: YOLO(str(path)) for kind, path in paths.items()}
    selector = HitterSelector()
    started = time.perf_counter()
    count = body_count = bat_count = 0
    first_time = previous_time = None
    encoder = None
    output_dimensions = None
    metadata_written = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        source = stack.enter_context(av.open(str(args.video)))
        if not source.streams.video:
            raise ValueError('Input contains no video stream')
        stream = source.streams.video[0]
        writer = stack.enter_context(args.output.open('x'))
        overlay = None
        if args.overlay:
            args.overlay.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also prevents accidental replacement after preflight.
            overlay_file = stack.enter_context(args.overlay.open('xb'))
            overlay = stack.enter_context(av.open(overlay_file, mode='w', format='mp4'))

        def emit(value):
            writer.write(json.dumps(value, allow_nan=False) + '\n')

        for index, frame in enumerate(source.decode(stream)):
            if args.max_frames is not None and index >= args.max_frames:
                break
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Frame has no presentation timestamp; refusing to invent timing')
            timestamp = frame.pts * frame.time_base
            if previous_time is not None and timestamp <= previous_time:
                raise ValueError('Presentation timestamps must strictly increase')
            if first_time is None:
                first_time = timestamp
            previous_time = timestamp
            image = rotate_upright(frame.to_ndarray(format='bgr24'), frame.rotation)
            height, width = image.shape[:2]
            if not metadata_written:
                emit({'type': 'metadata', 'schemaVersion': 1, 'task': 'batting-2d',
                      'sourceFile': args.video.name, 'width': width, 'height': height,
                      'coordinateSystem': 'upright unmirrored source pixels; origin top-left; x right, y down',
                      'bodyOrder': BODY_NAMES, 'batOrder': ['tip', 'knob'],
                      'models': {k: {'checkpoint': v['checkpoint'], 'sha256': v['sha256']} for k, v in specs.items()},
                      'thresholds': {'detection': args.confidence, 'bodyKeypoint': args.body_confidence, 'batKeypoint': args.bat_confidence},
                      'selection': {'body': 'largest visible body then gated temporal association',
                                    'bat': 'highest detection score; independent of body; no wrist association'},
                      'processing': 'no smoothing, interpolation, 3D reconstruction or physical speed estimation',
                      'device': args.device, 'ultralytics': ultralytics.__version__,
                      'maxFrames': args.max_frames})
                metadata_written = True
            predictions = {kind: model.predict(image, imgsz=specs[kind]['inputShape'][-1],
                           rect=False, conf=args.confidence, max_det=10 if kind == 'body' else 1,
                           device=args.device, verbose=False)[0] for kind, model in models.items()}
            poses = unpack(predictions['body'], specs['body']['keypointOrder'], width, height, args.body_confidence)
            # Only 12 limb joints were supervised by the batting body labels.
            poses = [{name: pose[name] for name in BODY_NAMES} for pose in poses]
            selected, status = selector.choose(poses)
            body = {'status': status, 'selectedIndex': selected, 'candidateCount': len(poses),
                    'points': poses[selected] if selected is not None else None}
            bats = unpack(predictions['bat'], ('tip', 'knob'), width, height, args.bat_confidence)
            candidate = bats[0] if bats else None
            accepted = bool(candidate and all(p['usable'] for p in candidate.values()))
            bat = {'accepted': accepted, 'points': candidate if accepted else None,
                   'candidate': candidate,
                   'detectionConfidence': float(predictions['bat'].boxes.conf[0]) if bats else None}
            emit({'type': 'frame', 'frameIndex': index, 'pts': frame.pts,
                  'timeBase': str(frame.time_base), 'timeSeconds': float(timestamp),
                  'relativeTimeSeconds': float(timestamp - first_time),
                  'width': width, 'height': height, 'sourceRotationCCW': frame.rotation,
                  'body': body, 'bat': bat})
            if overlay is not None:
                dimensions = (width + width % 2, height + height % 2)
                if encoder is None:
                    output_dimensions = dimensions
                    encoder = overlay.add_stream('libx264', rate=stream.average_rate or Fraction(30))
                    encoder.width, encoder.height = dimensions
                    encoder.pix_fmt = 'yuv420p'
                    encoder.time_base = frame.time_base
                    encoder.codec_context.time_base = frame.time_base
                    encoder.options = {'crf': '20', 'preset': 'fast'}
                if dimensions != output_dimensions:
                    raise ValueError('Overlay requires constant video dimensions')
                canvas = draw_overlay(image, body, bat)
                canvas = cv2.copyMakeBorder(canvas, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT)
                rendered = av.VideoFrame.from_ndarray(canvas, format='bgr24')
                rendered.time_base = frame.time_base
                rendered.pts = round((timestamp - first_time) / frame.time_base)
                for packet in encoder.encode(rendered):
                    overlay.mux(packet)
            count += 1
            body_count += selected is not None
            bat_count += accepted
            if count % 60 == 0:
                print(f'Processed {count} frames', file=sys.stderr, flush=True)
        if count == 0:
            raise ValueError('No video frames were decoded')
        if encoder is not None:
            for packet in encoder.encode():
                overlay.mux(packet)
        summary = {'type': 'summary', 'complete': True, 'frames': count,
                   'bodyObservedFrames': body_count, 'batAcceptedFrames': bat_count,
                   'elapsedSeconds': time.perf_counter() - started,
                   'limitedByMaxFrames': args.max_frames is not None and count == args.max_frames}
        emit(summary)
    print(json.dumps(summary, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New JSONL file')
    parser.add_argument('--overlay', type=Path, help='Optional new silent MP4 with 2D overlays')
    parser.add_argument('--device', default='cpu', help='cpu (default), mps on Apple silicon, or 0 for CUDA')
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--model-release', help='Use an archived body and bat pair by release ID')
    parser.add_argument('--body-confidence', type=float, default=.30)
    parser.add_argument('--bat-confidence', type=float, default=.25)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-frames', type=int, help='Optional positive frame limit for a smoke test')
    args = parser.parse_args(argv)
    if any(not 0 <= value <= 1 for value in (args.confidence, args.body_confidence, args.bat_confidence)):
        parser.error('Confidence thresholds must be between 0 and 1')
    if args.threads < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error('threads and max-frames must be positive')
    try:
        run(args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\nIncomplete output files, if any, should be removed before retrying.\n')


if __name__ == '__main__':
    main()
