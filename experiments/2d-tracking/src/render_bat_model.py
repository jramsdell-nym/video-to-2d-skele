"""Render a preselected cohort of unseen swings with the local endpoint model."""
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'labeler/training/config'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'labeler/training/matplotlib'))
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['YOLO_AUTOINSTALL'] = 'false'

import av
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

SELECTION = [
    ('jacob', '821a299e99cb0a60b164', 'Jacob Madrid', 'Right-handed · different swing from a training hitter'),
    ('kuhio', '81ee6b14d6abc41a9c7c', 'Kuhio Aloy', 'Right-handed · different swing from a training hitter'),
    ('trey', 'a386856f73f8c0abec55', 'Trey Snyder', 'Right-handed · different swing from a training hitter'),
    ('elian', '5a2441e765feb0f43bdd', 'Elian Peña', 'Left-handed · hitter not represented in training'),
    ('november6', '09874fd802816c950e4f', 'November 6 · left-handed swing', 'Different recording session · hitter identity unverified'),
]
TIP, KNOB = (105, 206, 255), (250, 206, 116)  # BGR; same colors as the labeler.
WIDTH, HEIGHT, TOP = 1920, 1200, 88


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def writer(path, rate, time_base):
    container = av.open(str(path), 'w', options={'movflags': '+faststart'})
    stream = container.add_stream('libx264', rate=rate)
    stream.width, stream.height = WIDTH, HEIGHT
    stream.pix_fmt = 'yuv420p'
    stream.time_base = time_base
    stream.codec_context.time_base = time_base
    stream.options = {'crf': '19', 'preset': 'fast', 'bf': '0'}
    return container, stream


def put(pair, image, pts, time_base):
    container, stream = pair
    frame = av.VideoFrame.from_ndarray(image, format='bgr24')
    frame.pts, frame.time_base = pts, time_base
    for packet in stream.encode(frame):
        container.mux(packet)


def close(pair):
    container, stream = pair
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def static_frame(title, subtitle, number, factor, model_info):
    image = Image.new('RGB', (WIDTH, HEIGHT), '#10171e')
    draw = ImageDraw.Draw(image)
    font = '/System/Library/Fonts/Helvetica.ttc'
    large, small = ImageFont.truetype(font, 33), ImageFont.truetype(font, 20)
    draw.text((26, 10), title, fill='#eef3f5', font=large)
    draw.text((26, 51), f'{subtitle}  |  {"¼ speed" if factor == 4 else "Real time"}  |  Unseen clip', fill='#a5b8c6', font=small)
    draw.text((1770, 15), f'{number} / 5', fill='#a6edcb', font=large)
    draw.text((26, 1172), f'BAT MODEL · {model_info["usableLabels"]} training frames  ·  Automatic predictions  ·  No smoothing or gap filling', fill='#a5b8c6', font=small)
    draw.ellipse((1400, 1179, 1411, 1190), fill='#ffce69')
    draw.text((1422, 1172), 'Barrel tip', fill='#ffce69', font=small)
    draw.ellipse((1600, 1179, 1611, 1190), fill='#74cefa')
    draw.text((1622, 1172), 'Handle knob', fill='#74cefa', font=small)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def annotate(source, points):
    image = source.copy()
    if not points:
        return image
    pixels = {k: tuple(np.round(np.array([p['x'], p['y']]) * [source.shape[1], source.shape[0]]).astype(int))
              for k, p in points.items()}
    cv2.line(image, pixels['tip'], pixels['knob'], (12, 17, 22), 7, cv2.LINE_AA)
    cv2.line(image, pixels['tip'], pixels['knob'], (235, 238, 231), 2, cv2.LINE_AA)
    for name, color in [('tip', TIP), ('knob', KNOB)]:
        x, y = pixels[name]
        cv2.circle(image, (x, y), 12, (12, 17, 22), 6, cv2.LINE_AA)
        cv2.circle(image, (x, y), 12, color, 3, cv2.LINE_AA)
        cv2.drawMarker(image, (x, y), color, cv2.MARKER_CROSS, 34, 2, cv2.LINE_AA)
        tx = max(4, min(source.shape[1] - 100, x + 19))
        ty = max(25, y - 16)
        cv2.rectangle(image, (tx - 5, ty - 23), (tx + 83, ty + 7), (12, 17, 22), -1)
        cv2.putText(image, name.upper(), (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, .66, color, 2, cv2.LINE_AA)
    return image


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_info = json.loads((ROOT / 'labeler/models/active.json').read_text())
    weights = ROOT / 'labeler' / model_info['weights']
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == model_info['weightsSHA256']
    clips = {c['id']: c for c in json.loads((ROOT / 'labeler/clips.json').read_text())['clips']}
    assert all(cid not in model_info['clipCounts'] for _, cid, _, _ in SELECTION)
    output = ROOT / 'outputs/bat-model-review' / model_info['id']
    output.mkdir(parents=True, exist_ok=True)
    selection = {'modelId': model_info['id'], 'selectedAt': datetime.now(timezone.utc).isoformat(),
                 'selectionRule': 'Five clips selected before inference for different hitters, handedness and session; no selected video contributed training labels.',
                 'clips': [{'slug': slug, 'clipId': cid, 'title': title, 'description': desc}
                           for slug, cid, title, desc in SELECTION]}
    dump(output / 'selection.json', selection)
    model = YOLO(str(weights))
    review = {'modelId': model_info['id'], 'weightsSHA256': model_info['weightsSHA256'],
              'trainingLabels': model_info['usableLabels'], 'trainingVideos': len(model_info['clipCounts']),
              'includedCompleteManualDrafts': model_info.get('includedCompleteManualDrafts', 0), 'device': device,
              'prediction': model_info['prediction'], 'coordinateSystem': 'original unmirrored source image; normalized top-left coordinates',
              'rendering': 'Every decoded source frame, no smoothing, interpolation or gap filling. Entire original image beneath an added title bar. Silent H.264 exports at real time and quarter speed.',
              'coverageNote': 'Usable-prediction counts are coverage, not localization accuracy. There are no manual reference labels for these clips in this review.',
              'clips': []}
    all_posters = []
    for number, (slug, cid, title, desc) in enumerate(SELECTION, 1):
        clip = clips[cid]
        assert hashlib.sha256(Path(clip['path']).read_bytes()).hexdigest() == clip['sha256']
        rows, samples, durations = [], [], []
        files = {factor: output / f'{slug}-{"quarter-speed" if factor == 4 else "realtime"}.mp4' for factor in [1, 4]}
        started = time.perf_counter()
        with av.open(clip['path']) as container:
            stream = container.streams.video[0]
            rate, tb = stream.average_rate, stream.time_base
            writers = {factor: writer(path, rate / factor, tb) for factor, path in files.items()}
            bases = {factor: static_frame(title, clip['trial'].removeprefix('markers_') + ' · ' + ('Left-handed' if clip['side'] == 'L' else 'Right-handed'), number, factor, model_info) for factor in files}
            t0 = clip['frames'][0]['pts']
            sample_indices = sorted(set(round((len(clip['frames']) - 1) * p) for p in [0, .25, .5, .67, .83, 1]))
            try:
                for i, frame in enumerate(container.decode(video=0)):
                    expected = clip['frames'][i]
                    assert frame.pts == expected['pts'] and str(frame.time_base) == expected['timeBase']
                    assert (frame.width, frame.height) == (1920, 1080)
                    rgb = frame.to_image()
                    start = time.perf_counter()
                    result = model.predict(rgb, imgsz=model_info['prediction']['imgsz'],
                                           conf=model_info['prediction']['threshold'], max_det=1,
                                           device=device, verbose=False)[0]
                    durations.append(time.perf_counter() - start)
                    points, score, raw = None, None, None
                    if len(result.boxes) and result.keypoints is not None:
                        xy = result.keypoints.xyn[0].cpu().tolist()
                        scores = result.keypoints.conf[0].cpu().tolist()
                        score = float(result.boxes.conf[0])
                        raw = {'box': result.boxes.xyxyn[0].cpu().tolist(), 'keypoints': xy, 'keypointScores': scores}
                        if len(xy) == 2 and all(0 < x < 1 and 0 < y < 1 for x, y in xy):
                            points = {name: {'x': p[0], 'y': p[1], 'score': scores[k]}
                                      for k, (name, p) in enumerate(zip(['tip', 'knob'], xy))}
                    rows.append({'frameIndex': i, 'source': expected, 'points': points,
                                 'score': score, 'raw': raw, 'status': 'predicted' if points else 'no_usable_prediction'})
                    source = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
                    marked = annotate(source, points)
                    for factor, pair in writers.items():
                        image = bases[factor].copy()
                        image[TOP:TOP + 1080] = marked
                        status = f"Frame {i + 1}/{len(clip['frames'])} | source {expected['time']:.3f}s"
                        cv2.putText(image, status, (1200, 36), cv2.FONT_HERSHEY_SIMPLEX, .66, (208, 221, 230), 1, cv2.LINE_AA)
                        if not points:
                            cv2.putText(image, 'NO USABLE PREDICTION', (1200, 69), cv2.FONT_HERSHEY_SIMPLEX, .62, (135, 160, 255), 2, cv2.LINE_AA)
                        put(pair, image, (frame.pts - t0) * factor, frame.time_base)
                        if factor == 4 and i in sample_indices:
                            samples.append(cv2.resize(image, (640, 400)))
                            if i == sample_indices[2]:
                                cv2.imwrite(str(output / f'{slug}-poster.jpg'), image)
                assert len(rows) == len(clip['frames'])
            finally:
                for pair in writers.values():
                    close(pair)
        usable = sum(r['points'] is not None for r in rows)
        entry = {'slug': slug, 'clipId': cid, 'title': title, 'description': desc, 'side': clip['side'],
                 'trial': clip['trial'], 'session': clip['session'], 'videoSHA256': clip['sha256'],
                 'trainingOverlapFrames': 0, 'sourceFrames': len(rows), 'usablePredictions': usable,
                 'missingPredictions': len(rows) - usable, 'sourceRate': str(rate),
                 'videos': {'quarter': str(files[4].relative_to(ROOT)), 'realtime': str(files[1].relative_to(ROOT))},
                 'poster': str((output / f'{slug}-poster.jpg').relative_to(ROOT)),
                 'predictions': str((output / f'{slug}-predictions.json').relative_to(ROOT)),
                 'meanInferenceMilliseconds': round(np.mean(durations) * 1000, 2)}
        dump(output / f'{slug}-predictions.json', {**entry, 'modelId': model_info['id'], 'frames': rows})
        cv2.imwrite(str(output / f'{slug}-contact-sheet.jpg'), np.vstack([np.hstack(samples[:3]), np.hstack(samples[3:])]))
        review['clips'].append(entry)
        all_posters.append(samples[2])
        dump(output / 'review.json', review)
        print(json.dumps({'clip': slug, 'usable': usable, 'frames': len(rows), 'elapsedSeconds': round(time.perf_counter() - started, 1)}), flush=True)
    cv2.imwrite(str(output / 'all-swings.jpg'), np.vstack(all_posters))
    # Keep the current gallery working until the new exports have passed validation.
    (output / 'gallery-data.js').write_text('window.BAT_REVIEW=' + json.dumps(review, ensure_ascii=False) + ';')
    print('Finished: ' + str(output), flush=True)


if __name__ == '__main__':
    main()
