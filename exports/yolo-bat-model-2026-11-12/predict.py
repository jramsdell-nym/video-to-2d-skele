#!/usr/bin/env python3
"""Run the packaged two-endpoint bat model on one upright image."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
SIZE = 768


def preprocess(image):
    """Centered square RGB letterbox; matches the packaged model contract."""
    width, height = image.size
    gain = min(SIZE / width, SIZE / height)
    resized_width, resized_height = round(width * gain), round(height * gain)
    left = round((SIZE - resized_width) / 2 - 0.1)
    top = round((SIZE - resized_height) / 2 - 0.1)
    resized = cv2.resize(np.asarray(image), (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    padded = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
    padded[top:top + resized_height, left:left + resized_width] = resized
    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, {'width': width, 'height': height, 'gain': gain, 'left': left, 'top': top}


def decode(raw, transform, confidence=0.25, keypoint_confidence=0.25):
    """Top-scoring bat candidate; preserves endpoint identities and missingness."""
    if raw.shape != (1, 11, 12096) or not np.isfinite(raw).all():
        raise ValueError('Expected finite model output shaped [1, 11, 12096]')
    index = int(np.argmax(raw[0, 4]))
    candidate = raw[0, :, index].astype(float)
    endpoints = {}
    for name, start in [('tip', 5), ('knob', 8)]:
        x = (candidate[start] - transform['left']) / transform['gain']
        y = (candidate[start + 1] - transform['top']) / transform['gain']
        score = float(candidate[start + 2])
        inside = 0 <= x < transform['width'] and 0 <= y < transform['height']
        endpoints[name] = {'x': float(x), 'y': float(y), 'confidence': score,
                           'normalized': [float(x / transform['width']), float(y / transform['height'])],
                           'usable': bool(inside and score >= keypoint_confidence)}
    reasons = []
    if candidate[4] < confidence:
        reasons.append('bat_score_below_threshold')
    reasons.extend(name + '_unusable' for name, point in endpoints.items() if not point['usable'])
    accepted = not reasons
    return {'accepted': accepted, 'reasons': reasons,
            'points': {name: [p['x'], p['y']] for name, p in endpoints.items()} if accepted else None,
            'candidate': {'anchorIndex': index, 'confidence': float(candidate[4]), **endpoints},
            'image': {'width': transform['width'], 'height': transform['height']},
            'coordinateSystem': 'EXIF-upright original image pixels, x right, y down',
            'thresholds': {'bat': confidence, 'endpoint': keypoint_confidence},
            'selectionPolicy': 'highest bat score; no hitter association, smoothing or gap filling'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/bat-tip-knob.onnx')
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--keypoint-confidence', type=float, default=0.25)
    parser.add_argument('--output', type=Path, help='Optional output JSON; also printed to stdout')
    parser.add_argument('--annotated', type=Path, help='Optional output image with accepted endpoints')
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1 or not 0 <= args.keypoint_confidence <= 1:
        parser.error('Confidence thresholds must be between zero and one')
    if not args.image.is_file() or not args.model.is_file():
        parser.error('Input image and model must exist locally')
    for output in [args.output, args.annotated]:
        if output and output.resolve() in [args.image.resolve(), args.model.resolve()]:
            parser.error('Output must not overwrite the input image or model')
    with Image.open(args.image) as opened:
        image = ImageOps.exif_transpose(opened).convert('RGB')
    tensor, transform = preprocess(image)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=['CPUExecutionProvider'])
    output = session.run(['output0'], {'images': tensor})[0]
    result = decode(output, transform, args.confidence, args.keypoint_confidence)
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(text)
    if args.annotated:
        draw = ImageDraw.Draw(image)
        if result['accepted']:
            points = result['points']
            draw.line([tuple(points['tip']), tuple(points['knob'])], fill='white', width=3)
            for name, color in [('tip', '#ffce69'), ('knob', '#74cefa')]:
                x, y = points[name]
                draw.ellipse((x-6, y-6, x+6, y+6), outline=color, width=3)
                draw.text((x+9, y+9), name.upper(), fill=color)
        else:
            draw.text((12, 12), 'No accepted bat endpoint pair', fill='white', stroke_width=2, stroke_fill='black')
        image.save(args.annotated)
    print(text, end='')


if __name__ == '__main__':
    main()
