#!/usr/bin/env python3
"""Reproduce the isolated Fan Lab bat Core ML export and same-input parity check.

Run with ../2d-tracking/.venv/bin/python; dependencies in .deps shadow only this process.
No source model, labeler, application, or shared environment files are modified.
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
(ROOT / '.runtime' / 'tmp').mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(ROOT / '.runtime' / 'tmp')
sys.path.insert(0, str(ROOT / '.deps'))
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / '.runtime' / 'yolo')
os.environ['MPLCONFIGDIR'] = str(ROOT / '.runtime' / 'matplotlib')
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import shutil
import time
import zipfile

import av
import coremltools as ct
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

SOURCE = PROJECT / 'experiments/2d-tracking/labeler/models/bat-20261109T164108Z/model.pt'
EXPECTED_SHA = '551e793cc59c9b8129b5be0dca139ec8f506127838cade2fde35ce441ecc35e6'
COPY = ROOT / 'bat-tip-knob-768.pt'
PACKAGE = COPY.with_suffix('.mlpackage')
SIZE = 768
THRESHOLD = .15


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def preprocess(rgb, rect=False):
    """Ultralytics centered letterbox: bilinear RGB uint8, gray 114, no stretch."""
    h, w = rgb.shape[:2]
    gain = min(SIZE / h, SIZE / w)
    resized_w, resized_h = round(w * gain), round(h * gain)
    dw, dh = SIZE - resized_w, SIZE - resized_h
    if rect:
        dw, dh = dw % 32, dh % 32
    left, right = round(dw / 2 - .1), round(dw / 2 + .1)
    top, bottom = round(dh / 2 - .1), round(dh / 2 + .1)
    resized = cv2.resize(rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, {'sourceWidth': w, 'sourceHeight': h, 'gain': gain,
                    'resizedWidth': resized_w, 'resizedHeight': resized_h,
                    'left': left, 'top': top, 'right': right, 'bottom': bottom,
                    'inputWidth': padded.shape[1], 'inputHeight': padded.shape[0]}


def infer_torch(model, rgb):
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).float() / 255
    with torch.inference_mode():
        y = model(tensor)
    return (y[0] if isinstance(y, tuple) else y).cpu().numpy()


def decode(raw, transform):
    """One-class max_det=1: top score survives NMS; never sigmoid scores twice."""
    predictions = raw[0]
    index = int(np.argmax(predictions[4]))
    runner_up_score = float(np.partition(predictions[4], -2)[-2])
    p = predictions[:, index].astype(float)
    keypoints = p[5:].reshape(2, 3)
    xy = keypoints[:, :2].copy()
    xy[:, 0] = (xy[:, 0] - transform['left']) / transform['gain']
    xy[:, 1] = (xy[:, 1] - transform['top']) / transform['gain']
    inside = bool(np.all((xy[:, 0] > 0) & (xy[:, 0] < transform['sourceWidth']) &
                         (xy[:, 1] > 0) & (xy[:, 1] < transform['sourceHeight'])))
    return {'anchorIndex': index, 'score': float(p[4]),
            'runnerUpScore': runner_up_score,
            'topScoreMargin': float(p[4] - runner_up_score),
            'detected': bool(p[4] >= THRESHOLD and inside),
            'tip': xy[0].tolist(), 'knob': xy[1].tolist(),
            'keypointScores': keypoints[:, 2].tolist(), 'insideImage': inside,
            'boxXYWHInputPixels': p[:4].tolist()}


def package_hash():
    entries = [{'path': str(p.relative_to(PACKAGE)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
               for p in sorted(PACKAGE.rglob('*')) if p.is_file()]
    canonical = json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()
    return {'sha256CanonicalFileManifest': hashlib.sha256(canonical).hexdigest(),
            'totalBytes': sum(p['bytes'] for p in entries), 'files': entries}


def archive_package():
    archive = PACKAGE.with_suffix('.mlpackage.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as bundle:
        for path in sorted(PACKAGE.rglob('*')):
            if path.is_file():
                entry = zipfile.ZipInfo(str(path.relative_to(ROOT)), date_time=(1980, 1, 1, 0, 0, 0))
                entry.external_attr = 0o100644 << 16
                bundle.writestr(entry, path.read_bytes())
    return {'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive)}


def main():
    global COPY, PACKAGE
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-export', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--precision', type=int, choices=[16, 32], default=16)
    args = parser.parse_args()
    report_prefix = '' if args.precision == 16 else 'fp32-'
    if args.precision == 32:
        COPY = ROOT / 'bat-tip-knob-768-fp32.pt'
        PACKAGE = COPY.with_suffix('.mlpackage')
    assert sha(SOURCE) == EXPECTED_SHA
    if not COPY.exists():
        shutil.copy2(SOURCE, COPY)
    assert sha(COPY) == EXPECTED_SHA
    torch.set_num_threads(4)
    model = YOLO(str(COPY))
    assert model.task == 'pose'
    assert model.model.model[-1].kpt_shape == [2, 3]
    assert model.names == {0: 'bat'}
    if not args.skip_export:
        exported = model.export(format='coreml', imgsz=SIZE, batch=1, dynamic=False,
                                nms=False, quantize=args.precision, device='cpu')
        assert Path(exported) == PACKAGE
    if args.export_only:
        print('Exported', PACKAGE, flush=True)
        return
    ml = ct.models.MLModel(str(PACKAGE), compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = ml.get_spec()
    input_name = spec.description.input[0].name
    output_name = spec.description.output[0].name
    input_spec = spec.description.input[0].type.imageType
    assert input_spec.width == SIZE and input_spec.height == SIZE
    assert list(spec.description.output[0].type.multiArrayType.shape) == [1, 11, 12096]
    assert spec.description.output[0].type.multiArrayType.dataType == ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32
    assert input_spec.colorSpace == ct.proto.FeatureTypes_pb2.ImageFeatureType.RGB
    # Fresh unfused FP32 eager model is the reference, not Core ML converted back.
    reference = YOLO(str(COPY)).model.cpu().float().eval()
    clips = json.loads((PROJECT / 'experiments/2d-tracking/labeler/clips.json').read_text())['clips']
    selected = ['3f2c8197ea1157fc218f', '57631ed6f2c6033afcee', 'a2ef542ebbe008cbcc80',
                'b371cf77e7eaca79dbfa', 'd6e08b1da2ff2d1d6d7a', 'e139920717760cae047a']
    frames_dir = ROOT / 'parity-inputs'
    frames_dir.mkdir(exist_ok=True)
    rows = []
    for clip in clips:
        if clip['id'] not in selected:
            continue
        indices = sorted(set(round((len(clip['frames']) - 1) * p) for p in [0, .25, .5, .67, .83, 1]))
        with av.open(clip['path']) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i not in indices:
                    continue
                assert frame.pts == clip['frames'][i]['pts']
                # Match labeler JPEG codec path, without touching its cache.
                buffer = io.BytesIO()
                frame.to_image().save(buffer, format='JPEG', quality=97, subsampling=0)
                original = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert('RGB'))
                square, transform = preprocess(original)
                from ultralytics.data.augment import LetterBox
                assert np.array_equal(square, LetterBox((SIZE, SIZE), auto=False)(image=original))
                sample_name = f'{clip["id"]}-{i:04d}.png'
                sample_path = frames_dir / sample_name
                Image.fromarray(square).save(sample_path)
                start = time.perf_counter()
                pt = infer_torch(reference, square)
                pt_seconds = time.perf_counter() - start
                start = time.perf_counter()
                core = np.asarray(ml.predict({input_name: Image.fromarray(square)})[output_name])
                core_seconds = time.perf_counter() - start
                assert pt.shape == core.shape == (1, 11, 12096)
                assert np.isfinite(core).all()
                pt_decoded, core_decoded = decode(pt, transform), decode(core, transform)
                distance = [float(np.linalg.norm(np.array(pt_decoded[k]) - core_decoded[k])) for k in ['tip', 'knob']]
                delta = np.abs(pt - core)
                candidate = pt_decoded['anchorIndex']
                same_candidate = [float(np.linalg.norm(pt[0, ch:ch+2, candidate] -
                                                       core[0, ch:ch+2, candidate]) / transform['gain'])
                                  for ch in [5, 8]]
                rect, rect_transform = preprocess(original, rect=True)
                rect_decoded = decode(infer_torch(reference, rect), rect_transform)
                shape_distance = [float(np.linalg.norm(np.array(pt_decoded[k]) - rect_decoded[k])) for k in ['tip', 'knob']]
                row = {'clipId': clip['id'], 'title': clip['title'], 'frameIndex': i,
                       'source': clip['frames'][i], 'originalVideo': clip['path'],
                       'preprocessedPNG': str(sample_path.relative_to(ROOT)), 'pngSHA256': sha(sample_path),
                       'letterbox': transform, 'pytorch': pt_decoded, 'coreml': core_decoded,
                       'rawAbsoluteErrorMax': float(delta.max()), 'rawAbsoluteErrorMean': float(delta.mean()),
                       'probabilityAbsoluteErrorMax': float(delta[:, [4, 7, 10], :].max()),
                       'endpointErrorOriginalPixels': distance,
                       'samePytorchCandidateEndpointErrorPixels': same_candidate,
                       'samePytorchCandidateRawAbsoluteErrorMax': float(delta[0, :, candidate].max()),
                       'pytorchSeconds': pt_seconds, 'coremlCPUSeconds': core_seconds,
                       'rectangularPytorch': rect_decoded, 'rectangularLetterbox': rect_transform,
                       'squareVersusRectEndpointPixels': shape_distance}
                rows.append(row)
                print(json.dumps({'clip': clip['id'], 'frame': i, 'endpointError': distance,
                                  'scores': [pt_decoded['score'], core_decoded['score']],
                                  'sameAnchor': pt_decoded['anchorIndex'] == core_decoded['anchorIndex']}), flush=True)
    endpoints = [d for row in rows for d in row['endpointErrorOriginalPixels']]
    accepted_shape_deltas = [d for row in rows if row['pytorch']['detected'] and row['rectangularPytorch']['detected']
                             for d in row['squareVersusRectEndpointPixels']]
    summary = {'frames': len(rows), 'endpointErrorMaxPixels': max(endpoints),
               'endpointErrorMeanPixels': float(np.mean(endpoints)),
               'endpointErrorP95Pixels': float(np.percentile(endpoints, 95)),
               'detectionDecisionMismatches': sum(r['pytorch']['detected'] != r['coreml']['detected'] for r in rows),
               'topAnchorMismatches': sum(r['pytorch']['anchorIndex'] != r['coreml']['anchorIndex'] for r in rows),
               'rawAbsoluteErrorMax': max(r['rawAbsoluteErrorMax'] for r in rows),
               'probabilityAbsoluteErrorMax': max(r['probabilityAbsoluteErrorMax'] for r in rows),
               'selectedScoreAbsoluteErrorMax': max(abs(r['pytorch']['score']-r['coreml']['score']) for r in rows),
               'sameCandidateEndpointErrorMaxPixels': max(d for r in rows for d in r['samePytorchCandidateEndpointErrorPixels']),
               'sameCandidateEndpointErrorMeanPixels': float(np.mean([d for r in rows for d in r['samePytorchCandidateEndpointErrorPixels']])),
               'coremlCPUSecondsMedian': float(np.median([r['coremlCPUSeconds'] for r in rows])),
               'squareVsRectDetectionMismatches': sum(r['pytorch']['detected'] != r['rectangularPytorch']['detected'] for r in rows),
               'squareVsRectEndpointMaxPixels': max(d for r in rows for d in r['squareVersusRectEndpointPixels'])}
    summary.update({'pytorchAcceptedFrames': sum(r['pytorch']['detected'] for r in rows),
                    'squareVsRectAcceptedEndpointMaxPixels': max(accepted_shape_deltas),
                    'squareVsRectAcceptedEndpointMeanPixels': float(np.mean(accepted_shape_deltas)),
                    'squareVsRectAcceptedEndpointP95Pixels': float(np.percentile(accepted_shape_deltas, 95))})
    write(report_prefix + 'parity-results.json', {'summary': summary, 'frames': rows})
    write(report_prefix + 'manifest.json', {'sourceModel': str(SOURCE), 'sourceSHA256': EXPECTED_SHA,
          'modelPackage': str(PACKAGE), 'package': package_hash(),
          'zipArchive': archive_package(),
          'export': {'format': 'coreml', 'imgsz': SIZE, 'batch': 1, 'dynamic': False, 'nms': False,
                     'quantize': args.precision, 'device': 'cpu'},
          'input': {'name': input_name, 'type': 'RGB image, uint8 0..255', 'width': SIZE, 'height': SIZE,
                    'inModelScale': 1/255, 'bias': [0, 0, 0], 'resize': 'centered aspect-fit letterbox, gray114'},
          'output': {'name': output_name, 'shape': [1, 11, 12096], 'dtype': 'Float32 MLMultiArray',
                     'axisOrder': ['batch', 'channel', 'anchor'],
                     'channels': ['centerX', 'centerY', 'width', 'height', 'batScore',
                                  'tipX', 'tipY', 'tipScore', 'knobX', 'knobY', 'knobScore'],
                     'coordinates': 'Pixels in letterboxed 768x768 image, top-left origin',
                     'scores': 'Already sigmoid probabilities; no separate objectness', 'nmsIncluded': False},
          'decode': {'threshold': THRESHOLD, 'maxDetections': 1, 'endpointOrder': ['tip', 'knob'],
                     'reverseMap': 'x=(inputX-left)/gain; y=(inputY-top)/gain',
                     'validity': 'both endpoints strictly inside original image; preserve confidence'},
          'toolchain': {name: importlib.metadata.version(name) for name in
                        ['ultralytics', 'torch', 'coremltools', 'numpy', 'protobuf']},
          'host': platform.platform(), 'coremlComputeUnitsTested': 'CPU_ONLY',
          'specificationVersion': spec.specificationVersion, 'modelMetadata': dict(ml.user_defined_metadata),
          'actualOutputDataTypeEnum': spec.description.output[0].type.multiArrayType.dataType,
          'actualInputColorSpaceEnum': input_spec.colorSpace,
          'parity': summary, 'notDeviceLatency': True})
    assert sha(SOURCE) == EXPECTED_SHA
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
