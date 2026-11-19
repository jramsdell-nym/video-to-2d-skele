"""Train a local two-endpoint pose model from an immutable human-label snapshot.

Run from the experiment: .venv/bin/python labeler/train_bat.py
No skipped frames, unfinished drafts, or unreviewed predictions are training data.
An explicit opt-in can include complete, manually clicked drafts without changing
their saved status. This is useful when the user labeled with arrow navigation.
"""
import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'training' / 'config'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'training' / 'matplotlib'))
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['YOLO_AUTOINSTALL'] = 'false'

import av
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils import SETTINGS


def write_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def training_rows(rows, include_complete_manual_drafts=False):
    selected = []
    for row in rows:
        points = row.get('points', {})
        if not all(points.get(k, {}).get('state') == 'visible' for k in ['tip', 'knob']):
            continue
        reviewed = row['status'] == 'reviewed'
        manual_draft = (include_complete_manual_drafts and row['status'] == 'draft'
                        and row.get('origin') == 'manual_labeler' and not row.get('assistance'))
        if reviewed or manual_draft:
            selected.append(row)
    return selected


def evaluate(model_path, rows, dataset, device):
    """All held-out frames count; do not drop detection misses from the report."""
    model = YOLO(str(model_path))
    details = []
    for row in rows:
        name = f"{row['clipId']}-{row['frameIndex']:04d}"
        result = model.predict(str(dataset / 'images' / (name + '.jpg')),
                               imgsz=768, conf=0.15, max_det=1, device=device, verbose=False)[0]
        item = {'clipId': row['clipId'], 'frameIndex': row['frameIndex'], 'detected': False}
        if len(result.boxes):
            xy = result.keypoints.xy[0].cpu().numpy()
            truth = np.array([[row['points'][k]['pixelX'], row['points'][k]['pixelY']]
                              for k in ['tip', 'knob']])
            errors = np.linalg.norm(xy - truth, axis=1)
            item.update(detected=True, tipErrorPixels=float(errors[0]), knobErrorPixels=float(errors[1]),
                        prediction=xy.tolist(), score=float(result.boxes.conf[0]))
        details.append(item)
    detected = [d for d in details if d['detected']]
    errors = [d[k] for d in detected for k in ['tipErrorPixels', 'knobErrorPixels']]
    return {'frames': len(rows), 'detected': len(detected), 'missed': len(rows) - len(detected),
            'meanEndpointErrorPixelsOnDetections': float(np.mean(errors)) if errors else None,
            'medianEndpointErrorPixelsOnDetections': float(np.median(errors)) if errors else None,
            'bothEndpointsWithin25Pixels': sum(max(d['tipErrorPixels'], d['knobErrorPixels']) <= 25 for d in detected),
            'bothEndpointsWithin50Pixels': sum(max(d['tipErrorPixels'], d['knobErrorPixels']) <= 50 for d in detected),
            'threshold': 0.15, 'details': details}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--refit-epochs', type=int, default=20)
    parser.add_argument('--holdout', default='e139920717760cae047a')
    parser.add_argument('--include-complete-manual-drafts', action='store_true',
                        help='Explicitly include drafts with both endpoints manually marked; never model-assisted or incomplete drafts')
    args = parser.parse_args()
    torch.set_num_threads(4)
    SETTINGS.update({k: False for k in ['sync', 'wandb', 'mlflow', 'comet', 'clearml', 'hub', 'tensorboard', 'neptune', 'dvc'] if k in SETTINGS})
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    run_id = 'bat-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = ROOT / 'training' / run_id
    run.mkdir(parents=True)
    status_path = ROOT / 'training' / 'status.json'
    started = time.monotonic()
    status = {'state': 'preparing', 'runId': run_id}
    write_json(status_path, status)
    try:
        with sqlite3.connect(f'file:{ROOT / "annotations.sqlite3"}?mode=ro', uri=True) as src:
            with sqlite3.connect(run / 'labels.sqlite3') as dest:
                src.backup(dest)
                all_rows = [json.loads(r[0]) for r in dest.execute('SELECT record FROM labels ORDER BY clip,frame')]
        rows = training_rows(all_rows, args.include_complete_manual_drafts)
        # A pose box cannot be reconstructed reliably from just one endpoint.
        # Keep such labels in the snapshot, but do not train fabricated targets.
        counts = Counter(r['clipId'] for r in rows)
        if len(counts) < 2 or len(rows) < 10:
            raise ValueError('Need at least 10 fully visible reviewed frames across two clips')
        if args.holdout not in counts:
            raise ValueError('Holdout clip has no usable labels; choose --holdout explicitly')
        clips = {c['id']: c for c in json.loads((ROOT / 'clips.json').read_text())['clips']}
        dataset = run / 'dataset'
        (dataset / 'images').mkdir(parents=True)
        (dataset / 'labels').mkdir()
        for clip_id in counts:
            c = clips[clip_id]
            assert hashlib.sha256(Path(c['path']).read_bytes()).hexdigest() == c['sha256']
            wanted = {r['frameIndex']: r for r in rows if r['clipId'] == clip_id}
            with av.open(c['path']) as container:
                for i, frame in enumerate(container.decode(video=0)):
                    if i not in wanted:
                        continue
                    row = wanted[i]
                    assert frame.pts == row['source']['pts'] and str(frame.time_base) == row['source']['timeBase']
                    assert frame.width == row['imageWidth'] and frame.height == row['imageHeight']
                    name = f'{clip_id}-{i:04d}'
                    frame.to_image().save(dataset / 'images' / (name + '.jpg'), quality=97, subsampling=0)
                    pts = np.array([[row['points'][k]['x'], row['points'][k]['y']] for k in ['tip', 'knob']])
                    margin = np.array([24 / frame.width, 24 / frame.height])
                    low, high = np.maximum(0, pts.min(0) - margin), np.minimum(1, pts.max(0) + margin)
                    center, size = (low + high) / 2, high - low
                    values = [0, *center, *size, *pts[0], 2, *pts[1], 2]
                    (dataset / 'labels' / (name + '.txt')).write_text(' '.join(str(float(v)) if j else '0' for j, v in enumerate(values)) + '\n')
        train = [r for r in rows if r['clipId'] != args.holdout]
        val = [r for r in rows if r['clipId'] == args.holdout]
        for name, group in [('train', train), ('val', val), ('all', rows)]:
            (dataset / (name + '.txt')).write_text(''.join(str(dataset / 'images' / f"{r['clipId']}-{r['frameIndex']:04d}.jpg") + '\n' for r in group))
        config = {'path': str(dataset), 'train': 'train.txt', 'val': 'val.txt', 'names': {0: 'bat'},
                  'kpt_shape': [2, 3], 'flip_idx': [0, 1]}
        (dataset / 'split.yaml').write_text(yaml.safe_dump(config))
        # Refit includes the former holdout; its validation is NOT independent.
        (dataset / 'all.yaml').write_text(yaml.safe_dump({**config, 'train': 'all.txt'}))
        manifest = {'id': run_id, 'createdAt': datetime.now(timezone.utc).isoformat(), 'device': device,
                    'sourceModel': 'yolo11s-pose.pt', 'endpointOrder': ['tip', 'knob'],
                    'snapshotRows': len(all_rows), 'usableLabels': len(rows), 'clipCounts': dict(counts),
                    'includedCompleteManualDrafts': sum(r['status'] == 'draft' for r in rows),
                    'includeCompleteManualDrafts': args.include_complete_manual_drafts,
                    'holdoutClipId': args.holdout, 'holdoutTitle': clips[args.holdout]['title'],
                    'initialTrainCount': len(train), 'heldoutCount': len(val),
                    'exclusions': 'Skips, absent frames, incomplete or model-assisted drafts, and frames without two visible endpoints are excluded. Complete manual drafts are included only with explicit opt-in.',
                    'bbox': 'Endpoint envelope plus 24 original-image pixels each side; clipped to image.',
                    'labels': rows, 'docs': 'https://docs.ultralytics.com/datasets/pose/'}
        write_json(run / 'snapshot.json', manifest)

        def progress(trainer):
            status.update(epoch=trainer.epoch + 1, epochs=trainer.epochs, elapsedSeconds=round(time.monotonic() - started))
            write_json(status_path, status)

        params = dict(imgsz=768, batch=8, workers=0, device=device, optimizer='AdamW', lr0=0.0008,
                      lrf=0.05, weight_decay=0.0005, warmup_epochs=3, nbs=8, freeze=10,
                      seed=42, deterministic=True, patience=25, plots=False, amp=False,
                      degrees=10, translate=0.12, scale=0.25, fliplr=0.5, flipud=0,
                      mosaic=0.25, close_mosaic=10, mixup=0, hsv_h=0.01, hsv_s=0.25, hsv_v=0.25,
                      project=str(run), exist_ok=False, verbose=False, save=True)
        status.update(state='training', stage='heldout', labels=len(rows), clips=len(counts))
        write_json(status_path, status)
        model = YOLO(str(EXP / 'models' / 'yolo11s-pose.pt'))
        model.add_callback('on_train_epoch_end', progress)
        model.train(data=str(dataset / 'split.yaml'), epochs=args.epochs, name='holdout', **params)
        initial_weights = run / 'holdout' / 'weights' / 'best.pt'
        report = evaluate(initial_weights, val, dataset, device)
        write_json(run / 'heldout-evaluation.json', report)
        status.update(state='training', stage='refit-all-reviewed', epoch=0, heldout=report)
        write_json(status_path, status)
        model = YOLO(str(initial_weights))
        model.add_callback('on_train_epoch_end', progress)
        model.train(data=str(dataset / 'all.yaml'), epochs=args.refit_epochs, name='refit',
                    **{**params, 'lr0': 0.0003, 'patience': 0, 'mosaic': 0, 'warmup_epochs': 1})
        # Last checkpoint is the fixed-length refit on ALL usable reviewed labels.
        final_weights = run / 'refit' / 'weights' / 'last.pt'
        model_dir = ROOT / 'models' / run_id
        model_dir.mkdir(parents=True)
        shutil.copy2(final_weights, model_dir / 'model.pt')
        public_manifest = {k: v for k, v in manifest.items() if k != 'labels'}
        public_manifest.update(weights=f'models/{run_id}/model.pt',
                               weightsSHA256=hashlib.sha256((model_dir / 'model.pt').read_bytes()).hexdigest(),
                               trainingKeys=[f"{r['clipId']}:{r['frameIndex']}" for r in rows],
                               heldoutEvaluation=report,
                               evaluationNote='Whole-video validation before refitting on all labels. Used for checkpoint selection, not an unbiased test set. The deployed model then includes that video; these are not final-model accuracy metrics.',
                               prediction={'imgsz': 768, 'threshold': 0.15}, elapsedSeconds=round(time.monotonic() - started))
        write_json(model_dir / 'manifest.json', public_manifest)
        write_json(ROOT / 'models' / 'active.json', public_manifest)
        status.update(state='ready', modelId=run_id, elapsedSeconds=round(time.monotonic() - started))
        write_json(status_path, status)
        print(json.dumps({'model': run_id, 'labels': len(rows), 'heldout': report, 'seconds': status['elapsedSeconds']}), flush=True)
    except Exception as exc:
        status.update(state='failed', error=str(exc))
        write_json(status_path, status)
        raise


if __name__ == '__main__':
    main()
