"""Read-only model suggestions, stored separately from human annotations."""
import hashlib
import io
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCK = threading.Lock()
_model = None
_model_id = None


def manifest():
    path = ROOT / 'models' / 'active.json'
    return json.loads(path.read_text()) if path.exists() else None


def status():
    info = manifest()
    progress = ROOT / 'training' / 'status.json'
    training = json.loads(progress.read_text()) if progress.exists() else None
    if training:
        training = {k: training[k] for k in ['state', 'stage', 'epoch', 'epochs', 'labels', 'clips'] if k in training}
    return {'ready': bool(info), 'model': {k: info[k] for k in
            ['id', 'createdAt', 'usableLabels', 'clipCounts', 'evaluationNote']} if info else None,
            'training': training}


def cache_path(model_id, clip_id, frame_index):
    if not re.fullmatch(r'bat-\d{8}T\d{6}Z', model_id):
        raise ValueError('Invalid model identifier')
    if not re.fullmatch(r'[a-f0-9]{20}', clip_id) or type(frame_index) is not int or frame_index < 0:
        raise ValueError('Invalid frame identifier')
    return ROOT / 'cache' / 'predictions' / model_id / f'{clip_id}-{frame_index}.json'


def cached(model_id, clip_id, frame_index):
    path = cache_path(model_id, clip_id, frame_index)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def predict(clip, frame_index, jpeg_loader):
    global _model, _model_id
    info = manifest()
    if not info:
        return {'available': False, 'reason': 'The first model is still training.'}
    path = cache_path(info['id'], clip['id'], frame_index)
    with LOCK:
        if path.exists():
            return json.loads(path.read_text())
        os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'training' / 'config'))
        os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'training' / 'matplotlib'))
        os.environ['YOLO_OFFLINE'] = 'true'
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        import torch
        from PIL import Image
        from ultralytics import YOLO
        if _model_id != info['id']:
            weight_path = ROOT / info['weights']
            if hashlib.sha256(weight_path.read_bytes()).hexdigest() != info['weightsSHA256']:
                raise ValueError('Model checksum mismatch')
            torch.set_num_threads(4)
            _model = YOLO(str(weight_path))
            _model_id = info['id']
        image = Image.open(io.BytesIO(jpeg_loader(clip, frame_index))).convert('RGB')
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        result = _model.predict(image, imgsz=info['prediction']['imgsz'],
                                conf=info['prediction']['threshold'], max_det=1,
                                device=device, verbose=False)[0]
        data = {'available': True, 'modelId': info['id'], 'clipId': clip['id'],
                'frameIndex': frame_index, 'source': clip['frames'][frame_index],
                'createdAt': datetime.now(timezone.utc).isoformat(), 'detected': False,
                'inTrainingSet': f'{clip["id"]}:{frame_index}' in info['trainingKeys'],
                'points': None, 'score': None}
        if len(result.boxes) and result.keypoints is not None:
            coords = result.keypoints.xyn[0].cpu().tolist()
            confidence = result.keypoints.conf[0].cpu().tolist()
            if len(coords) == 2 and all(0 < x < 1 and 0 < y < 1 for x, y in coords):
                data.update(detected=True, score=float(result.boxes.conf[0]),
                            points={name: {'state': 'visible', 'x': xy[0], 'y': xy[1],
                                           'modelScore': confidence[i]}
                                    for i, (name, xy) in enumerate(zip(['tip', 'knob'], coords))})
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data))
        temporary.replace(path)
        return data
