"""Immutable browser exports and native parity checks for the Nov 18 release."""
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
CYCLE = ROOT.parent.parent / 'cycles/2026-11-18'
SPECS = {
    'body': (640, 'edf3f8c06a8cf43d6c1727909222f46019306229969c2a0ebef0107b979ae9ee', 56),
    'bat': (768, '4f815fad7b24d325770ed0f2234fd61c138b1b90abd2ad97ce4c94a45206bfab', 11),
}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(4)
report = {'ultralytics': ultralytics.__version__, 'onnx': onnx.__version__,
          'onnxruntime': ort.__version__, 'opset': 17, 'models': {}}
groups = defaultdict(list)
for row in json.loads((CYCLE / 'training/body/evaluation-rows.json').read_text()):
    groups[row['videoSHA256']].append(row)
samples = []
for rows in groups.values():
    rows.sort(key=lambda r: r['frameIndex'])
    samples.extend(rows[i] for i in (0, len(rows)//2, len(rows)-1))
for kind, (size, expected, channels) in SPECS.items():
    source = CYCLE / f'training/{kind}/candidate/weights/last.pt'
    assert sha(source) == expected
    checkpoint = ROOT / f'{kind}-2026-11-18.pt'
    shutil.copy2(source, checkpoint)
    exported = Path(YOLO(str(checkpoint)).export(format='onnx', imgsz=size,
        batch=1, dynamic=False, simplify=False, opset=17, nms=False, device='cpu'))
    graph = onnx.load(exported)
    onnx.checker.check_model(graph)
    assert all(not t.external_data for t in graph.graph.initializer)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(exported), options, providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].shape == [1,3,size,size]
    reference = YOLO(str(checkpoint)).model.cpu().float().eval()
    reference.fuse()
    reference.model[-1].export = True
    reference.model[-1].format = 'onnx'
    cases = []
    for number, row in enumerate(samples):
        image_path = Path(row['imagePath'])
        assert sha(image_path) == row['imageSHA256']
        image = cv2.imread(str(image_path))
        h,w = image.shape[:2]
        ratio = min(size/w, size/h)
        rw,rh = round(w*ratio), round(h*ratio)
        px,py = (size-rw)//2,(size-rh)//2
        canvas = np.full((size,size,3), 114, dtype=np.uint8)
        canvas[py:py+rh,px:px+rw] = cv2.resize(image,(rw,rh),interpolation=cv2.INTER_LINEAR)
        tensor = np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
        with torch.inference_mode(): native = reference(torch.from_numpy(tensor)).numpy()
        browser = session.run(None, {'images': tensor})[0]
        assert list(browser.shape) == [1,channels,8400 if kind=='body' else 12096]
        assert np.isfinite(browser).all()
        selected = int(native[0,4].argmax())
        assert selected == int(browser[0,4].argmax())
        delta = np.abs(native[0,:,selected]-browser[0,:,selected])
        xy_indices = [i for j in range((channels-5)//3) for i in (5+j*3,6+j*3)]
        pixel_error = float(delta[xy_indices].max()/ratio)
        confidence_error = float(delta[[4]+list(range(7,channels,3))].max())
        assert pixel_error < .1, pixel_error
        assert confidence_error < .0001, confidence_error
        case = {'videoSHA256':row['videoSHA256'],'frameIndex':row['frameIndex'],
            'imagePath':str(image_path),'imageSHA256':row['imageSHA256'],
            'selected':selected,'score':float(browser[0,4,selected]),
            'maxRawAbsError':float(np.abs(native-browser).max()),
            'maxSelectedSourcePixelError':pixel_error,'maxSelectedConfidenceError':confidence_error}
        cases.append(case)
        if number == 1:
            tensor.tofile(ROOT / f'{kind}-input.f32')
            browser.tofile(ROOT / f'{kind}-output.f32')
            (ROOT / f'{kind}-golden.json').write_text(json.dumps({**case,'inputShape':list(tensor.shape),
                'outputShape':list(browser.shape),'transform':{'width':w,'height':h,'size':size,'padX':px,'padY':py,'scaleX':rw/w,'scaleY':rh/h}}, indent=2))
        print(kind,number+1,'/',len(samples),'pixel delta',pixel_error,flush=True)
    report['models'][kind] = {'checkpoint':str(source),'checkpointSHA256':expected,
        'path':str(exported),'sha256':sha(exported),'size':exported.stat().st_size,
        'inputShape':session.get_inputs()[0].shape,'outputShape':session.get_outputs()[0].shape,
        'metadata':{p.key:p.value for p in graph.metadata_props},'parityCases':cases}
    (ROOT/'release.json').write_text(json.dumps(report,indent=2))
    print('VERIFIED',kind,report['models'][kind]['sha256'],flush=True)
