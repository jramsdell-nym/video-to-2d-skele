#!/usr/bin/env python3
"""Build a private-data-free two-model handoff after a training cycle completes.

Use --spec for a frozen private JSON record containing checkpoint paths/hashes,
15 private parity rows (three from each of five videos), and aggregate scores.
No model defaults, source checkpoints, or input datasets are modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import pickletools
import platform
import re
import shutil
from string import Template
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = Path(__file__).resolve().parent / 'package_templates'
BODY_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
              'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
              'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
              'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
RUNTIME_PACKAGES = ('ultralytics', 'torch', 'torchvision', 'numpy', 'opencv-python', 'av', 'Pillow')
EXPORT_PACKAGES = ('onnx', 'onnxruntime')
SOURCE_FILES = ('predict.py', 'batting/__init__.py', 'batting/video.py', 'batting/tracking.py')
BASE_FILES = set(SOURCE_FILES) | {'README.md', 'LICENSE', 'THIRD_PARTY_NOTICE.txt',
    'verify_checksums.py', 'requirements.txt', 'requirements-export.txt',
    'package-manifest.json', 'evaluation-summary.json', 'validation.json',
    'clean-extraction-test.json', 'SHA256SUMS.txt', 'models/manifest.json',
    'models/body.pt', 'models/bat.pt', 'models/body.onnx', 'models/bat.onnx'}
PRIVATE_PATTERN = re.compile(r'/Users/|/home/|/private/|/mnt/|/tmp/|(?<![A-Za-z0-9])[A-Za-z]:[\\/]|gs://|s3://|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}')
RELEASE_ID = re.compile(r'20\d{2}-\d{2}-\d{2}(?:-r[1-9]\d*)?')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def exact_keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f'{label} must contain exactly: {sorted(keys)}')


def number(value, label, minimum=0, integer=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Invalid numeric {label}')
    if value < minimum or (integer and not isinstance(value, int)):
        raise ValueError(f'Invalid numeric {label}')
    return value


def validate_summary(summary):
    fields = ['baselineRelease', 'training', 'evaluation']
    if 'dataset' in summary:
        fields.append('dataset')
        dataset = summary['dataset']
        counts = ('addedCompletedAssignments', 'completedAssignments', 'submittedFrames')
        exact_keys(dataset, (*counts, 'body', 'bat'), 'dataset')
        for key in counts:
            number(dataset[key], key)
        if dataset['addedCompletedAssignments'] > dataset['completedAssignments']:
            raise ValueError('New completions exceed total submissions')
        for kind in ('body', 'bat'):
            task_fields = ('previousEligibleFrames', 'eligibleFrames', 'eligibleSwings')
            exact_keys(dataset[kind], task_fields, f'dataset.{kind}')
            for field in task_fields:
                number(dataset[kind][field], field)
            if dataset[kind]['eligibleFrames'] > dataset['submittedFrames']:
                raise ValueError('Eligible frame count exceeds submissions')
    exact_keys(summary, fields, 'summary')
    if not isinstance(summary['baselineRelease'], str) or not RELEASE_ID.fullmatch(summary['baselineRelease']):
        raise ValueError('Invalid baseline release ID')
    exact_keys(summary['training'], ('body', 'bat'), 'training')
    for kind, row in summary['training'].items():
        keys = ('epochs', 'imageSize', 'learningRate', 'seed', 'freeze',
                'trainFrames', 'trainVideos', 'valFrames', 'valVideos')
        exact_keys(row, keys, f'training.{kind}')
        for key in keys:
            number(row[key], key, integer=key != 'learningRate')
        if row['imageSize'] != (640 if kind == 'body' else 768) or row['epochs'] < 1:
            raise ValueError('Unexpected image size or no completed training epochs')
    evaluation = summary['evaluation']
    exact_keys(evaluation, ('frames', 'videos', 'sessions', 'bodyJoints', 'wrists', 'batPairs'), 'evaluation')
    for key in ('frames', 'videos', 'sessions'):
        number(evaluation[key], key, minimum=1)
    if evaluation['videos'] != 5:
        raise ValueError('Expected the fixed five-video benchmark')
    for metric in ('bodyJoints', 'wrists', 'batPairs'):
        exact_keys(evaluation[metric], ('baseline', 'candidate'), metric)
        for model, row in evaluation[metric].items():
            exact_keys(row, ('correct', 'total'), f'{metric}.{model}')
            number(row['correct'], 'correct')
            number(row['total'], 'total', minimum=1)
            if row['correct'] > row['total']:
                raise ValueError('Correct count exceeds denominator')
        if evaluation[metric]['baseline']['total'] != evaluation[metric]['candidate']['total']:
            raise ValueError('Baseline and candidate must use the same denominator')
    return deepcopy(summary)


def checked_file(path, expected):
    path = Path(path).resolve()
    if not re.fullmatch('[a-f0-9]{64}', expected) or not path.is_file() or sha256(path) != expected:
        raise ValueError('Frozen input file missing or checksum changed')
    return path


def read_spec(path):
    value = json.loads(Path(path).read_text())
    exact_keys(value, ('schemaVersion', 'release', 'models', 'parityRows', 'parityRowsSHA256', 'summary'), 'spec')
    if value['schemaVersion'] != 1 or not RELEASE_ID.fullmatch(value['release']):
        raise ValueError('Invalid package spec version/release')
    exact_keys(value['models'], ('body', 'bat'), 'models')
    validate_summary(value['summary'])
    for kind, row in value['models'].items():
        exact_keys(row, ('checkpoint', 'sha256', 'imageSize'), kind)
        checked_file(row['checkpoint'], row['sha256'])
        if row['imageSize'] != value['summary']['training'][kind]['imageSize']:
            raise ValueError('Model and training image sizes differ')
    rows_path = checked_file(value['parityRows'], value['parityRowsSHA256'])
    rows = json.loads(rows_path.read_text())
    if not isinstance(rows, list) or len(rows) != 15:
        raise ValueError('Exactly 15 private parity samples are required')
    groups = Counter(row['videoSHA256'] for row in rows)
    if len(groups) != 5 or set(groups.values()) != {3}:
        raise ValueError('Parity samples must span five videos, three each')
    if len({(row['videoSHA256'], row['frameIndex']) for row in rows}) != 15:
        raise ValueError('Duplicate parity samples')
    for row in rows:
        checked_file(row['imagePath'], row['imageSHA256'])
        number(row['frameIndex'], 'frameIndex')
    return value, rows


def reject_private_strings(value):
    if isinstance(value, str) and PRIVATE_PATTERN.search(value):
        raise ValueError('Private path, cloud URI, or email found in distributable metadata')
    if isinstance(value, dict):
        for key, child in value.items():
            reject_private_strings(key)
            reject_private_strings(child)
    if isinstance(value, (list, tuple)):
        for child in value:
            reject_private_strings(child)


def sanitize_checkpoint(source, target, kind, size, release):
    """Rebuild a plain pose module from an allowlisted architecture and exact state."""
    import torch
    import ultralytics
    from ultralytics.nn.tasks import PoseModel
    raw = torch.load(source, map_location='cpu', weights_only=False)
    original = raw.get('ema') or raw['model']
    if type(original) is not PoseModel:
        raise ValueError('Only standard Ultralytics PoseModel checkpoints are supported')
    config_keys = ('nc', 'kpt_shape', 'scales', 'backbone', 'head', 'scale', 'ch', 'channels')
    config = {key: deepcopy(original.yaml[key]) for key in config_keys if key in original.yaml}
    reject_private_strings(config)
    names = BODY_NAMES if kind == 'body' else ['tip', 'knob']
    shape = (len(names), 3)
    if tuple(config['kpt_shape']) != shape:
        raise ValueError('Unexpected checkpoint keypoint layout')
    with redirect_stdout(io.StringIO()):
        model = PoseModel(cfg=config, ch=3, nc=1, data_kpt_shape=shape, verbose=False)
    state = original.state_dict()
    model.load_state_dict({key:tensor.detach().clone() for key,tensor in state.items()}, strict=True, assign=True)
    if any(state[key].dtype != tensor.dtype or not torch.equal(state[key], tensor)
           for key, tensor in model.state_dict().items()):
        raise ValueError('Sanitized checkpoint changed a model tensor')
    model.names = {0: 'person' if kind == 'body' else 'bat'}
    model.args = {'task': 'pose', 'imgsz': size}
    model.kpt_names = {0: names}
    model.eval()
    torch.save({'model': model, 'ema': None, 'optimizer': None, 'epoch': -1,
                'train_args': model.args, 'version': ultralytics.__version__,
                'date': release, 'license': 'AGPL-3.0',
                'docs': 'https://docs.ultralytics.com'}, target)
    audit_checkpoint(target)
    return {'stateTensorsIdentical': True, 'tensorCount': len(state),
            'sourceCheckpointSHA256': sha256(source),
            'packagedCheckpointSHA256': sha256(target)}


def audit_checkpoint(path):
    with zipfile.ZipFile(path) as archive:
        pickle_names = [name for name in archive.namelist() if name.endswith('/data.pkl')]
        if len(pickle_names) != 1:
            raise ValueError('Unexpected checkpoint container')
        for _, value, _ in pickletools.genops(archive.read(pickle_names[0])):
            if isinstance(value, str):
                reject_private_strings(value)


def prepare_tensor(image, size):
    import cv2
    import numpy as np
    height, width = image.shape[:2]
    gain = min(size / width, size / height)
    new_width, new_height = round(width * gain), round(height * gain)
    left, top = round((size-new_width)/2-.1), round((size-new_height)/2-.1)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top+new_height, left:left+new_width] = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    gains = (new_width / width, new_height / height)
    return np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255, gains


def raw_model(path):
    from ultralytics import YOLO
    model = YOLO(str(path)).model.cpu().float().eval()
    model.fuse(verbose=False)
    model.model[-1].export = True
    model.model[-1].format = 'onnx'
    return model


def export_validate(source, checkpoint, kind, size, rows):
    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO
    exported = Path(YOLO(str(checkpoint)).export(format='onnx', imgsz=size, batch=1,
        dynamic=False, simplify=False, half=False, opset=17, nms=False, device='cpu'))
    if exported != checkpoint.with_suffix('.onnx'):
        raise ValueError('Exporter wrote an unexpected path')
    graph = onnx.load(exported)
    del graph.metadata_props[:]
    graph.doc_string = ''
    graph.graph.doc_string = ''
    for node in graph.graph.node:
        node.doc_string = ''
    for key, value in {'task':'pose', 'license':'AGPL-3.0', 'precision':'FP32',
                       'keypoints':json.dumps(BODY_NAMES if kind == 'body' else ['tip','knob'])}.items():
        item = graph.metadata_props.add()
        item.key, item.value = key, value
    if any(t.external_data for t in graph.graph.initializer):
        raise ValueError('External ONNX weights are not supported')
    if any(node.op_type == 'NonMaxSuppression' for node in graph.graph.node):
        raise ValueError('NMS must not be embedded')
    if {x.version for x in graph.opset_import if x.domain in ('', 'ai.onnx')} != {17}:
        raise ValueError('Unexpected ONNX opset')
    onnx.checker.check_model(graph)
    onnx.save(graph, exported)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(exported), options, providers=['CPUExecutionProvider'])
    channels, candidates = (56, 8400) if kind == 'body' else (11, 12096)
    input_shape, output_shape = [1,3,size,size], [1,channels,candidates]
    if session.get_inputs()[0].shape != input_shape or session.get_outputs()[0].shape != output_shape:
        raise ValueError('Unexpected ONNX tensor dimensions')
    original, packaged = raw_model(source), raw_model(checkpoint)
    pixel_errors, confidence_errors, raw_errors, source_errors = [], [], [], []
    for row in rows:
        path = checked_file(row['imagePath'], row['imageSHA256'])
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError('Cannot decode parity sample')
        tensor, gains = prepare_tensor(image, size)
        with torch.inference_mode():
            native = original(torch.from_numpy(tensor)).numpy()
            sanitized = packaged(torch.from_numpy(tensor)).numpy()
        portable = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        if any(list(x.shape) != output_shape or not np.isfinite(x).all() for x in (native, sanitized, portable)):
            raise ValueError('Invalid parity output')
        source_error = float(np.max(np.abs(native-sanitized)))
        if source_error > 1e-5:
            raise ValueError('Sanitized checkpoint changes native inference')
        selected = int(native[0,4].argmax())
        if selected != int(portable[0,4].argmax()):
            raise ValueError('Native/ONNX top candidate mismatch')
        delta = np.abs(native[0,:,selected]-portable[0,:,selected])
        xy = [i for joint in range((channels-5)//3) for i in (5+joint*3,6+joint*3)]
        conf = [4] + list(range(7, channels, 3))
        pixel_error = float((delta[xy].reshape(-1, 2) / np.asarray(gains)).max())
        confidence_error = float(delta[conf].max())
        if pixel_error >= .1 or confidence_error >= .0001:
            raise ValueError('Native/ONNX parity tolerance exceeded')
        pixel_errors.append(pixel_error)
        confidence_errors.append(confidence_error)
        raw_errors.append(float(np.abs(native-portable).max()))
        source_errors.append(source_error)
    return {'inputShape':input_shape, 'outputShape':output_shape,
            'inputName':session.get_inputs()[0].name, 'outputName':session.get_outputs()[0].name,
            'samples':len(rows), 'videos':5, 'topCandidateMismatches':0,
            'maxSourceVersusSanitizedRawError':max(source_errors),
            'maxSelectedSourcePixelError':max(pixel_errors),
            'maxSelectedConfidenceError':max(confidence_errors),
            'maxAllOutputAbsError':max(raw_errors),
            'opset':17, 'precision':'FP32', 'dynamic':False, 'nms':False}


def write_readme(stage, spec, versions):
    evaluation = spec['summary']['evaluation']
    baseline = spec['summary']['baselineRelease']
    lines = [f"| Metric | {baseline} | {spec['release']} | Change |", '|---|---:|---:|---:|']
    for key, label in [('bodyJoints','Body joint agreement'),('wrists','Wrist agreement'),('batPairs','Both BAT endpoints agree')]:
        before, after = (evaluation[key][model] for model in ('baseline','candidate'))
        a, b = before['correct']/before['total'], after['correct']/after['total']
        lines.append(f"| {label} | {a:.2%} ({before['correct']}/{before['total']}) | {b:.2%} ({after['correct']}/{after['total']}) | {100*(b-a):+.2f} pp |")
    allocation = ['| Model | Training frames / swings | Validation frames / swings | Evaluation frames / swings |', '|---|---:|---:|---:|']
    for kind in ('body','bat'):
        row = spec['summary']['training'][kind]
        allocation.append(f"| {kind.upper()} | {row['trainFrames']} / {row['trainVideos']} | {row['valFrames']} / {row['valVideos']} | {evaluation['frames']} / {evaluation['videos']} |")
    dataset_note = ''
    dataset = spec['summary'].get('dataset')
    if dataset:
        dataset_note = (f"The labeler recorded {dataset['addedCompletedAssignments']:,} newly completed swings since the baseline. "
                        f"The current export contains {dataset['completedAssignments']:,} completed submissions "
                        f"and {dataset['submittedFrames']:,} submitted frames.\n\n")
        dataset_note += '| Usable supervision | Eligible swings | Eligible frames | Change from baseline |\n|---|---:|---:|---:|\n'
        for task, label in [('body', 'Body pose'), ('bat', 'Bat tip and knob')]:
            counts = dataset[task]
            change = counts['eligibleFrames'] - counts['previousEligibleFrames']
            dataset_note += f"| {label} | {counts['eligibleSwings']:,} | {counts['eligibleFrames']:,} | {change:+,} |\n"
    values = {'release':spec['release'], 'baseline':baseline, 'results_table':'\n'.join(lines),
              'dataset_note':dataset_note, 'allocation_table':'\n'.join(allocation), 'python_version':platform.python_version(),
              'platform_name':platform.system()+' '+platform.machine(),
              'body_size':640, 'bat_size':768, 'body_candidates':8400, 'bat_candidates':12096}
    (stage/'README.md').write_text(Template((TEMPLATES/'README.md').read_text()).substitute(values))


def write_checksums(stage):
    lines = []
    for path in sorted(stage.rglob('*')):
        if path.is_file() and path.name != 'SHA256SUMS.txt':
            lines.append(f'{sha256(path)}  {path.relative_to(stage).as_posix()}')
    (stage/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n')


def audit_package(stage, complete=True):
    files = {path.relative_to(stage).as_posix() for path in stage.rglob('*') if path.is_file()}
    expected = BASE_FILES if complete else BASE_FILES - {'clean-extraction-test.json'}
    if files != expected:
        raise ValueError(f'Unexpected package inventory: {sorted(files ^ expected)}')
    for path in stage.rglob('*'):
        if path.is_symlink():
            raise ValueError('Symlinks must not enter a package')
        if path.is_file() and path.suffix in ('.json', '.md', '.txt', '.py'):
            reject_private_strings(path.read_text())
    for kind in ('body','bat'):
        audit_checkpoint(stage/f'models/{kind}.pt')


def zip_package(stage, destination):
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                archive.write(path, arcname=(Path(stage.name)/path.relative_to(stage)).as_posix())


def safe_extract(archive_path, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            target = (destination/info.filename).resolve()
            if not target.is_relative_to(destination) or info.filename.startswith('/'):
                raise ValueError('Unsafe ZIP member')
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('ZIP symlink is not allowed')
        archive.extractall(destination)


def run_clean(command, cwd):
    environment = {key:value for key,value in os.environ.items() if key not in ('PYTHONPATH','PYTHONHOME')}
    environment.update({'PYTHONNOUSERSITE':'1', 'PYTHONDONTWRITEBYTECODE':'1',
                        'YOLO_OFFLINE':'true', 'YOLO_AUTOINSTALL':'false',
                        'YOLO_CONFIG_DIR':str(cwd/'.cache/ultralytics'),
                        'MPLCONFIGDIR':str(cwd/'.cache/matplotlib')})
    result = subprocess.run([sys.executable,'-E','-s',*command], cwd=cwd, env=environment,
                            capture_output=True, text=True, timeout=300)
    if result.returncode:
        # The captured exception may include a private source path: keep it out of distributable files.
        raise RuntimeError('Clean extracted-package command failed; inspect the local process log')
    return result


def clean_smoke(stage, rows, work):
    import av
    import cv2
    trial = work/'trial.zip'
    zip_package(stage, trial)
    extraction = work/'extracted'
    safe_extract(trial, extraction)
    extracted = extraction/stage.name
    run_clean(['verify_checksums.py'], extracted)
    run_clean(['predict.py','--help'], extracted)
    video = work/'private-smoke.mp4'
    times = [0, 3, 10]
    with av.open(str(video),'w') as output:
        stream = output.add_stream('libx264',rate=60)
        image = cv2.imread(rows[0]['imagePath'])
        height,width = image.shape[:2]
        stream.width,stream.height = width+width%2,height+height%2
        stream.pix_fmt='yuv420p'
        stream.time_base=Fraction(1,60)
        stream.codec_context.time_base=Fraction(1,60)
        stream.options={'crf':'18','preset':'fast'}
        for timestamp in times:
            frame = av.VideoFrame.from_ndarray(cv2.copyMakeBorder(image,0,height%2,0,width%2,cv2.BORDER_CONSTANT),format='bgr24')
            frame.pts=timestamp
            frame.time_base=Fraction(1,60)
            for packet in stream.encode(frame): output.mux(packet)
        for packet in stream.encode(): output.mux(packet)
    prediction,overlay=work/'private-predictions.jsonl',work/'private-overlay.mp4'
    run_clean(['predict.py',str(video),'--output',str(prediction),'--overlay',str(overlay),'--device','cpu','--threads','4'],extracted)
    records=[json.loads(line) for line in prediction.read_text().splitlines()]
    frames=[row for row in records if row['type']=='frame']
    if len(frames)!=3 or records[-1].get('complete') is not True:
        raise ValueError('Clean runner did not produce complete three-frame output')
    with av.open(str(video)) as container:
        source_times=[frame.pts*frame.time_base for frame in container.decode(video=0)]
    with av.open(str(overlay)) as container:
        output_times=[frame.pts*frame.time_base for frame in container.decode(video=0)]
    if output_times != [value-source_times[0] for value in source_times]:
        raise ValueError('Clean runner overlay changed presentation timing')
    for record,time in zip(frames,source_times):
        if Fraction(record['pts'])*Fraction(record['timeBase'])!=time:
            raise ValueError('Clean runner JSON changed presentation timing')
    manifest=json.loads((stage/'models/manifest.json').read_text())
    for kind in ('body','bat'):
        if records[0]['models'][kind]['sha256']!=manifest['defaults'][kind]['sha256']:
            raise ValueError('Clean runner used the wrong checkpoint')
    return {'complete':True,'platform':platform.system(),'architecture':platform.machine(),
            'python':platform.python_version(),'device':'cpu','frames':3,
            'allOverlayFramesDecoded':True,'allJSONTimestampsVerified':True,
            'allOverlayTimestampsVerified':True,'irregularSourceTimestamps':True,
            'packageChecksumsPassed':True,'usedPackageLocalModels':True,
            'historicalRepositoryOnPythonPath':False,
            'runtimeDependencies':'Installed pinned dependencies; no fresh network installation tested',
            'testedFiles':{name:sha256(stage/name) for name in (*SOURCE_FILES,'models/body.pt','models/bat.pt','models/manifest.json')}}


def build(spec_path, output_root):
    spec,rows=read_spec(spec_path)
    versions={name:importlib.metadata.version(name) for name in (*RUNTIME_PACKAGES,*EXPORT_PACKAGES)}
    import torch
    torch.set_num_threads(4)
    os.environ['YOLO_OFFLINE']='true'
    os.environ['YOLO_AUTOINSTALL']='false'
    output_root=Path(output_root).resolve()
    # Generated packages must stay under the repository's ignored outputs directory.
    if not output_root.is_relative_to(ROOT/'outputs'):
        raise ValueError('Use an ignored output directory under this repository outputs/')
    output_root.mkdir(parents=True,exist_ok=True)
    name=f"fan-lab-2d-models-{spec['release']}"
    final_zip,final_dir=output_root/f'{name}.zip',output_root/name
    if final_zip.exists() or final_dir.exists():
        raise FileExistsError('Release already exists; do not overwrite a distributed package')
    with tempfile.TemporaryDirectory(prefix='.package-build-',dir=output_root) as temporary:
        work=Path(temporary)
        stage=work/name
        (stage/'models').mkdir(parents=True)
        os.environ['YOLO_CONFIG_DIR']=str(work/'ultralytics-cache')
        os.environ['MPLCONFIGDIR']=str(work/'matplotlib-cache')
        defaults,validation={},{}
        for kind in ('body','bat'):
            print(f'Preparing {kind} checkpoint and validating export',flush=True)
            row=spec['models'][kind]
            source=checked_file(row['checkpoint'],row['sha256'])
            checkpoint=stage/f'models/{kind}.pt'
            sanitization=sanitize_checkpoint(source,checkpoint,kind,row['imageSize'],spec['release'])
            parity=export_validate(source,checkpoint,kind,row['imageSize'],rows)
            validation[kind]={'checkpointSanitization':sanitization,**parity}
            defaults[kind]={'checkpoint':f'models/{kind}.pt','sha256':sha256(checkpoint),
                            'onnx':f'models/{kind}.onnx','onnxSHA256':sha256(checkpoint.with_suffix('.onnx')),
                            'inputShape':parity['inputShape'],'outputShape':parity['outputShape'],
                            'inputName':parity['inputName'],'outputName':parity['outputName'],
                            'keypointOrder':BODY_NAMES if kind=='body' else ['tip','knob']}
        write_json(stage/'models/manifest.json',{'schemaVersion':1,'release':spec['release'],'defaults':defaults})
        write_json(stage/'evaluation-summary.json',validate_summary(spec['summary']))
        write_json(stage/'validation.json',{'scope':'Numerical export parity, not pose accuracy',
                   'versions':versions,'sampleSelection':'Three frozen private samples per evaluation video; five videos',
                   'models':validation})
        write_json(stage/'package-manifest.json',{'schemaVersion':1,'release':spec['release'],
            'sourceRepository':'https://github.com/jramsdell-nym/video-to-2d-skele',
            'sourceCheckpointSHA256':{kind:row['sha256'] for kind,row in spec['models'].items()},
            'generatorSHA256':sha256(Path(__file__)),'sourceFilesSHA256':{name:sha256(ROOT/name) for name in SOURCE_FILES},
            'modelDefaultPromotion':False,'privateFootageIncluded':False,'labelsIncluded':False,
            'metadataSanitized':True,'summaryBaseline':spec['summary']['baselineRelease'],'reusedDevelopmentBenchmark':True})
        for filename in SOURCE_FILES:
            target=stage/filename
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/filename,target)
        for filename in ('verify_checksums.py','THIRD_PARTY_NOTICE.txt'):
            shutil.copyfile(TEMPLATES/filename,stage/filename)
        shutil.copyfile(ROOT/'licenses/ULTRALYTICS-AGPL-3.0.txt',stage/'LICENSE')
        (stage/'requirements.txt').write_text('\n'.join(f'{name}=={versions[name]}' for name in RUNTIME_PACKAGES)+'\n')
        (stage/'requirements-export.txt').write_text('-r requirements.txt\n'+'\n'.join(f'{name}=={versions[name]}' for name in EXPORT_PACKAGES)+'\n')
        write_readme(stage,spec,versions)
        write_checksums(stage)
        audit_package(stage,complete=False)
        write_json(stage/'clean-extraction-test.json',clean_smoke(stage,rows,work))
        write_checksums(stage)
        audit_package(stage)
        final_staged_zip=work/f'{name}.zip'
        zip_package(stage,final_staged_zip)
        final_extraction=work/'final-extraction'
        safe_extract(final_staged_zip,final_extraction)
        run_clean(['verify_checksums.py'],final_extraction/name)
        run_clean(['predict.py','--help'],final_extraction/name)
        # Recheck every frozen input, including source models, before publishing output.
        read_spec(spec_path)
        shutil.move(stage,final_dir)
        shutil.move(final_staged_zip,final_zip)
    receipt={'zip':str(final_zip),'sha256':sha256(final_zip),'bytes':final_zip.stat().st_size,
             'packageDirectory':str(final_dir),'complete':True,'models':list(defaults)}
    write_json(output_root/f'{name}.receipt.json',receipt)
    (output_root/f'{name}.zip.sha256').write_text(f"{receipt['sha256']}  {final_zip.name}\n")
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.spec,args.output_root),indent=2))


if __name__=='__main__':
    main()
