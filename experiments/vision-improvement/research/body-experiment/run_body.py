"""Isolated body-pose development run. Prepare/validate never start training.

Pinned source images and labels are durable local copies; cache/output are local.
Revalidate every snapshot image and label hash before training or inference.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parent
SOURCE = Path("/tmp/bat-labeler-three-swing-training/dataset")
WEIGHTS = Path("/tmp/bat-labeler-pose-model/yolo11n-pose.pt")
EXPECTED_BASE_SHA = "869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runtime-config"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "matplotlib"))
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"
from score_body import JOINTS, finite, key


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def prepare(run):
    from PIL import Image
    import torch
    if run.exists():
        raise ValueError("Refusing to overwrite existing experiment")
    raw = json.loads((SOURCE/"annotations.json").read_text())
    if raw.get("schema") != 2 or raw.get("joints") != JOINTS or raw.get("sourceJointIndices") != list(range(5,17)):
        raise ValueError("Unexpected twelve-point export schema")
    if sha(WEIGHTS) != EXPECTED_BASE_SHA:
        raise ValueError("Generic base checkpoint bytes changed")
    checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    if checkpoint["model"].model[-1].kpt_shape != [17,3] or not checkpoint.get("train_args",{}).get("data", "").endswith("coco-pose.yaml"):
        raise ValueError("Require original generic seventeen-point COCO weights")
    del checkpoint
    rows = raw["records"]
    if len(set(map(key, rows))) != len(rows):
        raise ValueError("Duplicate reference identities")
    if any(r.get("status") != "reviewed" or r.get("split") not in ("train","val") or len(r["points"]) != 12 for r in rows):
        raise ValueError("Expected reviewed twelve-point train/validation records")
    split = {s:[r for r in rows if r["split"]==s] for s in ("train","val")}
    if any(not values for values in split.values()):
        raise ValueError("Both splits must be nonempty")
    for identity in (lambda r:r["session"], lambda r:r["videoSHA256"], lambda r:(r["session"],r["trial"])):
        if {identity(r) for r in split["train"]} & {identity(r) for r in split["val"]}:
            raise ValueError("Session/video/canonical-trial leakage")
    run.mkdir(parents=True)
    shutil.copyfile(WEIGHTS, run/"generic-yolo11n-pose.pt")
    pinned, safe_rows = [], []
    previous_times = {}
    keep = ("schema","videoSHA256","frameIndex","frame","width","height","session","trial","camera","source","points","box","status","updatedAt","reviewedAt","reviewMethod","split","image")
    for r in sorted(rows, key=key):
        source_image = (SOURCE/r["image"]).resolve()
        if not source_image.is_relative_to(SOURCE.resolve()) or not source_image.is_file():
            raise ValueError("Missing or escaped source image")
        if r["width"] <= 0 or r["height"] <= 0 or not isinstance(r["frameIndex"], int):
            raise ValueError("Invalid image dimensions/frame index")
        t = float(Fraction(r["frame"]["pts"])*Fraction(r["frame"]["timeBase"]))
        if abs(t-r["frame"]["time"]) > 2e-6 or r["frame"]["index"] != r["frameIndex"]:
            raise ValueError("Inconsistent source timestamp identity")
        if t <= previous_times.get(r["videoSHA256"], -math.inf):
            raise ValueError("Nonincreasing presentation timestamps")
        previous_times[r["videoSHA256"]] = t
        for p in r["points"]:
            if p.get("visibility") not in (0,1,2) or (p["visibility"] and not all(finite(p.get(a)) and 0 <= p[a] <= 1 for a in ("x","y"))):
                raise ValueError("Invalid point supervision")
        if not any(p["visibility"] for p in r["points"]):
            raise ValueError("No localized body supervision; do not invent negative")
        box = r["box"]
        if len(box)!=4 or not all(finite(x) and 0<=x<=1 for x in box) or box[0]>=box[2] or box[1]>=box[3]:
            raise ValueError("Invalid person box")
        expected = [0,(box[0]+box[2])/2,(box[1]+box[3])/2,box[2]-box[0],box[3]-box[1]]
        for p in r["points"]:
            expected.extend([p["x"],p["y"],p["visibility"]] if p["visibility"] else [0,0,0])
        source_label = SOURCE/"labels"/r["split"]/(source_image.stem+".txt")
        actual = [float(x) for x in source_label.read_text().split()]
        if len(actual)!=41 or any(not finite(a) or abs(a-b)>1e-6 for a,b in zip(actual,expected)):
            raise ValueError("Exported label disagrees with reviewed coordinates")
        with Image.open(source_image) as image:
            image.load()
            if image.size != (r["width"],r["height"]):
                raise ValueError("Decoded image dimensions mismatch")
        image = run/"dataset"/r["image"]
        label = run/"dataset"/"labels"/r["split"]/(source_image.stem+".txt")
        image.parent.mkdir(parents=True,exist_ok=True)
        label.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source_image,image)
        shutil.copyfile(source_label,label)
        pinned.extend([{"path":str(image),"sha256":sha(image),"role":"durable byte-identical PNG snapshot"},
                       {"path":str(label),"sha256":sha(label),"role":"local label snapshot"}])
        safe = {k:r[k] for k in keep if k in r}
        safe["originalPredictionModel"] = r.get("prediction",{}).get("model")
        safe_rows.append(safe)
    for s in ("train","val"):
        subset = [r for r in safe_rows if r["split"]==s]
        save(run/f"{s}-rows.json",subset)
        (run/"dataset"/f"{s}.txt").write_text("".join(str(run/"dataset"/r["image"])+"\n" for r in subset))
    yaml = f"path: {run/'dataset'}\ntrain: train.txt\nval: val.txt\nnames:\n  0: person\nkpt_shape: [12, 3]\nflip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]\n"
    (run/"dataset"/"data.yaml").write_text(yaml)
    for path in [run/"generic-yolo11n-pose.pt",run/"train-rows.json",run/"val-rows.json",run/"dataset"/"data.yaml",run/"dataset"/"train.txt",run/"dataset"/"val.txt"]:
        pinned.append({"path":str(path),"sha256":sha(path),"role":"local frozen input"})
    save(run/"inputs.json",pinned)
    save(run/"manifest.json",{
        "createdAt":datetime.now(timezone.utc).isoformat(),"runnerSHA256":sha(__file__),
        "sourceAnnotations":str(SOURCE/"annotations.json"),"sourceAnnotationsSHA256":sha(SOURCE/"annotations.json"),
        "sourceWeightsSHA256":EXPECTED_BASE_SHA,"sourceWeightsMeaning":"Original generic YOLO11n COCO-17 checkpoint, 2024-09-25; no previous body-specific fit",
        "inputManifestSHA256":sha(run/"inputs.json"),"jointOrder":JOINTS,"genericCOCOSlots":list(range(5,17)),
        "splits":{s:{"frames":len(v),"sessions":sorted({r['session'] for r in v}),"videos":sorted({r['videoSHA256'] for r in v})} for s,v in split.items()},
        "reviewMethods":dict(Counter(r.get("reviewMethod","legacy_or_missing") for r in rows)),
        "pointSources":dict(Counter(p.get("source","unspecified") for r in rows for p in r['points'])),
        "selection":"Fixed 40-epoch candidate last.pt; no validation-based checkpoint selection or all-data refit",
        "candidatePolicy":"Top model confidence person at box>=0.25, no reference-box matching; keep up to ten candidates. Square640, keypoint>=0.3 for primary; threshold0 diagnostic also retained.",
        "limitations":["Two sessions, three swings, athlete IDs incomplete: cannot claim player-held-out accuracy.","Most reference joints interpolated; frame review does not imply manual joint localization.","This is development data, not a final qualification test.","Ultralytics clips predicted points to source bounds; this is not native Vision/iPhone parity.","No negatives, occluded targets or 3D truth in this snapshot."]})
    print(json.dumps({"run":str(run),"splits":{s:len(v) for s,v in split.items()},"pinnedFiles":len(pinned)}),flush=True)


def validate(run):
    manifest=json.loads((run/"manifest.json").read_text())
    if sha(run/"inputs.json") != manifest["inputManifestSHA256"]:
        raise ValueError("Input hash manifest changed")
    for f in json.loads((run/"inputs.json").read_text()):
        if sha(f["path"]) != f["sha256"]:
            raise ValueError(f"Frozen input changed: {f['path']}")
    print(json.dumps({"validated":str(run),"inputFiles":len(json.loads((run/'inputs.json').read_text()))}),flush=True)
    return manifest


def runtime():
    import torch,ultralytics
    from ultralytics.utils import SETTINGS
    torch.set_num_threads(4)
    SETTINGS.update({k:False for k in ("sync","wandb","mlflow","comet","clearml","hub","tensorboard","neptune","dvc") if k in SETTINGS})
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable: coordinate GPU launch with the training owner")
    return {"torch":torch.__version__,"ultralytics":ultralytics.__version__,"device":"mps"}


def train(run):
    from ultralytics import YOLO
    validate(run)
    env=runtime()
    if (run/"candidate").exists():
        raise ValueError("Candidate output already exists")
    params=dict(data=str(run/"dataset"/"data.yaml"),epochs=40,imgsz=640,batch=8,workers=0,
                device="mps",seed=42,deterministic=True,optimizer="AdamW",lr0=.0008,lrf=.05,
                weight_decay=.0005,warmup_epochs=3,nbs=8,freeze=0,patience=0,amp=False,
                degrees=10,translate=.1,scale=.25,fliplr=.5,flipud=0,mosaic=0,mixup=0,
                hsv_h=.01,hsv_s=.25,hsv_v=.25,plots=False,save=True,verbose=False,
                project=str(run),name="candidate",exist_ok=False)
    save(run/"training-config.json",{"runtime":env,"params":params,"runnerSHA256":sha(__file__)})
    started=time.monotonic()
    model=YOLO(str(run/"generic-yolo11n-pose.pt"))
    model.add_callback("on_train_epoch_end",lambda trainer:save(run/"training-status.json",{"state":"training","epoch":trainer.epoch+1,"epochs":40,"elapsedSeconds":time.monotonic()-started}))
    model.train(**params)
    path=run/"candidate"/"weights"/"last.pt"
    save(run/"training-status.json",{"state":"complete","epochs":40,"seconds":time.monotonic()-started,"checkpoint":str(path),"checkpointSHA256":sha(path)})


@contextmanager
def inference_warning_guard(logger, records):
    """Preserve warnings and fail rather than score a time-limited NMS result."""
    class Guard(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                message = record.getMessage()
                records.append(message)
                if "nms" in message.lower():
                    raise RuntimeError(f"Inference invalidated by NMS warning: {message}")
    handler = Guard(level=logging.WARNING)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)


def predict_one(model, image):
    """A scalar path avoids Ultralytics treating a path list as one large batch."""
    results = iter(model.predict(source=str(image),imgsz=640,rect=False,conf=.25,
                                 max_det=10,batch=1,device="mps",stream=True,verbose=False))
    try:
        result = next(results)
    except StopIteration:
        raise ValueError(f"No prediction result for {image}") from None
    try:
        next(results)
    except StopIteration:
        pass
    else:
        raise ValueError(f"Multiple prediction results for a single source image: {image}")
    if Path(result.path).resolve() != Path(image).resolve():
        raise ValueError(f"Prediction identity mismatch for {image}: {result.path}")
    return result


def predict(run,which):
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    from score_body import score
    validate(run)
    env=runtime()
    checkpoint=run/"generic-yolo11n-pose.pt" if which=="generic" else run/"candidate"/"weights"/"last.pt"
    rows=json.loads((run/"val-rows.json").read_text())
    images=[str(run/"dataset"/r["image"]) for r in rows]
    if (run/f"{which}-val-predictions.json").exists():
        raise ValueError("Prediction output already exists; preserve completed inference")
    model=YOLO(str(checkpoint))
    expected=17 if which=="generic" else 12
    if model.model.model[-1].kpt_shape != [expected,3]:
        raise ValueError("Unexpected prediction landmark head")
    out=[]
    warnings=[]
    started=time.monotonic()
    try:
        with inference_warning_guard(LOGGER,warnings):
            for r,image in zip(rows,images,strict=True):
                result=predict_one(model,image)
                item={"videoSHA256":r["videoSHA256"],"frameIndex":r["frameIndex"],"detected":bool(len(result.boxes)),"candidates":[]}
                if len(result.boxes):
                    points=result.keypoints.xy.cpu().tolist()
                    confs=result.keypoints.conf.cpu().tolist()
                    scores=result.boxes.conf.cpu().tolist()
                    boxes=result.boxes.xyxy.cpu().tolist()
                    chosen=max(range(len(scores)),key=lambda i:scores[i])
                    for i in range(len(scores)):
                        item["candidates"].append({"score":scores[i],"box":boxes[i],"prediction":points[i][5:17] if expected==17 else points[i],"keypointConfidences":confs[i][5:17] if expected==17 else confs[i]})
                    item.update(item["candidates"][chosen],selectedCandidateIndex=chosen)
                out.append(item)
    except Exception as exc:
        save(run/f"{which}-prediction-status.json",{"state":"failed","completedFrames":len(out),"expectedFrames":len(rows),"warnings":warnings,"error":str(exc),"runnerSHA256":sha(__file__)})
        raise
    if len(out)!=len(rows):
        raise ValueError("Incomplete prediction stream")
    save(run/f"{which}-val-predictions.json",out)
    save(run/f"{which}-prediction-manifest.json",{"checkpointSHA256":sha(checkpoint),"runtime":env,"frames":len(out),"expectedFrames":len(rows),"secondsIncludingWarmup":time.monotonic()-started,"imgsz":640,"rect":False,"boxThreshold":.25,"sourceBatchPolicy":"one scalar source path per predict call; exactly one matching result required","batch":1,"warnings":warnings,"nmsWarningPolicy":"fail inference; no partial prediction output","runnerSHA256":sha(__file__),"scorerSHA256":sha(Path(__file__).with_name("score_body.py")),"selection":"maximum-confidence model candidate; no ground-truth selection","genericCOCOSlots":list(range(5,17)) if expected==17 else None})
    save(run/f"{which}-prediction-status.json",{"state":"complete","frames":len(out),"expectedFrames":len(rows),"warnings":warnings,"runnerSHA256":sha(__file__)})
    for threshold,suffix in ((.3,"primary"),(0,"all-scores")):
        result=score(rows,out,threshold)
        save(run/f"{which}-{suffix}-score.json",result)
        print(json.dumps({"model":which,"policy":suffix,"jointPCK0.05BodyScale":result['jointPCK0.05BodyScale'],"macroVideoJointPCK0.05BodyScale":result['macroVideoJointPCK0.05BodyScale'],"wristPCK0.05BodyScale":result['wrists']['jointPCK0.05BodyScale']}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","validate","train","predict"))
    p.add_argument("--run",type=Path,default=ROOT/"session-holdout-20261111")
    p.add_argument("--model",choices=("generic","candidate"),default="generic")
    a=p.parse_args(); run=a.run.resolve()
    if a.stage=="prepare":prepare(run)
    elif a.stage=="validate":validate(run)
    elif a.stage=="train":train(run)
    else:predict(run,a.model)
