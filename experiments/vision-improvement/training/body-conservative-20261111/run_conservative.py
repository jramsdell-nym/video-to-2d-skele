"""One bounded 17-head conservative body-pose development experiment."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1] / "research" / "body-experiment" / "session-holdout-20261111"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runtime-config"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "matplotlib"))
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"
sys.path.insert(0, str(SOURCE / "execution-source"))
from score_body import score

FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
BASE_SHA = "869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def prepare():
    if (ROOT / "manifest.json").exists():
        raise RuntimeError("Refusing to replace frozen inputs")
    import torch
    from ultralytics.utils.loss import KeypointLoss
    from ultralytics.utils.metrics import OKS_SIGMA
    old_manifest = json.loads((SOURCE / "manifest.json").read_text())
    assert sha(SOURCE / "inputs.json") == old_manifest["inputManifestSHA256"]
    for item in json.loads((SOURCE / "inputs.json").read_text()):
        assert sha(item["path"]) == item["sha256"]
    assert sha(SOURCE / "generic-yolo11n-pose.pt") == BASE_SHA
    shutil.copyfile(SOURCE / "generic-yolo11n-pose.pt", ROOT / "generic-yolo11n-pose.pt")
    files = []
    rows = {}
    for split in ("train", "val"):
        rows[split] = json.loads((SOURCE / f"{split}-rows.json").read_text())
        save(ROOT / f"{split}-rows.json", rows[split])
        images = []
        for row in rows[split]:
            source = SOURCE / "dataset" / row["image"]
            destination = ROOT / "dataset" / row["image"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                assert destination.resolve() == source.resolve()
            else:
                destination.symlink_to(source.resolve())
            images.append(str(destination))
            box = row["box"]
            values = [0, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]]
            values += [0, 0, 0] * 5
            for point in row["points"]:
                values += [point["x"], point["y"], point["visibility"]] if point["visibility"] else [0, 0, 0]
            assert len(values) == 56 and values[5:20] == [0] * 15
            label = ROOT / "dataset" / "labels" / split / (source.stem + ".txt")
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(" ".join(map(str, values)) + "\n")
            files.extend([{"path": str(destination), "sha256": sha(destination)}, {"path": str(label), "sha256": sha(label)}])
        (ROOT / "dataset" / f"{split}.txt").write_text("\n".join(images) + "\n")
    for field in ("session", "videoSHA256"):
        assert not {r[field] for r in rows["train"]} & {r[field] for r in rows["val"]}
    assert len(rows["train"]) == 120 and len(rows["val"]) == 240
    text = f"path: {ROOT / 'dataset'}\ntrain: train.txt\nval: val.txt\nnames:\n  0: person\nkpt_shape: [17, 3]\nflip_idx: {FLIP}\n"
    (ROOT / "dataset" / "data.yaml").write_text(text)
    for p in (ROOT / "generic-yolo11n-pose.pt", ROOT / "train-rows.json", ROOT / "val-rows.json", ROOT / "dataset" / "data.yaml", ROOT / "dataset" / "train.txt", ROOT / "dataset" / "val.txt"):
        files.append({"path": str(p), "sha256": sha(p)})
    save(ROOT / "inputs.json", files)
    # Direct differentiable check of the installed loss: v0 facial coordinates
    # receive exactly zero coordinate gradient, with nonzero body gradient.
    pred = torch.ones((1, 17, 3), requires_grad=True)
    truth = torch.zeros((1, 17, 3))
    mask = torch.tensor([[False] * 5 + [True] * 12])
    loss = KeypointLoss(torch.tensor(OKS_SIGMA))(pred, truth, mask, torch.tensor([[10000.0]]))
    loss.backward()
    assert torch.count_nonzero(pred.grad[:, :5, :2]) == 0
    assert torch.count_nonzero(pred.grad[:, 5:, :2]) > 0
    save(ROOT / "manifest.json", {"createdAt": datetime.now(timezone.utc).isoformat(), "sourceExperiment": str(SOURCE), "sourceWeightsSHA256": BASE_SHA,
        "inputManifestSHA256": sha(ROOT / "inputs.json"), "runnerSHA256": sha(__file__), "scorerSHA256": sha(SOURCE / "execution-source" / "score_body.py"),
        "trainFrames": 120, "developmentFrames": 240, "keypointShape": [17, 3], "bodySlots": list(range(5, 17)), "flipIndex": FLIP,
        "facialTargets": "Five [0,0,0] triplets: no invented face coordinates. v0 exactly masks coordinate loss; visibility BCE does receive negative targets and can lower unused facial confidence.",
        "coordinateMaskGradientTest": "passed: zero facial coordinate gradients, nonzero body gradients", "sigmas": [float(s) for s in OKS_SIGMA],
        "selection": "Predeclared twenty epochs, fixed last.pt; no selection or tuning from development metrics. Already-inspected development set, not a fresh test.",
        "comparison": "Conservative recipe bundle: 17-head preservation, original COCO sigmas, freeze10, lower learning/warmup rates and twenty epochs differ from the rejected12-head experiment. Not a causal head-count ablation.",
        "limits": ["Two sessions, three swings; mostly interpolated/model-assisted reference labels.", "MPS scatter/index operations can be nondeterministic despite seed42.", "Offline PyTorch top-confidence-person evaluation; not a full iPhone/CoreML policy replay."]})
    print("Prepared120 training and240 development frames with17 slots; facial coordinate-mask test passed", flush=True)


def validate():
    m = json.loads((ROOT / "manifest.json").read_text())
    assert sha(ROOT / "inputs.json") == m["inputManifestSHA256"]
    for item in json.loads((ROOT / "inputs.json").read_text()):
        assert sha(item["path"]) == item["sha256"]
    assert sha(SOURCE / "execution-source" / "score_body.py") == m["scorerSHA256"]
    assert sha(__file__) == m["runnerSHA256"], "Runner changed after inputs were frozen"
    return m


def runtime():
    import torch
    from ultralytics.utils import SETTINGS
    torch.set_num_threads(4)
    assert torch.backends.mps.is_available(), "MPS unavailable; do not silently run onCPU"
    SETTINGS.update({k: False for k in ("sync", "wandb", "mlflow", "comet", "clearml", "hub", "tensorboard", "neptune", "dvc") if k in SETTINGS})
    return torch


def train():
    from ultralytics import YOLO
    m = validate()
    torch = runtime()
    assert not (ROOT / "candidate").exists()
    model = YOLO(str(ROOT / "generic-yolo11n-pose.pt"))
    assert model.model.model[-1].kpt_shape == [17, 3]
    initial = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
    params = dict(data=str(ROOT / "dataset" / "data.yaml"), epochs=20, imgsz=640, batch=8, workers=0, device="mps", seed=42, deterministic=True,
        optimizer="AdamW", lr0=.0001, lrf=.1, weight_decay=.0005, warmup_epochs=1, warmup_bias_lr=.0001, nbs=8, freeze=10, patience=0, amp=False,
        degrees=10, translate=.1, scale=.25, fliplr=.5, flipud=0, mosaic=0, mixup=0, hsv_h=.01, hsv_s=.25, hsv_v=.25,
        val=False, plots=False, save=True, verbose=False, project=str(ROOT), name="candidate", exist_ok=False)
    save(ROOT / "training-config.json", {"params": params, "runnerSHA256": sha(__file__), "sourceWeightsSHA256": m["sourceWeightsSHA256"], "warmupNote": "Installed trainer rounds warmup_epochs*num_batches: 15 iterations for one epoch here."})
    def verify_loaded(trainer):
        loaded = trainer.model.state_dict()
        mismatched = [k for k, v in initial.items() if k not in loaded or not torch.equal(v, loaded[k].detach().cpu())]
        assert not mismatched, f"Pretrained tensors changed before fitting: {mismatched}"
        assert trainer.model.model[-1].kpt_shape == [17, 3]
        save(ROOT / "initialization-verification.json", {"allPretrainedTensorsIdenticalBeforeFit": True, "tensorsCompared": len(initial), "poseHeadTensorsCompared": sum('.cv4.' in k for k in initial), "sourceWeightsSHA256": BASE_SHA, "keypointShape": [17, 3], "extraStateKeys": sorted(set(loaded) - set(initial))})
    start = time.monotonic()
    model.add_callback("on_pretrain_routine_end", verify_loaded)
    model.add_callback("on_train_epoch_end", lambda t: save(ROOT / "training-status.json", {"state": "training", "epoch": t.epoch + 1, "epochs": 20, "elapsedSeconds": time.monotonic() - start}))
    model.train(**params)
    ckpt = ROOT / "candidate" / "weights" / "last.pt"
    save(ROOT / "training-status.json", {"state": "complete", "epochs": 20, "seconds": time.monotonic() - start, "checkpoint": str(ckpt), "checkpointSHA256": sha(ckpt)})


def predict():
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    validate()
    assert not (ROOT / "candidate-val-predictions.json").exists(), "Refusing to overwrite accepted predictions"
    runtime()
    class AbortNMS(logging.Handler):
        def emit(self, record):
            if "NMS" in record.getMessage() and "warning" in record.levelname.lower():
                raise RuntimeError(record.getMessage())
    LOGGER.addHandler(AbortNMS())
    ckpt = ROOT / "candidate" / "weights" / "last.pt"
    model = YOLO(str(ckpt))
    assert model.model.model[-1].kpt_shape == [17, 3]
    rows = json.loads((ROOT / "val-rows.json").read_text())
    predictions = []
    started = time.monotonic()
    for row in rows:
        path = ROOT / "dataset" / row["image"]
        results = list(model.predict(str(path), imgsz=640, rect=False, conf=.25, max_det=10, batch=1, device="mps", stream=True, verbose=False))
        assert len(results) == 1 and Path(results[0].path).resolve() == path.resolve()
        result = results[0]
        item = {"videoSHA256": row["videoSHA256"], "frameIndex": row["frameIndex"], "imagePath": str(path), "detected": bool(len(result.boxes)), "candidates": []}
        if item["detected"]:
            xy, conf, boxes, scores = result.keypoints.xy.cpu().tolist(), result.keypoints.conf.cpu().tolist(), result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()
            for i in range(len(scores)):
                item["candidates"].append({"score": scores[i], "box": boxes[i], "prediction": xy[i][5:17], "keypointConfidences": conf[i][5:17], "all17Prediction": xy[i], "all17Confidences": conf[i]})
            selected = max(range(len(scores)), key=lambda i: scores[i])
            item.update(item["candidates"][selected], selectedCandidateIndex=selected)
        predictions.append(item)
    assert len(predictions) == 240
    save(ROOT / "candidate-val-predictions.json", predictions)
    save(ROOT / "prediction-manifest.json", {"checkpointSHA256": sha(ckpt), "runnerSHA256": sha(__file__), "frames": len(predictions), "seconds": time.monotonic() - started, "device": "mps", "batch": 1, "imgsz": 640, "rect": False, "boxThreshold": .25, "sourcePolicy": "scalar path, exactly one matched result, NMS warnings fatal", "bodySlots": list(range(5, 17)), "selection": "Highest-confidence person, no ground-truth matching"})
    for threshold, suffix in ((.3, "primary"), (0, "all-scores")):
        result = score(rows, predictions, threshold)
        save(ROOT / f"candidate-{suffix}-score.json", result)
        print(json.dumps({"policy": suffix, "jointPCK0.05BodyScale": result["jointPCK0.05BodyScale"], "wristPCK0.05BodyScale": result["wrists"]["jointPCK0.05BodyScale"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "train", "predict"))
    args = p.parse_args()
    {"prepare": prepare, "train": train, "predict": predict}[args.stage]()
