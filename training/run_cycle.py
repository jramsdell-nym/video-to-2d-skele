"""Frozen-input, fixed-recipe 2D body/bat candidate training and evaluation.

Prepare consumes a root-approved JSON protocol. No source labels, deployed
models, external services or installed packages are modified. Every model uses
the same scalar-image prediction path and original-pixel output contract.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import shutil
import time

REPO = Path(__file__).resolve().parents[1]
VISION = REPO / "experiments/vision-improvement"
FLIP17 = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def configure_environment(run):
    """Generated runtime state belongs to the private output, never the archive."""
    os.environ["YOLO_CONFIG_DIR"] = str(run / "runtime-config")
    os.environ["MPLCONFIGDIR"] = str(run / "matplotlib")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"


def output_path(path):
    path = Path(path).resolve()
    base = (REPO / "outputs").resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError("Training outputs must use a new subdirectory of this repository's ignored outputs directory")
    return path


def fixed_recipe(task, profile="continuation"):
    if profile not in ("continuation", "short-continuation"):
        raise ValueError(f"Unknown training recipe: {profile}")
    epochs, learning_rate = (10, .000025) if profile == "short-continuation" else (20, .00005)
    return {"epochs": epochs, "imgsz": 640 if task == "body" else 768,
            "batch": 8, "freeze": 10, "lr0": learning_rate, "mosaic": 0}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def key(row):
    return row["videoSHA256"], int(row["frameIndex"])


def verify_frame_identity(row):
    assert re.fullmatch(r"[0-9a-f]{64}", row["videoSHA256"]), "Invalid source video hash"
    assert type(row["frameIndex"]) is int and row["frameIndex"] >= 0, "Invalid frame index"
    frame = row.get("frame", row.get("sourceFrame"))
    assert isinstance(frame, dict) and frame.get("index") == row["frameIndex"], "Missing or mismatched source frame identity"
    assert type(frame.get("pts")) is int, "Source PTS must be an integer"
    time_base = Fraction(frame["timeBase"])
    assert time_base > 0, "Invalid time base"
    if "time" in frame:
        assert finite(frame["time"]) and abs(float(frame["pts"] * time_base) - frame["time"]) < 2e-6, "Source PTS/time mismatch"


def group(row):
    return row.get("canonicalGroup") or f"{row['session']}:{row['trial']}"


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def label(row, task):
    if task == "body":
        assert row["status"] == "reviewed" and len(row["points"]) == 12
        points = [[0, 0, 0] for _ in range(5)]
        for point in row["points"]:
            visibility = point["visibility"]
            assert visibility in (0, 1, 2)
            if visibility:
                assert all(finite(point[a]) and 0 <= point[a] <= 1 for a in ("x", "y"))
                points.append([point["x"], point["y"], visibility])
            else:
                points.append([0, 0, 0])
        box = row["box"]
    else:
        if row["status"] == "absent":
            assert row.get("explicitAbsenceReviewed") is True
            return ""
        assert row["status"] == "reviewed"
        points = []
        for name in ("tip", "knob"):
            point = row["points"][name]
            visibility = point.get("visibility", 2 if point["state"] == "visible" else None)
            assert visibility in (1, 2)
            assert point["state"] == ("visible" if visibility == 2 else "occluded")
            assert all(finite(point[a]) and 0 <= point[a] <= 1 for a in ("x", "y"))
            for axis, size in (("x", row["imageWidth"]), ("y", row["imageHeight"])):
                assert abs(point["pixel" + axis.upper()] - point[axis] * size) < .01
            points.append([point["x"], point["y"], visibility])
        width, height = row["imageWidth"], row["imageHeight"]
        box = [max(0, min(p[0] for p in points) - 24 / width),
               max(0, min(p[1] for p in points) - 24 / height),
               min(1, max(p[0] for p in points) + 24 / width),
               min(1, max(p[1] for p in points) + 24 / height)]
    assert len(box) == 4 and all(finite(v) and 0 <= v <= 1 for v in box)
    assert box[2] > box[0] and box[3] > box[1]
    values = [0, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2,
              box[2] - box[0], box[3] - box[1]]
    values += [v for point in points for v in point]
    assert len(values) == (56 if task == "body" else 11)
    return " ".join(map(str, values)) + "\n"


def prepare(run, protocol_path):
    assert not (run / "manifest.json").exists(), "Refusing to replace frozen inputs"
    protocol = read(protocol_path)
    assert protocol["approvedForTraining"] is True
    task = protocol["task"]
    assert task in ("body", "bat")
    recipe = fixed_recipe(task, protocol.get("recipeProfile", "continuation"))
    assert protocol["recipe"] == recipe, "Protocol must contain the approved fixed continuation recipe"
    rows = {split: read(path) for split, path in protocol["rows"].items()}
    assert set(rows) == {"train", "val", "evaluation"}
    assert rows["train"] and rows["val"] and rows["evaluation"]
    for split, values in rows.items():
        assert len(set(map(key, values))) == len(values), f"Duplicate {split} rows"
        for row in values:
            verify_frame_identity(row)
    for left, right in (("train", "val"), ("train", "evaluation"), ("val", "evaluation")):
        for identity in (key, group, lambda r: r["videoSHA256"]):
            assert not set(map(identity, rows[left])) & set(map(identity, rows[right])), f"{left}/{right} leakage"
    seen_source_videos = set(protocol["evaluationExclusionVideos"])
    assert not seen_source_videos & {r["videoSHA256"] for r in rows["evaluation"]}, "Warm-start checkpoint saw evaluation videos"
    seen_source_groups = set(protocol["evaluationExclusionGroups"])
    assert not seen_source_groups & set(map(group, rows["evaluation"])), "Compared checkpoint saw a synchronized evaluation trial"
    run.mkdir(parents=True, exist_ok=True)
    frozen = []
    def freeze(path):
        frozen.append({"path": str(Path(path).resolve()), "sha256": sha(path)})
    shutil.copyfile(protocol_path, run / "protocol.json")
    freeze(run / "protocol.json")
    scorer_source = VISION / ("research/body-experiment/score_body.py" if task == "body" else "quality/score_bat.py")
    shutil.copyfile(scorer_source, run / "scorer.py")
    freeze(run / "scorer.py")
    shutil.copyfile(__file__, run / "runner-source.py")
    freeze(run / "runner-source.py")
    models = {"initial": protocol["initial"], **protocol["baselines"]}
    for name, model in models.items():
        assert sha(model["path"]) == model["sha256"], f"Wrong {name} weights"
        freeze(model["path"])
    for split, values in rows.items():
        image_paths = []
        for row in values:
            image = Path(row["imagePath"])
            assert image.is_file() and sha(image) == row["imageSHA256"]
            from PIL import Image
            with Image.open(image) as decoded:
                expected = (row["width"], row["height"]) if task == "body" else (row["imageWidth"], row["imageHeight"])
                assert decoded.size == expected, "Image/annotation geometry mismatch"
            encoded_label = label(row, task)
            freeze(image)
            if split != "evaluation":
                stem = f"{row['videoSHA256']}_{row['frameIndex']}"
                target = run / "dataset/images" / split / (stem + image.suffix)
                target.parent.mkdir(parents=True, exist_ok=True)
                assert not target.exists()
                target.symlink_to(image.resolve())
                target_label = run / "dataset/labels" / split / (stem + ".txt")
                target_label.parent.mkdir(parents=True, exist_ok=True)
                target_label.write_text(encoded_label)
                freeze(target_label)
                image_paths.append(str(target))
        save(run / f"{split}-rows.json", values)
        freeze(run / f"{split}-rows.json")
        if split != "evaluation":
            listing = run / "dataset" / f"{split}.txt"
            listing.write_text("\n".join(image_paths) + "\n")
            freeze(listing)
    shape = [17, 3] if task == "body" else [2, 3]
    yaml = run / "dataset/data.yaml"
    yaml.write_text(f"path: {run / 'dataset'}\ntrain: train.txt\nval: val.txt\nnames:\n  0: {'person' if task == 'body' else 'bat'}\nkpt_shape: {shape}\nflip_idx: {FLIP17 if task == 'body' else [0, 1]}\n")
    freeze(yaml)
    save(run / "inputs.json", list({item["path"]: item for item in frozen}.values()))
    save(run / "manifest.json", {
        "createdAt": datetime.now(timezone.utc).isoformat(), "task": task,
        "inputManifestSHA256": sha(run / "inputs.json"), "runnerSHA256": sha(__file__),
        "models": models, "keypointShape": shape,
        "splits": {name: {"frames": len(values), "videos": len({r['videoSHA256'] for r in values}), "groups": sorted(set(map(group, values)))} for name, values in rows.items()},
        "recipe": recipe,
        "selection": f"Fixed final checkpoint last.pt after {recipe['epochs']} epochs. Scoring uses the frozen evaluation rows.",
        "limits": protocol.get("limits", []),
        "bodyMask": "Body12 targets map to COCO slots5:17. Five facial points v0 suppress coordinate loss; negative visibility targets may reduce facial confidence." if task == "body" else None,
        "batOrder": ["tip", "knob"] if task == "bat" else None,
        "batSupervisionPolicy": "Train on reviewed localizable visibility1 or2 preserving original coordinates/masks. Primary paired-endpoint score uses both-visible reference pairs; partial/occluded targets are reported separately. Manual/source metadata does not establish independent label placement." if task == "bat" else None,
        "batVisibilityCounts": {split: {name: {str(v): sum(r.get('points', {}).get(name, {}).get('visibility', 2 if r.get('points', {}).get(name, {}).get('state') == 'visible' else None) == v for r in values) for v in (0, 1, 2)} for name in ('tip', 'knob')} for split, values in rows.items()} if task == "bat" else None,
    })
    print(json.dumps(read(run / "manifest.json")["splits"]), flush=True)


def validate(run):
    manifest = read(run / "manifest.json")
    assert sha(__file__) == manifest["runnerSHA256"], "Runner changed after freeze"
    assert sha(run / "inputs.json") == manifest["inputManifestSHA256"]
    for item in read(run / "inputs.json"):
        assert sha(item["path"]) == item["sha256"], f"Input changed: {item['path']}"
    return manifest


def runtime():
    import torch
    import ultralytics
    from ultralytics.utils import SETTINGS
    torch.set_num_threads(4)
    assert torch.backends.mps.is_available(), "MPS unavailable; request narrowly authorized host execution"
    SETTINGS.update({k: False for k in ("sync", "wandb", "mlflow", "comet", "clearml", "hub", "tensorboard", "neptune", "dvc") if k in SETTINGS})
    return torch, {"platform": platform.platform(), "torch": torch.__version__, "ultralytics": ultralytics.__version__, "device": "mps"}


def train(run):
    configure_environment(run)
    from ultralytics import YOLO
    manifest = validate(run)
    torch, versions = runtime()
    assert not (run / "candidate").exists(), "Refusing to overwrite fit"
    model = YOLO(manifest["models"]["initial"]["path"])
    assert model.model.model[-1].kpt_shape == manifest["keypointShape"]
    initial = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
    params = dict(data=str(run / "dataset/data.yaml"), **manifest["recipe"], workers=0, device="mps", seed=42, deterministic=True,
        optimizer="AdamW", lrf=.1, weight_decay=.0005, warmup_epochs=1, warmup_bias_lr=.0001, nbs=8, patience=0, amp=False,
        degrees=10, translate=.1, scale=.25, fliplr=.5, flipud=0, mixup=0, hsv_h=.01, hsv_s=.25, hsv_v=.25,
        val=False, plots=False, save=True, verbose=False, project=str(run), name="candidate", exist_ok=False)
    save(run / "training-config.json", {"params": params, "runtime": versions, "runnerSHA256": sha(__file__), "inputManifestSHA256": manifest["inputManifestSHA256"]})
    def verify(trainer):
        loaded = trainer.model.state_dict()
        mismatches = [k for k, value in initial.items() if k not in loaded or not torch.equal(value, loaded[k].detach().cpu())]
        assert not mismatches, f"Pretrained values did not transfer: {mismatches}"
        assert trainer.model.model[-1].kpt_shape == manifest["keypointShape"]
        save(run / "initialization-verification.json", {"allPretrainedTensorsIdenticalBeforeFit": True, "tensorsCompared": len(initial), "poseHeadTensorsCompared": sum('.cv4.' in k for k in initial), "shape": manifest["keypointShape"]})
    started = time.monotonic()
    model.add_callback("on_pretrain_routine_end", verify)
    model.add_callback("on_train_epoch_end", lambda trainer: save(run / "training-status.json", {"state": "training", "epoch": trainer.epoch + 1, "epochs": params["epochs"], "elapsedSeconds": time.monotonic() - started}))
    try:
        model.train(**params)
    except Exception as exc:
        save(run / "training-status.json", {"state": "failed", "seconds": time.monotonic() - started, "errorType": type(exc).__name__, "error": str(exc)})
        raise
    checkpoint = run / "candidate/weights/last.pt"
    save(run / "training-status.json", {"state": "complete", "epochs": params["epochs"], "seconds": time.monotonic() - started, "checkpoint": str(checkpoint), "checkpointSHA256": sha(checkpoint)})


def predict(run, name, inference_rows=None, inference_rows_sha256=None):
    configure_environment(run)
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    manifest = validate(run)
    _, versions = runtime()
    visual_only = inference_rows is not None
    output_directory = run / "video-predictions" if visual_only else run
    output_directory.mkdir(exist_ok=True)
    output = output_directory / f"{name}-predictions.json"
    assert not output.exists(), "Refusing to overwrite accepted predictions"
    checkpoint = run / "candidate/weights/last.pt" if name == "candidate" else Path(manifest["models"][name]["path"])
    if name == "candidate":
        status = read(run / "training-status.json")
        assert status["state"] == "complete" and sha(checkpoint) == status["checkpointSHA256"], "Candidate is not the verified final checkpoint"
    task = manifest["task"]
    rows_path = Path(inference_rows) if visual_only else run / "evaluation-rows.json"
    if visual_only:
        assert inference_rows_sha256 and sha(rows_path) == inference_rows_sha256, "Supply hash of frozen video-frame manifest"
    rows = read(rows_path)
    assert len(set(map(key, rows))) == len(rows), "Duplicate inference frame identities"
    for row in rows:
        verify_frame_identity(row)
    model = YOLO(str(checkpoint))
    assert model.model.model[-1].kpt_shape == manifest["keypointShape"]
    class RejectNMSWarning(logging.Handler):
        def emit(self, record):
            if "NMS" in record.getMessage() and record.levelno >= logging.WARNING:
                raise RuntimeError("NMS warning; incomplete inference is not an accuracy miss")
    handler = RejectNMSWarning()
    LOGGER.addHandler(handler)
    records = []
    started = time.monotonic()
    try:
        for i, row in enumerate(rows):
            path = Path(row["imagePath"])
            if visual_only:
                assert sha(path) == row["imageSHA256"]
            results = list(model.predict(str(path), imgsz=manifest["recipe"]["imgsz"], rect=False, conf=.25 if task == "body" else .15, max_det=10, batch=1, device="mps", stream=True, verbose=False))
            assert len(results) == 1 and Path(results[0].path).resolve() == path.resolve()
            result = results[0]
            record = {"videoSHA256": row["videoSHA256"], "frameIndex": row["frameIndex"],
                      "frame": row.get("frame", row.get("sourceFrame")),
                      "width": row.get("width", row.get("imageWidth")),
                      "height": row.get("height", row.get("imageHeight")),
                      "imagePath": str(path), "detected": bool(len(result.boxes)),
                      "prediction": None, "candidates": []}
            if "clipId" in row:
                record["clipId"] = row["clipId"]
            if record["detected"]:
                xy, confidence = result.keypoints.xy.cpu().tolist(), result.keypoints.conf.cpu().tolist()
                boxes, scores = result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()
                for j in range(len(scores)):
                    candidate = {"score": scores[j], "box": boxes[j], "prediction": xy[j][5:17] if task == "body" else xy[j], "keypointConfidences": confidence[j][5:17] if task == "body" else confidence[j]}
                    if task == "body":
                        candidate.update(all17Prediction=xy[j], all17Confidences=confidence[j])
                    else:
                        candidate["endpointConfidences"] = confidence[j]
                    record["candidates"].append(candidate)
                selected = max(range(len(scores)), key=lambda j: scores[j])
                record.update(record["candidates"][selected], selectedCandidateIndex=selected)
            records.append(record)
            if (i + 1) % 100 == 0:
                print(f"{name}: {i + 1}/{len(rows)}", flush=True)
    finally:
        LOGGER.removeHandler(handler)
    assert len(records) == len(rows) and set(map(key, rows)) == set(map(key, records))
    save(output, records)
    save(output_directory / f"{name}-prediction-manifest.json", {"checkpoint": str(checkpoint), "checkpointSHA256": sha(checkpoint), "rowsPath": str(rows_path), "rowsSHA256": sha(rows_path), "predictionsSHA256": sha(output), "frames": len(records), "seconds": time.monotonic() - started, "runtime": versions, "imgsz": manifest["recipe"]["imgsz"], "batch": 1, "rect": False, "max_det": 10, "selection": "Highest box confidence; no reference matching or temporal postprocessing", "sourcePolicy": "Scalar image calls, exactly one matching result, NMS warnings fatal", "visualOnly": visual_only})
    if visual_only:
        print(f"Saved {len(records)} video predictions for {name}", flush=True)
        return
    spec = importlib.util.spec_from_file_location("frozen_scorer", run / "scorer.py")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    if task == "body":
        for threshold, suffix in ((.3, "primary"), (0, "all-confidence")):
            save(run / f"{name}-{suffix}-score.json", scorer.score(rows, records, threshold))
    else:
        save(run / f"{name}-raw-score.json", scorer.score(rows, records))
        gated = []
        for record in records:
            copy = dict(record)
            copy["detected"] = record["detected"] and record.get("score", 0) >= .25
            if copy["detected"]:
                copy["prediction"] = [p if c >= .25 else None for p, c in zip(record["prediction"], record["endpointConfidences"])]
            gated.append(copy)
        save(run / f"{name}-primary-score.json", scorer.score(rows, gated))
    print(f"Saved {len(records)} predictions and scores for {name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "train", "predict"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--model", default="candidate")
    parser.add_argument("--inference-rows", type=Path)
    parser.add_argument("--inference-rows-sha256")
    args = parser.parse_args()
    run = output_path(args.run)
    if args.stage == "prepare":
        prepare(run, args.protocol.resolve())
    elif args.stage == "train":
        train(run)
    else:
        predict(run, args.model, args.inference_rows, args.inference_rows_sha256)
