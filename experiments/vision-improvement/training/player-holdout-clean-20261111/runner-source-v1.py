"""Isolated, leakage-aware two-endpoint YOLO development experiment.

Run with ../../2d-tracking/.venv/bin/python from this directory (or any cwd).
Never writes the labeler database, models/active.json, or shared trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parents[1] / "2d-tracking"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runtime-config"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "matplotlib"))
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def key(row):
    return f"{row['clipId']}:{row['frameIndex']}"


def eligible(row):
    if key(row) == "821a299e99cb0a60b164:56":
        return False, "quarantined_visually_confirmed_swapped_endpoints"
    if row["status"] != "reviewed":
        return False, "not_human_reviewed"
    if not all(row.get("points", {}).get(p, {}).get("state") == "visible" for p in ("tip", "knob")):
        return False, "no_complete_visible_pair_not_a_negative"
    for p in ("tip", "knob"):
        point = row["points"][p]
        for axis, size in (("x", row["imageWidth"]), ("y", row["imageHeight"])):
            value = point.get(axis)
            pixel = point.get("pixel" + axis.upper())
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                return False, "invalid_coordinates"
            if not isinstance(pixel, (int, float)) or not math.isfinite(pixel) or abs(pixel - value * size) > .01:
                return False, "coordinate_mismatch"
    return True, None


def prepare(run):
    import av
    import yaml
    if run.exists():
        raise ValueError("Refusing to overwrite an existing immutable experiment")
    run.mkdir(parents=True)
    source = EXP / "labeler" / "annotations.sqlite3"
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src:
        with sqlite3.connect(run / "labels.sqlite3") as dest:
            src.backup(dest)
            rows = [json.loads(r[0]) for r in dest.execute("SELECT record FROM labels ORDER BY clip,frame")]
    clip_data = json.loads((EXP / "labeler" / "clips.json").read_text())
    save(run / "clips.json", clip_data)
    clips = {c["id"]: c for c in clip_data["clips"]}
    selected, excluded = [], []
    for row in rows:
        ok, reason = eligible(row)
        (selected if ok else excluded).append(row if ok else {"key": key(row), "reason": reason, "row": row})
    split = {"train": [], "val": [], "test": []}
    for row in selected:
        title = clips[row["clipId"]]["title"]
        group = "test" if title == "Randy Guzman" else "val" if title == "Trey Snyder" else "train"
        split[group].append(row)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not {clips[r["clipId"]]["title"] for r in split[a]} & {clips[r["clipId"]]["title"] for r in split[b]}
        assert not {r["videoSHA256"] for r in split[a]} & {r["videoSHA256"] for r in split[b]}
    dataset = run / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "labels").mkdir()
    extracted = set()
    for clip_id in sorted({r["clipId"] for r in selected}):
        c = clips[clip_id]
        assert sha(c["path"]) == c["sha256"]
        wanted = {r["frameIndex"]: r for r in selected if r["clipId"] == clip_id}
        with av.open(c["path"]) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i not in wanted:
                    continue
                row = wanted[i]
                assert row["videoSHA256"] == c["sha256"]
                assert frame.pts == row["source"]["pts"] and str(frame.time_base) == row["source"]["timeBase"]
                assert (frame.width, frame.height) == (row["imageWidth"], row["imageHeight"])
                stem = f"{clip_id}-{i:04d}"
                frame.to_image().save(dataset / "images" / (stem + ".jpg"), quality=97, subsampling=0)
                pts = [[row["points"][p][a] for a in ("x", "y")] for p in ("tip", "knob")]
                low = [max(0, min(p[a] for p in pts) - 24 / size) for a, size in enumerate((frame.width, frame.height))]
                high = [min(1, max(p[a] for p in pts) + 24 / size) for a, size in enumerate((frame.width, frame.height))]
                values = [0, *[(a + b) / 2 for a, b in zip(low, high)], *[b - a for a, b in zip(low, high)], *pts[0], 2, *pts[1], 2]
                (dataset / "labels" / (stem + ".txt")).write_text(" ".join(map(str, values)) + "\n")
                extracted.add(key(row))
    assert extracted == {key(r) for r in selected}
    for name, group in split.items():
        save(run / f"{name}-rows.json", group)
        (dataset / f"{name}.txt").write_text("".join(str(dataset / "images" / f"{r['clipId']}-{r['frameIndex']:04d}.jpg") + "\n" for r in group))
    # Test is deliberately absent from the trainer YAML and is never used for selection.
    config = {"path": str(dataset), "train": "train.txt", "val": "val.txt", "names": {0: "bat"}, "kpt_shape": [2, 3], "flip_idx": [0, 1]}
    (dataset / "split.yaml").write_text(yaml.safe_dump(config))
    save(run / "excluded.json", excluded)
    weights = EXP / "models" / "yolo11s-pose.pt"
    manifest = {
        "createdAt": datetime.now(timezone.utc).isoformat(), "run": str(run),
        "sourceWeights": str(weights), "sourceWeightsSHA256": sha(weights),
        "sourceWeightsMeaning": "Generic pretrained COCO human pose weights; not any bat-fitted checkpoint. Two-keypoint head initialized by trainer with common seed.",
        "snapshotSHA256": sha(run / "labels.sqlite3"), "runnerSHA256": sha(__file__),
        "totalRows": len(rows), "selectedRows": len(selected), "excludedCounts": dict(Counter(r["reason"] for r in excluded)),
        "splits": {name: {"count": len(group), "players": sorted({clips[r["clipId"]]["title"] for r in group}), "clips": sorted({r["clipId"] for r in group}), "keys": [key(r) for r in group]} for name, group in split.items()},
        "sessions": sorted({r["session"] for r in selected}),
        "endpointOrder": ["tip", "knob"], "bbox": "endpoint envelope plus 24 original-image pixels per side, image clipped",
        "negativePolicy": "No explicit absent-bat reviewed labels are available. Uncertain, occluded, skipped or incomplete frames are never converted to negative targets.",
        "reviewPolicy": "Only status=reviewed, valid, complete visible pairs. Human-reviewed model assistance retained with provenance. Drafts excluded. Quality agent visually confirmed swapped endpoints at 821a299e99cb0a60b164:56; quarantined without changing original labels. Evidence: ../../quality/reports/inspection/821a299e99cb0a60b164-56.png.",
        "selectionPolicy": "Predeclared 40-epoch final checkpoints (last.pt), common initialization/seed/split. Baseline original mosaic=0.25, candidate only changes mosaic=0. No iterative tuning on test.",
        "evaluationPolicy": "Predeclared conf=0.15, max_det=1, imgsz=768, rect=False square input. Also score deployment-like box>=0.25 and both endpoint confidences>=0.25. All held-out eligible frames count; missing detection fails PCK. Exact original pixel coordinates. Top-1 model predictions do not replay application wrist-associated candidate selection.",
        "limitations": ["All samples originate from one session and camera; test is player-held-out development evidence, not independent-session qualification.", "Small correlated frames, no explicit negative examples, no scored occluded endpoints, no 3D endpoint truth.", "Default Ultralytics custom two-keypoint OKS sigmas=0.5 are permissive; use pixel/bat-normalized errors and miss-aware PCK instead of pose mAP to judge quality."],
    }
    save(run / "manifest.json", manifest)
    print(json.dumps({"run": str(run), "splits": {k: v["count"] for k, v in manifest["splits"].items()}, "excluded": manifest["excludedCounts"]}), flush=True)


def configure_runtime():
    import torch
    import ultralytics
    from ultralytics.utils import SETTINGS
    torch.set_num_threads(4)
    SETTINGS.update({k: False for k in ("sync", "wandb", "mlflow", "comet", "clearml", "hub", "tensorboard", "neptune", "dvc") if k in SETTINGS})
    return torch, {"platform": platform.platform(), "torch": torch.__version__, "ultralytics": ultralytics.__version__, "device": "mps" if torch.backends.mps.is_available() else "cpu"}


def train(run, recipe, epochs):
    from ultralytics import YOLO
    torch, runtime = configure_runtime()
    if runtime["device"] != "mps":
        raise RuntimeError("Apple GPU unavailable; run with narrowly authorized host execution instead of accidental slow CPU training")
    manifest = json.loads((run / "manifest.json").read_text())
    assert sha(manifest["sourceWeights"]) == manifest["sourceWeightsSHA256"]
    assert epochs == 40, "This experiment predeclares forty epochs"
    output = run / recipe
    if output.exists():
        raise ValueError("Refusing to overwrite an existing training run")
    params = dict(imgsz=768, batch=8, workers=0, device=runtime["device"], optimizer="AdamW", lr0=.0008,
                  lrf=.05, weight_decay=.0005, warmup_epochs=3, nbs=8, freeze=10, seed=42,
                  deterministic=True, patience=0, plots=False, amp=False, degrees=10, translate=.12,
                  scale=.25, fliplr=.5, flipud=0, mosaic=.25 if recipe == "baseline" else 0,
                  close_mosaic=10, mixup=0, hsv_h=.01, hsv_s=.25, hsv_v=.25,
                  project=str(run), exist_ok=False, verbose=False, save=True, save_period=10)
    started = time.monotonic()
    save(run / f"{recipe}-config.json", {"runtime": runtime, "parameters": params, "epochs": epochs, "sourceWeightsSHA256": manifest["sourceWeightsSHA256"], "checkpointSelection": "fixed final epoch, last.pt"})
    model = YOLO(manifest["sourceWeights"])
    def progress(trainer):
        save(run / f"{recipe}-status.json", {"state": "training", "recipe": recipe, "epoch": trainer.epoch + 1, "epochs": trainer.epochs, "elapsedSeconds": round(time.monotonic() - started, 2)})
    model.add_callback("on_train_epoch_end", progress)
    model.train(data=str(run / "dataset" / "split.yaml"), epochs=epochs, name=recipe, **params)
    checkpoint = output / "weights" / "last.pt"
    save(run / f"{recipe}-status.json", {"state": "complete", "recipe": recipe, "epochs": epochs, "elapsedSeconds": round(time.monotonic() - started, 2), "checkpoint": str(checkpoint), "checkpointSHA256": sha(checkpoint)})


def predict(run, recipe, split_name):
    from ultralytics import YOLO
    torch, runtime = configure_runtime()
    rows = json.loads((run / f"{split_name}-rows.json").read_text())
    checkpoint = run / recipe / "weights" / "last.pt"
    model = YOLO(str(checkpoint))
    images = [str(run / "dataset" / "images" / f"{r['clipId']}-{r['frameIndex']:04d}.jpg") for r in rows]
    details = []
    start = time.monotonic()
    for row, result in zip(rows, model.predict(images, imgsz=768, rect=False, conf=.15, max_det=1, device=runtime["device"], batch=8, verbose=False, stream=True)):
        item = {"clipId": row["clipId"], "frameIndex": row["frameIndex"], "detected": bool(len(result.boxes))}
        if len(result.boxes):
            item.update(prediction=result.keypoints.xy[0].cpu().tolist(), score=float(result.boxes.conf[0]), endpointConfidences=result.keypoints.conf[0].cpu().tolist())
        details.append(item)
    save(run / f"{recipe}-{split_name}-predictions.json", details)
    save(run / f"{recipe}-{split_name}-prediction-manifest.json", {"checkpointSHA256": sha(checkpoint), "runtime": runtime, "confidenceThreshold": .15, "maxDetections": 1, "imageSize": 768, "rect": False, "frames": len(rows), "elapsedSecondsIncludingWarmup": time.monotonic() - start})
    print(json.dumps({"recipe": recipe, "split": split_name, "frames": len(rows), "detected": sum(r["detected"] for r in details)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "train", "predict"))
    p.add_argument("--run", type=Path, default=ROOT / "player-holdout-clean-20261111")
    p.add_argument("--recipe", choices=("baseline", "candidate"), default="baseline")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--epochs", type=int, default=40)
    a = p.parse_args()
    run = a.run.resolve()
    if a.stage == "prepare": prepare(run)
    elif a.stage == "train": train(run, a.recipe, a.epochs)
    else: predict(run, a.recipe, a.split)
