"""Freeze fresh reviewed training rows while retaining the Nov18 benchmark.

Reads local audited/materialized snapshots only. No network access, fitting,
source-label edits, or writes to historical/cancelled experiment directories.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("training_contract", REPO / "training/run_cycle.py")
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        output.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def key(row):
    return row["videoSHA256"], row["frameIndex"]


def group(row):
    return row.get("canonicalGroup") or f"{row['session']}/{row['trial']}"


def trial_keys(row):
    supplied = row.get("trialKeys")
    if supplied:
        return set(supplied)
    trial = row.get("trial")
    return {"trial:" + trial} if trial and trial != "unknown" else {"video:" + row["videoSHA256"]}


def protected_identity(rows):
    return {"sessions": {r["session"] for r in rows if r.get("session")},
            "videos": {r["videoSHA256"] for r in rows},
            "groups": {group(r) for r in rows},
            "trialKeys": set().union(*(trial_keys(r) for r in rows))}


def choose_training(fresh, protected):
    selected, excluded = [], []
    if len({key(row) for row in fresh}) != len(fresh):
        raise ValueError("Duplicate fresh frame identities; audit must resolve them before splitting")
    for row in fresh:
        if row.get("status") != "reviewed":
            raise ValueError("Fresh audited rows must be reviewed")
        if not row.get("session") or row["session"] == "unknown":
            reason = "unknown_session"
        elif row["session"] in protected["sessions"]:
            reason = "protected_session"
        elif row["videoSHA256"] in protected["videos"]:
            reason = "protected_video"
        elif group(row) in protected["groups"]:
            reason = "protected_group"
        elif trial_keys(row) & protected["trialKeys"]:
            reason = "protected_trial_key"
        else:
            reason = None
        if reason:
            excluded.append({"videoSHA256": row["videoSHA256"], "frameIndex": row["frameIndex"], "reason": reason})
        else:
            selected.append(row)
    if not selected:
        raise ValueError("No fresh training frames remain outside protected benchmark sessions")
    return sorted(selected, key=key), excluded


def validate_snapshot(snapshot):
    """Check the audited output against materialized images and pinned sources."""
    from PIL import Image
    inventory = read(snapshot / "inventory.json")
    image_rows = read(snapshot / "image-manifest.json")
    images = {key(row): row for row in image_rows}
    if len(images) != len(image_rows):
        raise ValueError("Duplicate image-manifest identities")
    verified_images, verified_sources, all_rows = set(), {}, {}
    inputs = []
    for name in ("inventory.json", "image-manifest.json", "cloud-objects.json", "exclusions.json", "assignment-inventory.json"):
        path = snapshot / name
        inputs.append({"path": str(path), "sha256": sha(path)})
    for task in ("body", "bat"):
        rows_path = snapshot / f"{task}-rows.json"
        rows = read(rows_path)
        if len(rows) != inventory["currentBodyFrames" if task == "body" else "currentBatFrames"]:
            raise ValueError("Audited inventory and materialized row counts differ")
        if len(set(map(key, rows))) != len(rows):
            raise ValueError("Duplicate audited rows")
        for row in rows:
            contract.verify_frame_identity(row)
            contract.label(row, task)
            entry = images[key(row)]
            path = Path(row["imagePath"])
            if path.resolve() != Path(entry["imagePath"]).resolve() or row["imageSHA256"] != entry["imageSHA256"]:
                raise ValueError("Image manifest and row source differ")
            identity = (str(path.resolve()), row["imageSHA256"])
            if identity not in verified_images:
                if sha(path) != row["imageSHA256"]:
                    raise ValueError("Materialized image hash changed")
                with Image.open(path) as image:
                    if image.size != (row["width"], row["height"]):
                        raise ValueError("Materialized image dimensions changed")
                verified_images.add(identity)
            video = snapshot / "sources" / (row["videoSHA256"] + ".mp4")
            source_identity = (row["source"]["uri"], str(row["source"]["generation"]))
            if row["videoSHA256"] in verified_sources:
                if verified_sources[row["videoSHA256"]] != source_identity:
                    raise ValueError("One video hash has inconsistent pinned source identities")
            else:
                meta_path = video.with_suffix(".mp4.metadata.json")
                meta = read(meta_path)
                if (meta["uri"], str(meta["generation"])) != source_identity or meta["sha256"] != row["videoSHA256"] or sha(video) != row["videoSHA256"]:
                    raise ValueError("Pinned source video integrity check failed")
                verified_sources[row["videoSHA256"]] = source_identity
                inputs.append({"path": str(meta_path), "sha256": sha(meta_path)})
        all_rows[task] = rows
        inputs.append({"path": str(rows_path), "sha256": sha(rows_path)})
    return all_rows, {"inputs": inputs, "verifiedImages": len(verified_images), "verifiedSourceVideos": len(verified_sources)}


def supervised_changes(old, new, task):
    old_map, new_map = {key(r): r for r in old}, {key(r): r for r in new}
    common = old_map.keys() & new_map.keys()
    return {"previousTrainingFrames": len(old), "freshTrainingFrames": len(new),
            "addedFrameIdentities": len(new_map.keys() - old_map.keys()),
            "removedFrameIdentities": len(old_map.keys() - new_map.keys()),
            "retainedFrameIdentities": len(common),
            "retainedFramesWithChangedSupervision": sum(contract.label(old_map[k], task) != contract.label(new_map[k], task) for k in common)}


def verified_prior(run, *, check_protocol=False):
    manifest = read(run / "manifest.json")
    inputs_path = run / "inputs.json"
    if sha(inputs_path) != manifest["inputManifestSHA256"]:
        raise ValueError("Historical input manifest hash changed")
    frozen = {str(Path(item["path"]).resolve()): item["sha256"] for item in read(inputs_path)}
    if check_protocol:
        path = run / "protocol.json"
        if frozen.get(str(path.resolve())) != sha(path):
            raise ValueError("Previous training protocol no longer matches its frozen hash")
    rows = {}
    for split in ("train", "val", "evaluation"):
        path = run / f"{split}-rows.json"
        if frozen.get(str(path.resolve())) != sha(path):
            raise ValueError("Historical split rows no longer match their frozen hashes")
        rows[split] = read(path)
    return rows


def previous_checkpoint(run):
    status = read(run / "training-status.json")
    if status.get("state") != "complete":
        raise ValueError("Previous training must have a completed final checkpoint")
    path = run / "candidate/weights/last.pt"
    digest = sha(path)
    scored = read(run / "candidate-prediction-manifest.json")
    if digest != status["checkpointSHA256"] or digest != scored["checkpointSHA256"]:
        raise ValueError("Previous final checkpoint differs from the checkpoint used for evaluation")
    if scored["rowsSHA256"] != sha(run / "evaluation-rows.json"):
        raise ValueError("Previous prediction receipt differs from the frozen evaluation rows")
    return path


def prepare(source, snapshot, output, repository=REPO, approved=False, *, prior_run=False,
            recipe_profile="continuation", baseline_release=None):
    source, snapshot, output, repository = (Path(p).resolve() for p in (source, snapshot, output, repository))
    base = repository / "outputs"
    if output == base or not output.is_relative_to(base):
        raise ValueError("New protocols must use this repository's ignored outputs directory")
    if output == source or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Refusing to write into the historical source cycle")
    destinations = [output / f"{t}-protocol.json" for t in ("body", "bat")]
    destinations += [output / "split-receipt.json", output / "private-parity-rows.json", output / "training-exclusions.json"]
    destinations += [output / "dataset" / f"{t}-{s}-rows.json" for t in ("body", "bat") for s in ("train", "val", "evaluation")]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Refusing to overwrite an existing protocol/split")
    fresh, integrity = validate_snapshot(snapshot)
    previous_tasks = {task: (source if prior_run else source / "training") / task for task in ("body", "bat")}
    prior = {task: verified_prior(folder, check_protocol=prior_run) for task, folder in previous_tasks.items()}
    protected = protected_identity([r for task in prior.values() for split in ("val", "evaluation") for r in task[split]])
    defaults = {} if prior_run else read(repository / "models/manifest.json")["defaults"]
    recipe = contract.fixed_recipe("body", recipe_profile)
    receipt = {"sourceCycle": str(source), "freshSnapshot": str(snapshot), "approvedForTraining": approved,
               "datasetPolicy": "Newest audited reviewed training rows from unprotected sessions only. Missing/removed prior training rows are not resurrected.",
               "evaluationPolicy": "Byte-identical previous validation and evaluation rows. Reused benchmark, not a blind or independent new test. All old validation/evaluation sessions and linked identities remain excluded from training, including newly labeled frames.",
               "recipe": f"{recipe['epochs']} continuation epochs with a fresh optimizer, lr0={recipe['lr0']}, and final-epoch checkpoint selection.",
               "snapshotIntegrity": integrity, "protected": {k: sorted(v) for k, v in protected.items()}, "tasks": {}}
    if baseline_release:
        receipt["baselineRelease"] = baseline_release
    if prior_run:
        receipt["previousRun"] = str(source)
        previous_receipt = source / "split-receipt.json"
        if previous_receipt.exists():
            receipt["previousSnapshot"] = read(previous_receipt)["freshSnapshot"]
    prepared, excluded_all, parity = {}, {}, []
    for task in ("body", "bat"):
        train, excluded = choose_training(fresh[task], protected)
        excluded_all[task] = excluded
        old_folder = previous_tasks[task]
        old_protocol = read(old_folder / "protocol.json")
        rows = {"train": train, "val": prior[task]["val"], "evaluation": prior[task]["evaluation"]}
        if prior_run:
            initial = previous_checkpoint(old_folder)
        else:
            initial = repository / defaults[task]["checkpoint"]
            archived_hash = read(old_folder / "training-status.json")["checkpointSHA256"]
            if sha(initial) != archived_hash or archived_hash != defaults[task]["sha256"]:
                raise ValueError("Continuation baseline differs from the verified Nov18 checkpoint")
        model_hash = sha(initial)
        lineage_rows = prior[task]["train"] + train
        videos = set(old_protocol["evaluationExclusionVideos"]) | {r["videoSHA256"] for r in lineage_rows}
        groups = set(old_protocol["evaluationExclusionGroups"]) | {group(r) for r in lineage_rows}
        for split in ("val", "evaluation"):
            if videos & {r["videoSHA256"] for r in rows[split]} or groups & {group(r) for r in rows[split]}:
                raise ValueError("A benchmark source overlaps the continuation lineage")
        paths = {split: str(output / "dataset" / f"{task}-{split}-rows.json") for split in rows}
        model = {"path": str(initial), "sha256": model_hash, "size": initial.stat().st_size}
        protocol = {"approvedForTraining": approved, "task": task, "rows": paths, "initial": model, "baselines": {"previous": model},
                    "evaluationExclusionVideos": sorted(videos), "evaluationExclusionGroups": sorted(groups), "recipeProfile": recipe_profile, "recipe": contract.fixed_recipe(task, recipe_profile),
                    "limits": old_protocol["limits"] + [receipt["evaluationPolicy"], "New training labels retain their source review/interpolation provenance; no claim of independent gold-standard truth."]}
        prepared[task] = (rows, protocol)
        receipt["tasks"][task] = {"initialSHA256": model_hash, "freshEligibleFrames": len(fresh[task]),
            "excludedFreshFrames": len(excluded), "exclusionReasons": dict(Counter(item["reason"] for item in excluded)),
            "trainingChanges": supervised_changes(prior[task]["train"], train, task), "splits": {}}
    # All validation occurs before writing any split/protocol. The two benchmark
    # files are copied as bytes so their exact historical hashes are preserved.
    for task, (rows, protocol) in prepared.items():
        for split, values in rows.items():
            path = Path(protocol["rows"][split])
            path.parent.mkdir(parents=True, exist_ok=True)
            if split == "train":
                write(path, values)
            else:
                old_path = previous_tasks[task] / f"{split}-rows.json"
                shutil.copyfile(old_path, path)
                if sha(path) != sha(old_path):
                    raise ValueError("Historical benchmark bytes changed during copy")
            receipt["tasks"][task]["splits"][split] = {"frames": len(values), "swings": len({r["videoSHA256"] for r in values}),
                "sessions": len({r["session"] for r in values}), "rowsSHA256": sha(path)}
        write(output / f"{task}-protocol.json", protocol)
        if task == "body":
            for video in sorted({r["videoSHA256"] for r in rows["evaluation"]}):
                frames = sorted((r for r in rows["evaluation"] if r["videoSHA256"] == video), key=lambda r: r["frameIndex"])
                parity.extend(frames[i] for i in (0, len(frames) // 2, len(frames) - 1))
    write(output / "private-parity-rows.json", parity)
    write(output / "training-exclusions.json", excluded_all)
    write(output / "split-receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    previous = parser.add_mutually_exclusive_group(required=True)
    previous.add_argument("--source-cycle", type=Path, help="Archived benchmark cycle with training/body and training/bat")
    previous.add_argument("--prior-run", type=Path, help="Completed, scored run with body and bat final checkpoints")
    parser.add_argument("--recipe-profile", choices=("continuation", "short-continuation"), default="continuation")
    parser.add_argument("--baseline-release", help="Human-readable release identifier for the scored starting checkpoints")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approve-for-training", action="store_true")
    args = parser.parse_args()
    result = prepare(args.prior_run or args.source_cycle, args.snapshot, args.output, approved=args.approve_for_training,
                     prior_run=args.prior_run is not None, recipe_profile=args.recipe_profile, baseline_release=args.baseline_release)
    print(json.dumps(result["tasks"], indent=2))
