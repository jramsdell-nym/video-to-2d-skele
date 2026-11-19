"""Create the package builder's numeric-only summary from completed run artifacts."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_counts(run):
    receipt = read(run / "split-receipt.json")
    snapshot = Path(receipt["freshSnapshot"])
    frozen_files = {item["path"]: item["sha256"] for item in receipt["snapshotIntegrity"]["inputs"]}
    for name in ("inventory.json", "assignment-inventory.json"):
        path = snapshot / name
        if sha(path) != frozen_files[str(path)]:
            raise ValueError(f"Audited dataset input changed: {name}")
    inventory = read(snapshot / "inventory.json")
    prior = Path(receipt.get("previousSnapshot") or Path(receipt["sourceCycle"]) / "dataset")
    previous_inventory = read(prior / "inventory.json")
    prior_ids = {row["assignmentId"] for row in read(prior / "assignment-inventory.json")}
    current_ids = {row["assignmentId"] for row in read(snapshot / "assignment-inventory.json")}
    counts = {"addedCompletedAssignments": len(current_ids - prior_ids),
              "completedAssignments": len(current_ids), "submittedFrames": inventory["queueClaimedFrames"]}
    for task in ("body", "bat"):
        frame_field = f"current{task.title()}Frames"
        counts[task] = {"previousEligibleFrames": previous_inventory[frame_field],
                        "eligibleFrames": inventory[frame_field],
                        "eligibleSwings": inventory[f"{task}CanonicalSwings"]}
    return counts


def build(run, release, output):
    run = run.resolve()
    receipt = read(run / "split-receipt.json")
    spec = {"schemaVersion": 1, "release": release, "models": {},
            "parityRows": str(run / "private-parity-rows.json"),
            "parityRowsSHA256": sha(run / "private-parity-rows.json"),
            "summary": {"baselineRelease": receipt.get("baselineRelease", "2026-11-18"),
                        "training": {}, "evaluation": {}}}
    if "freshSnapshot" in receipt:
        spec["summary"]["dataset"] = dataset_counts(run)
    scores = {}
    reference_keys = None
    for task in ("body", "bat"):
        folder = run / task
        status = read(folder / "training-status.json")
        assert status["state"] == "complete", f"Incomplete {task} fit"
        checkpoint = folder / "candidate/weights/last.pt"
        assert sha(checkpoint) == status["checkpointSHA256"]
        config = read(folder / "training-config.json")["params"]
        manifest = read(folder / "manifest.json")
        spec["models"][task] = {"checkpoint": str(checkpoint), "sha256": sha(checkpoint), "imageSize": config["imgsz"]}
        train, val = manifest["splits"]["train"], manifest["splits"]["val"]
        spec["summary"]["training"][task] = {"epochs": config["epochs"], "imageSize": config["imgsz"],
            "learningRate": config["lr0"], "seed": config["seed"], "freeze": config["freeze"],
            "trainFrames": train["frames"], "trainVideos": train["videos"],
            "valFrames": val["frames"], "valVideos": val["videos"]}
        rows = read(folder / "evaluation-rows.json")
        keys = {(r["videoSHA256"], r["frameIndex"]) for r in rows}
        assert len(keys) == len(rows)
        if reference_keys is None:
            reference_keys = keys
            spec["summary"]["evaluation"].update(frames=len(rows), videos=len({r["videoSHA256"] for r in rows}),
                                                   sessions=len({r["session"] for r in rows}))
        else:
            assert keys == reference_keys, "Body and BAT evaluate different frame identities"
        frozen_rows_sha = sha(folder / "evaluation-rows.json")
        for name, expected_checkpoint in (("previous", manifest["models"]["previous"]["sha256"]),
                                          ("candidate", status["checkpointSHA256"])):
            prediction = read(folder / f"{name}-prediction-manifest.json")
            if (prediction["checkpointSHA256"] != expected_checkpoint or
                    prediction["rowsSHA256"] != frozen_rows_sha or prediction["visualOnly"] or
                    prediction["predictionsSHA256"] != sha(folder / f"{name}-predictions.json")):
                raise ValueError(f"{task} {name} predictions do not match the frozen evaluation")
        scores[task] = {name: read(folder / f"{name}-primary-score.json") for name in ("previous", "candidate")}
    for field, task, nested, correct, total in [
        ("bodyJoints", "body", None, "within0.05BodyScale", "referenceJoints"),
        ("wrists", "body", "wrists", "within0.05BodyScale", "referenceJoints"),
        ("batPairs", "bat", None, "bothEndpointsWithin0.1BatLength", "normalizableFrames"),
    ]:
        values = {}
        for source_name, result_name in (("previous", "baseline"), ("candidate", "candidate")):
            score = scores[task][source_name]
            if nested:
                score = score[nested]
            values[result_name] = {"correct": score[correct], "total": score[total]}
            assert 0 <= score[correct] <= score[total]
        assert values["baseline"]["total"] == values["candidate"]["total"]
        spec["summary"]["evaluation"][field] = values
    assert not output.exists(), "Refusing to overwrite a package spec"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec, indent=2) + "\n")
    print(json.dumps(spec["summary"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--release", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build(args.run, args.release, args.output)
