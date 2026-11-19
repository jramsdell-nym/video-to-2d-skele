"""Compare two fitted checkpoints on newly submitted swings in held-out sessions."""
import argparse
import json
from pathlib import Path

import prepare_fresh_training as splits

cycle = splits.contract
TASKS = ("body", "bat")


def freeze(run, previous):
    selection = run / "new-swings"
    selection.mkdir(mode=0o700, exist_ok=False)
    old_rows = []
    eval_sessions = set()
    seen_videos, seen_groups = set(), set()
    for task in TASKS:
        old_rows.extend(splits.read(previous / "dataset" / f"{task}-rows.json"))
        old_eval = splits.read(previous / task / "evaluation-rows.json")
        eval_sessions.update(r["session"] for r in old_eval)
        prior = splits.read(previous / task / "protocol.json")
        seen_videos.update(prior["evaluationExclusionVideos"])
        seen_groups.update(prior["evaluationExclusionGroups"])
    old = splits.protected_identity(old_rows)
    seen_videos.update(old["videos"])
    seen_groups.update(old["groups"])
    receipt = {"policy": "All eligible newly seen swings in the prior evaluation sessions; exclude prior recording, swing and trial identities and both initializers' known training lineages.", "tasks": {}}
    for task in TASKS:
        rows = splits.read(run / "dataset" / f"{task}-rows.json")
        fresh = [r for r in rows if r["session"] in eval_sessions
                 and r["videoSHA256"] not in seen_videos
                 and splits.group(r) not in seen_groups
                 and not splits.trial_keys(r).intersection(old["trialKeys"])]
        if not fresh or len(set(map(splits.key, fresh))) != len(fresh):
            raise ValueError(f"Empty or duplicate new {task} evaluation rows")
        path = selection / f"{task}-rows.json"
        splits.write(path, sorted(fresh, key=splits.key))
        receipt["tasks"][task] = {"frames": len(fresh), "videos": len({r["videoSHA256"] for r in fresh}),
                                  "sessions": len({r["session"] for r in fresh}), "sha256": splits.sha(path)}
    splits.write(selection / "selection.json", receipt)
    print(json.dumps(receipt, indent=2))


def score(run):
    selection = run / "new-swings"
    receipt = splits.read(selection / "selection.json")
    report = {}
    for task in TASKS:
        primary = run / task
        folder = selection / task
        rows_path = selection / f"{task}-rows.json"
        if splits.sha(rows_path) != receipt["tasks"][task]["sha256"]:
            raise ValueError("The frozen supplemental sample changed")
        rows = splits.read(rows_path)
        train = splits.read(primary / "train-rows.json")
        blocked = splits.protected_identity(train + splits.read(primary / "val-rows.json"))
        for r in rows:
            if (r["session"] in blocked["sessions"] or r["videoSHA256"] in blocked["videos"]
                    or splits.group(r) in blocked["groups"] or splits.trial_keys(r) & blocked["trialKeys"]):
                raise ValueError("Supplemental evaluation overlaps training or validation")
        protocol = splits.read(primary / "protocol.json")
        status = splits.read(primary / "training-status.json")
        checkpoint = primary / "candidate/weights/last.pt"
        if status["state"] != "complete" or splits.sha(checkpoint) != status["checkpointSHA256"]:
            raise ValueError(f"Missing verified final {task} fit")
        protocol["baselines"] = {"previous": protocol["baselines"]["previous"],
                                 "retrained": {"path": str(checkpoint), "sha256": status["checkpointSHA256"]}}
        protocol["rows"]["evaluation"] = str(rows_path)
        path = selection / f"{task}-protocol.json"
        if not (folder / "manifest.json").exists():
            cycle.save(path, protocol)
            cycle.prepare(folder, path)
        elif cycle.read(folder / "protocol.json") != protocol:
            raise ValueError("Supplemental protocol differs from the frozen revision")
        for model in ("previous", "retrained"):
            if not (folder / f"{model}-primary-score.json").exists():
                cycle.predict(folder, model)
            result = splits.read(folder / f"{model}-primary-score.json")
            metrics = report.setdefault(task, {})
            if task == "body":
                for name, table in (("joints", result), ("wrists", result["wrists"])):
                    metrics.setdefault(name, {})[model] = [table["within0.05BodyScale"], table["referenceJoints"]]
            else:
                metrics.setdefault("batPairs", {})[model] = [result["bothEndpointsWithin0.1BatLength"], result["normalizableFrames"]]
    cycle.save(selection / "aggregate.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "score"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--previous-run", type=Path)
    args = parser.parse_args()
    run = cycle.output_path(args.run)
    if args.action == "freeze":
        if not args.previous_run:
            parser.error("--previous-run is required to freeze new-swings selection")
        freeze(run, args.previous_run.resolve())
    else:
        score(run)
