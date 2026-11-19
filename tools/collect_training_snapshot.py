"""Run the audited labeler snapshot stages into a separate private output directory."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path


def private_output(path, repository):
    path = path.resolve()
    base = (repository / "outputs").resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError("Private snapshots must use a subdirectory of this repository's ignored outputs directory")
    return path


def compare_snapshots(current, previous):
    """Compare supervision, ignoring bookkeeping and decoded-image locations."""
    def index(folder, task):
        rows = json.loads((folder / f"{task}-rows-unresolved.json").read_text())
        result = {(r["videoSHA256"], r["frameIndex"]): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Snapshot contains duplicate frame identities")
        return result

    def supervision(row, task):
        def point(p):
            return {k: p.get(k) for k in ("x", "y", "visibility", "state")}
        points = row["points"]
        targets = [point(p) for p in points] if task == "body" else {k: point(v) for k, v in points.items()}
        return {"points": targets, "box": row.get("box"), "frame": row["frame"],
                "width": row["width"], "height": row["height"]}

    report = {}
    for task in ("body", "bat"):
        before, after = index(previous, task), index(current, task)
        old_keys, new_keys = set(before), set(after)
        revised = sum(supervision(before[k], task) != supervision(after[k], task) for k in old_keys & new_keys)
        report[task] = {"previousFrames": len(before), "currentFrames": len(after),
                        "netAddedFrames": len(after) - len(before), "larger": len(after) > len(before),
                        "newFrameIdentities": len(new_keys - old_keys),
                        "removedFrameIdentities": len(old_keys - new_keys), "revisedExistingTargets": revised,
                        "previousVideos": len({k[0] for k in before}),
                        "currentVideos": len({k[0] for k in after})}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fetch", "audit", "images"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-to", type=Path, help="Prior private snapshot directory for current-versus-prior counts after audit")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    source = repository / "experiments/vision-improvement/cycles/2026-11-18/dataset/snapshot.py"
    spec = importlib.util.spec_from_file_location("audited_snapshot", source)
    snapshot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(snapshot)
    snapshot.ROOT = private_output(args.output, repository)
    snapshot.ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "audit" and args.compare_to:
        with contextlib.redirect_stdout(io.StringIO()):
            snapshot.audit()
    else:
        {"fetch": snapshot.fetch, "audit": snapshot.audit, "images": snapshot.images}[args.stage]()
    if args.stage == "audit" and args.compare_to:
        comparison = compare_snapshots(snapshot.ROOT, args.compare_to.resolve())
        snapshot.save(snapshot.ROOT / "comparison-to-previous.json", comparison)
        inventory_path = snapshot.ROOT / "inventory.json"
        inventory = snapshot.read(inventory_path)
        # The archived auditor compared with Nov11. Make the current report
        # compare with the explicitly selected snapshot instead.
        for task in ("body", "bat"):
            title = task.title()
            inventory[f"prior{title}Frames"] = comparison[task]["previousFrames"]
            inventory[f"{task}LargerThanPriorSnapshot"] = comparison[task]["larger"]
        inventory["comparisonToPrevious"] = comparison
        snapshot.save(inventory_path, inventory)
        print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
