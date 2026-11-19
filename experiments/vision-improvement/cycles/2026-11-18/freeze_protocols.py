"""Freeze the preselected session split after source/label review is complete."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    assert not path.exists(), f"Refusing to replace {path}"
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    plan = read(ROOT / "partition-plan.json")
    selection = read(ROOT / "videos/frozen-selection.json")
    templates = {task: read(ROOT / f"training/{task}-protocol-template.json") for task in ("body", "bat")}
    exclusion_videos = sorted(set().union(*(set(t["evaluationExclusionVideos"]) for t in templates.values())))
    exclusion_groups = sorted(set().union(*(set(t["evaluationExclusionGroups"]) for t in templates.values())))
    receipt = {"frozenAt": datetime.now(timezone.utc).isoformat(), "planSHA256": sha(ROOT / "partition-plan.json"),
               "selectionSHA256": sha(ROOT / "videos/frozen-selection.json"), "tasks": {}, "files": {}}
    for task in ("body", "bat"):
        rows_path = ROOT / f"dataset/{task}-rows.json"
        rows = read(rows_path)
        partitions = {"train": [], "val": [], "evaluation": []}
        for row in rows:
            assert row.get("imagePath") and Path(row["imagePath"]).is_file()
            assert row.get("imageSHA256")
            split = "evaluation" if row["session"] in plan["evaluationSessions"] else "val" if row["session"] in plan["validationSessions"] else "train"
            partitions[split].append(row)
        assert set(r["assignmentId"] for r in partitions["evaluation"]) == set(plan["evaluationAssignments"])
        assert set(r["videoSHA256"] for r in partitions["evaluation"]) == set(c["source"]["sha256"] for c in selection["clips"])
        for split, values in partitions.items():
            assert values
            assert len(set((r["videoSHA256"], r["frameIndex"]) for r in values)) == len(values)
            for other, other_values in partitions.items():
                if split != other:
                    for field in ("session", "videoSHA256", "canonicalGroup"):
                        assert not {r[field] for r in values} & {r[field] for r in other_values}
        assert not set(exclusion_videos) & {r["videoSHA256"] for r in partitions["evaluation"]}
        assert not set(exclusion_groups) & {r["canonicalGroup"] for r in partitions["evaluation"]}
        receipt["tasks"][task] = {}
        protocol = templates[task]
        protocol.update(approvedForTraining=True, evaluationExclusionVideos=exclusion_videos,
                        evaluationExclusionGroups=exclusion_groups,
                        splitPlanSHA256=receipt["planSHA256"], selectionSHA256=receipt["selectionSHA256"])
        protocol["limits"] += plan["limits"]
        for split, values in partitions.items():
            path = ROOT / f"dataset/splits/{task}-{split}-rows.json"
            write_new(path, values)
            protocol["rows"][split] = str(path)
            receipt["files"][str(path)] = sha(path)
            receipt["tasks"][task][split] = {
                "frames": len(values), "videos": len({r["videoSHA256"] for r in values}),
                "sessions": sorted({r["session"] for r in values}),
                "reviewMethods": dict(Counter(r.get("reviewMethod", "unknown") for r in values)),
                "knownAthletes": sorted({r["athleteId"] for r in values if r.get("athleteId")}),
                "framesMissingAthleteId": sum(not r.get("athleteId") for r in values)}
        path = ROOT / f"training/{task}-protocol.json"
        write_new(path, protocol)
        receipt["files"][str(path)] = sha(path)
        receipt["files"][str(rows_path)] = sha(rows_path)
    write_new(ROOT / "split-receipt.json", receipt)
    print(json.dumps(receipt["tasks"], indent=2))
