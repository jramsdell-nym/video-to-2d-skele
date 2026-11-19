"""Read-only audit for schema-2 body export plus separately reviewed bat points."""
import argparse
import csv
import hashlib
import json
import math
import sqlite3
import struct
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from audit_labels import HERE, LEGACY, sha256, write_json

JOINTS = ["Left shoulder", "Right shoulder", "Left elbow", "Right elbow", "Left wrist", "Right wrist",
          "Left hip", "Right hip", "Left knee", "Right knee", "Left ankle", "Right ankle"]


def count_by_source(records):
    return dict(Counter(p.get("source", "unspecified") for r in records for p in r["points"]))


def point_valid(p):
    return bool(p and all(isinstance(p.get(k), (int, float)) and not isinstance(p[k], bool)
                         and math.isfinite(p[k]) and 0 <= p[k] <= 1 for k in ("x", "y")))


def summarize(records):
    bat_eligible = [r for r in records if all(point_valid(r.get(name)) and r[name].get("visibility") == 2 for name in ("batTip", "batKnob")) and r["batKnob"].get("reviewedAt")]
    frame = [r for r in records if r.get("reviewMethod") == "frame"]
    no_interpolation = [r for r in frame if all(p.get("source") != "interpolated" for p in r["points"])]
    all_manual = [r for r in frame if all(p.get("source") == "manual" for p in r["points"])]
    return {"frames": len(records), "statuses": dict(Counter(r.get("status", "unknown") for r in records)),
            "reviewMethods": dict(Counter(r.get("reviewMethod", "unspecified") for r in records)),
            "pointSources": count_by_source(records),
            "pointVisibility": dict(Counter(p.get("visibility", "unknown") for r in records for p in r["points"])),
            "explicitFrameReview": {"frames": len(frame), "pointSources": count_by_source(frame),
                                    "noInterpolatedJoints": len(no_interpolation), "allManualJoints": len(all_manual)},
            "batEligibleVisibleReviewedPairs": len(bat_eligible),
            "batEligibleFrameReviewedManualPairs": sum(r.get("reviewMethod") == "frame" and all(r[n].get("source") == "manual" for n in ("batTip", "batKnob")) for r in bat_eligible),
            "batTipSources": dict(Counter((r.get("batTip") or {}).get("source", "missing") for r in records)),
            "batKnobSources": dict(Counter((r.get("batKnob") or {}).get("source", "missing") for r in records))}


def audit_export(dataset, sources_path=None):
    document = json.loads((dataset / "annotations.json").read_text())
    if document.get("schema") != 2 or document.get("joints") != JOINTS or document.get("sourceJointIndices") != list(range(5, 17)):
        raise ValueError("Expected schema-2 reduced 12-body-joint export; do not slice these points as 17 COCO slots")
    records = document["records"]
    sources = {r["uri"]: r for r in json.loads(sources_path.read_text())} if sources_path else {}
    submission_hashes = {}
    if sources_path:
        for source in sources.values():
            path = sources_path.parent / "submitted" / (source["id"] + ".json")
            if path.exists():
                submission = json.loads(path.read_text())
                source["clip"] = submission["clip"]
                submission_hashes[source["id"]] = sha256(path)
    flags = []
    images, video_groups, split_groups, source_deltas = [], defaultdict(list), defaultdict(list), defaultdict(list)
    identity = Counter((r["videoSHA256"], r["frameIndex"]) for r in records)
    def flag(r, reason, evidence, priority="review"):
        flags.append({"videoSHA256": r["videoSHA256"], "frameIndex": r["frameIndex"], "reason": reason,
                      "priority": priority, "evidence": evidence})
    for r in records:
        video_groups[r["videoSHA256"]].append(r)
        split_groups[r.get("split", "unknown")].append(r)
        if r.get("status") != "reviewed":
            flag(r, "unreviewed_export_record", "An export training record is not reviewed.", "block")
        if identity[(r["videoSHA256"], r["frameIndex"])] > 1:
            flag(r, "duplicate_frame_identity", "Multiple records share the same source video/frame.", "block")
        if len(r["points"]) != 12:
            flag(r, "wrong_body_joint_count", f"Expected 12, got {len(r['points'])}", "block")
            continue
        path = (dataset / r["image"]).resolve()
        if not path.is_relative_to(dataset.resolve()) or not path.exists():
            flag(r, "missing_or_escaped_image", r["image"], "block")
        else:
            with path.open("rb") as stream:
                header = stream.read(24)
            if header[:8] != b"\x89PNG\r\n\x1a\n" or len(header) < 24:
                flag(r, "invalid_png_header", r["image"], "block")
            else:
                width, height = struct.unpack(">II", header[16:24])
                if (width, height) != (r["width"], r["height"]):
                    flag(r, "png_dimension_mismatch", f"Image={width}x{height}; label={r['width']}x{r['height']}", "block")
            images.append({"videoSHA256": r["videoSHA256"], "frameIndex": r["frameIndex"], "path": r["image"], "sha256": sha256(path)})
        f = r.get("frame", {})
        try:
            valid_time = f["index"] == r["frameIndex"] and abs(float(f["pts"] * Fraction(f["timeBase"])) - f["time"]) <= 2e-6
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            valid_time = False
        if not valid_time:
            flag(r, "invalid_exact_timestamp", "Frame index/PTS/timebase/time disagrees.", "block")
        if sources:
            s = sources.get(r.get("source", {}).get("uri"))
            if not s or r.get("source", {}).get("generation") != s["generation"] or s.get("clip", {}).get("sha256") != r["videoSHA256"]:
                flag(r, "source_generation_mismatch", "Source SHA/URI/generation missing or differs from handoff.", "block")
            elif f != s["clip"]["frames"][r["frameIndex"]]:
                flag(r, "source_frame_mismatch", "Exact frame metadata differs from source catalog.", "block")
        box = r.get("box", [])
        for i, p in enumerate(r["points"]):
            if p.get("visibility") not in (0, 1, 2) or p.get("visibility") and not point_valid(p):
                flag(r, "invalid_joint", JOINTS[i], "block")
            elif p.get("visibility") and (len(box) != 4 or not box[0] <= p["x"] <= box[2] or not box[1] <= p["y"] <= box[3]):
                flag(r, "joint_outside_person_box", JOINTS[i], "block")
            people = (r.get("prediction") or {}).get("people", [])
            selected = r.get("person", 0)
            if isinstance(selected, int) and selected < len(people):
                original_points = people[selected].get("points", [])
                if len(original_points) == 17:
                    original = original_points[i+5]
                    if point_valid(p) and point_valid(original):
                        source_deltas[p.get("source", "unspecified")].append(math.hypot((p["x"]-original["x"])*r["width"], (p["y"]-original["y"])*r["height"]))
        label = dataset / "labels" / r.get("split", "") / (Path(r["image"]).stem + ".txt")
        if not label.exists():
            flag(r, "missing_yolo_label", str(label), "block")
        else:
            values = list(map(float, label.read_text().split()))
            expected = [0, (box[0]+box[2])/2, (box[1]+box[3])/2, box[2]-box[0], box[3]-box[1]]
            for p in r["points"]:
                expected.extend([p["x"], p["y"], p["visibility"]] if p["visibility"] else [0, 0, 0])
            if len(values) != len(expected) or any(not math.isfinite(x) or abs(x-y)>1e-6 for x, y in zip(values, expected)):
                flag(r, "yolo_target_mismatch", "Text label differs from record body order/box/coordinates.", "block")
        if not (r.get("batKnob") or {}).get("reviewedAt") or not point_valid(r.get("batKnob")):
            flag(r, "missing_reviewed_knob", "Body review is not sufficient to establish a bat knob target.", "coverage")
    duplicates = defaultdict(list)
    for item in images:
        duplicates[item["sha256"]].append(f"{item['videoSHA256']}:{item['frameIndex']}")
    duplicate_images = {digest: group for digest, group in duplicates.items() if len(group) > 1}
    overlap = {}
    for field in ("videoSHA256", "session", "trial"):
        groups = defaultdict(set)
        for r in records:
            groups[r[field]].add(r.get("split", "unknown"))
        overlap[field] = {k: sorted(v) for k,v in groups.items() if len(v) > 1}
    with sqlite3.connect(f"file:{LEGACY / 'annotations.sqlite3'}?mode=ro", uri=True) as connection:
        legacy = [json.loads(r[0]) for r in connection.execute("SELECT record FROM labels")]
    comparison = {field: sorted({r[field] for r in records} & {r[field] for r in legacy}) for field in ("videoSHA256", "session", "trial")}
    selections = {"frameReviewed": [], "frameReviewedNoInterpolatedJoints": [], "frameReviewedAllManualJoints": []}
    for r in records:
        if r.get("reviewMethod") == "frame":
            compact = {k: r[k] for k in ("videoSHA256", "frameIndex", "session", "split")}
            selections["frameReviewed"].append(compact)
            if all(p.get("source") != "interpolated" for p in r["points"]):
                selections["frameReviewedNoInterpolatedJoints"].append(compact)
            if all(p.get("source") == "manual" for p in r["points"]):
                selections["frameReviewedAllManualJoints"].append(compact)
    return {"schema": "body-export-quality-v1", "sourceManifest": str(dataset / "annotations.json"),
            "sourceManifestSHA256": sha256(dataset / "annotations.json"),
            "sourceLinkageManifest": str(sources_path) if sources_path else None,
            "sourceLinkageManifestSHA256": sha256(sources_path) if sources_path else None,
            "submissionSnapshotSHA256": submission_hashes,
            "summary": summarize(records), "byVideo": {k:summarize(v) for k,v in video_groups.items()},
            "bySplit": {k:summarize(v) for k,v in split_groups.items()},
            "duplicateImageHashes": duplicate_images, "sourceOverlapAcrossSplits": overlap,
            "sourceOverlapWithLegacyDatabase": comparison, "flags": flags,
            "flagCounts": dict(Counter(f["reason"] for f in flags)), "imageChecksums": images,
            "sensitivitySelections": selections,
            "savedLabelDisagreementWithOriginalPredictionByPointSource": {source: {"points":len(v), "exactMatches":sum(d < 1e-6 for d in v), "meanPixels":sum(v)/len(v)} for source,v in source_deltas.items()},
            "interpretation": ["Saved-label/original-prediction agreement is not independent model accuracy.",
                               "Frame-reviewed rows may retain interpolated or model-derived joints; per-point provenance remains relevant.",
                               "No-interpolation and all-manual subsets are sensitivity slices, not automatically independent gold labels.",
                               "Only two sessions and three videos: supplied split has validation but no reserved test.",
                               "All-visible body annotations do not prove the joints were visually observable.",
                               "PNG hash/dimension and exported metadata checks do not establish current cloud status or verify source MP4 pixel identity."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/tmp/bat-labeler-three-swing-training/dataset"))
    parser.add_argument("--sources", type=Path, default=Path("/tmp/bat-labeler-three-swing-training/training-sources.json"))
    parser.add_argument("--output", type=Path, default=HERE / "reports")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    result = audit_export(args.dataset, args.sources)
    write_json(args.output / "body-export-audit.json", result)
    with (args.output / "body-export-review-queue.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["priority", "reason", "videoSHA256", "frameIndex", "evidence"])
        writer.writeheader()
        writer.writerows(result["flags"])
    print(json.dumps({k: result[k] for k in ("summary", "flagCounts", "sourceOverlapAcrossSplits", "sourceOverlapWithLegacyDatabase")}, indent=2))


if __name__ == "__main__":
    main()
