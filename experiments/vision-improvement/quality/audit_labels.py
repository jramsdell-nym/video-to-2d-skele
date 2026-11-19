"""Read-only provenance, geometry, temporal and split audit of legacy bat labels.

Flags request human inspection. They do not delete, repair, or relabel data.
Run with Python 3; no model or GPU dependencies.
"""
import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from score_bat import directed_angle, finite_xy, key, percentile, score, truth_xy

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parents[1] / "2d-tracking" / "labeler"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def split_audit(rows, assignments, clips):
    """assignments maps clipId to split; identify shared exact media and groups."""
    fields = ("videoSHA256", "session", "trial", "athlete")
    groups = {field: defaultdict(set) for field in fields}
    for row in rows:
        clip_id = row["clipId"]
        if clip_id not in assignments:
            continue
        clip = clips.get(clip_id, {})
        values = {"videoSHA256": row.get("videoSHA256") or clip.get("sha256"),
                  "session": row.get("session") or clip.get("session"),
                  "trial": row.get("trial") or clip.get("trial"),
                  "athlete": clip.get("title")}
        for field, value in values.items():
            if value:
                groups[field][value].add(assignments[clip_id])
    return {field: {k: sorted(v) for k, v in values.items() if len(v) > 1}
            for field, values in groups.items()}


def timestamp(row):
    source = row.get("source", {})
    try:
        time = float(source["pts"] * Fraction(source["timeBase"]))
        if not math.isfinite(time) or abs(time - source["time"]) > 1e-6:
            return None
        return time
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def audit(rows, clips):
    flags = []
    def flag(row, reason, evidence, priority="review"):
        flags.append({"clipId": row["clipId"], "frameIndex": row["frameIndex"],
                      "title": clips.get(row["clipId"], {}).get("title", "unknown"),
                      "status": row.get("status"), "reason": reason,
                      "priority": priority, "evidence": evidence})
    unique = defaultdict(list)
    per_clip = defaultdict(list)
    lengths = []
    coordinate_valid = {}
    identical_assistance = []
    for row in rows:
        clip = clips.get(row["clipId"])
        per_clip[row["clipId"]].append(row)
        unique[(row.get("videoSHA256"), row["frameIndex"])].append(row)
        if not clip:
            flag(row, "missing_clip_catalog", "Cannot locate source media.", "block")
            continue
        source = row.get("source", {})
        index = row["frameIndex"]
        if row.get("videoSHA256") != clip.get("sha256"):
            flag(row, "source_hash_mismatch", "Label/video catalog hashes differ.", "block")
        if not 0 <= index < len(clip.get("frames", [])):
            flag(row, "frame_out_of_range", "Frame is absent from catalog.", "block")
        elif source != clip["frames"][index]:
            flag(row, "source_frame_mismatch", "Saved PTS/time/index differs from exact frame catalog.", "block")
        if (row.get("imageWidth"), row.get("imageHeight")) != (clip.get("width"), clip.get("height")):
            flag(row, "image_dimensions_mismatch", "Label coordinate dimensions differ from media.", "block")
        if timestamp(row) is None:
            flag(row, "invalid_timebase", "PTS*timeBase is missing or disagrees with source time.", "block")
        points = row.get("points", {})
        valid = True
        for name in ("tip", "knob"):
            point = points.get(name, {})
            state = point.get("state")
            if state not in ("visible", "unmarked", "occluded", "uncertain"):
                flag(row, "unknown_point_state", f"{name} state={state}", "block")
            if state == "visible":
                normalized = [point.get("x"), point.get("y")]
                pixel = [point.get("pixelX"), point.get("pixelY")]
                if not finite_xy(pixel) or not finite_xy(normalized):
                    flag(row, "invalid_coordinate", f"{name} is visible but coordinates are not finite.", "block")
                    valid = False
                elif not all(0 <= x <= 1 for x in normalized):
                    flag(row, "out_of_image", f"{name} normalized coordinates={normalized}", "block")
                    valid = False
                elif math.dist(pixel, [normalized[0]*clip["width"], normalized[1]*clip["height"]]) > .01:
                    flag(row, "pixel_normalized_mismatch", f"{name} normalized and pixel coordinates disagree.", "block")
                    valid = False
        complete = all(points.get(name, {}).get("state") == "visible" for name in ("tip", "knob"))
        if complete and valid:
            xy = [truth_xy(row, name) for name in ("tip", "knob")]
            coordinate_valid[key(row)] = xy
            length = math.dist(*xy)
            lengths.append(length)
            if length < 10:
                flag(row, "very_short_projected_bat", f"Length={length:.2f} px; may be end-on view, blur, or coincident endpoints.")
        if row.get("status") == "draft":
            flag(row, "draft_requires_review", "Excluded from strict training and evaluation; complete coordinates do not imply review.", "workflow")
        assistance = row.get("assistance", {}).get("suggestedPoints", {})
        if assistance and complete and valid:
            differences = [math.hypot(points[n]["x"]-assistance.get(n, {}).get("x", -100),
                                      points[n]["y"]-assistance.get(n, {}).get("y", -100)) for n in ("tip", "knob")]
            if max(differences) < 1e-8:
                identical_assistance.append(f"{row['clipId']}:{index}")
                flag(row, "accepted_prediction_unchanged", "Both saved endpoints exactly match the suggestion. Audit a blind sample; this is not itself an error.", "provenance")
    duplicate_groups = []
    for identity, group in unique.items():
        if len(group) > 1:
            duplicate_groups.append({"videoSHA256": identity[0], "frameIndex": identity[1], "clips": [r["clipId"] for r in group]})
            for row in group:
                flag(row, "duplicate_exact_frame", "Another label record uses the same video hash and frame index.", "block")
    temporal_pairs, temporal_triples = 0, 0
    for clip_id, group in per_clip.items():
        group = sorted([r for r in group if key(r) in coordinate_valid and timestamp(r) is not None], key=lambda r: r["frameIndex"])
        for a, b in zip(group, group[1:]):
            dt = timestamp(b) - timestamp(a)
            if dt <= 0:
                flag(b, "nonincreasing_timestamps", f"Previous frame={a['frameIndex']}, dt={dt}", "block")
                continue
            if dt > .05:
                continue
            temporal_pairs += 1
            pa, pb = coordinate_valid[key(a)], coordinate_valid[key(b)]
            angle = directed_angle(pa, pb)
            direct = sum(math.dist(pa[i], pb[i]) for i in (0, 1))
            reverse = sum(math.dist(pa[i], pb[1-i]) for i in (0, 1))
            if angle is not None and angle > 120 and reverse < .5 * direct:
                flag(b, "possible_temporal_endpoint_swap", f"Previous frame={a['frameIndex']}, dt={dt:.5f}s, direction change={angle:.1f}deg, swapped/direct movement={reverse/direct:.3f}. Real motion is possible.")
        for a, b, c in zip(group, group[1:], group[2:]):
            t0, t1, t2 = map(timestamp, (a, b, c))
            if not 0 < t1-t0 <= .1 or not 0 < t2-t1 <= .1:
                continue
            temporal_triples += 1
            pa, pb, pc = [coordinate_valid[key(r)] for r in (a, b, c)]
            fraction = (t1-t0)/(t2-t0)
            errors = [math.dist(pb[i], [pa[i][d]+fraction*(pc[i][d]-pa[i][d]) for d in (0, 1)]) for i in (0, 1)]
            cutoff = max(60, .75*math.dist(*pb))
            if max(errors) > cutoff:
                flag(b, "temporal_interpolation_outlier", f"Neighbors={a['frameIndex']},{c['frameIndex']}; tip/knob residual={errors[0]:.1f}/{errors[1]:.1f}px; cutoff={cutoff:.1f}px. Inspect swing acceleration and occlusion before changing.")
    return {"flags": flags, "duplicateFrameGroups": duplicate_groups,
            "acceptedPredictionsUnchanged": identical_assistance,
            "temporalPairsChecked": temporal_pairs, "temporalTriplesChecked": temporal_triples,
            "lengthPixels": {"count": len(lengths), "minimum": min(lengths) if lengths else None,
                             "median": percentile(lengths, .5), "p95": percentile(lengths, .95),
                             "maximum": max(lengths) if lengths else None}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeler", type=Path, default=LEGACY)
    parser.add_argument("--output", type=Path, default=HERE / "reports")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    db = args.labeler / "annotations.sqlite3"
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as connection:
        connection.execute("BEGIN")
        records = [json.loads(r[0]) for r in connection.execute("SELECT record FROM labels ORDER BY clip,frame")]
        event_count = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    clips = {c["id"]: c for c in json.loads((args.labeler / "clips.json").read_text())["clips"]}
    result = audit(records, clips)
    visual_path = args.output / "visual-adjudications.json"
    visual_review = []
    if visual_path.exists():
        for item in json.loads(visual_path.read_text()).get("confirmedIssues", []):
            matching = [r for r in records if r["clipId"] == item["clipId"] and r["frameIndex"] == item["frameIndex"]]
            current = hashlib.sha256(json.dumps(matching[0], sort_keys=True, separators=(",", ":")).encode()).hexdigest() if matching else None
            visual_review.append({**item, "matchesCurrentRecord": current == item.get("originalRecordSHA256")})
    reviewed = [r for r in records if r["status"] == "reviewed"]
    complete = [r for r in reviewed if all(r["points"][n]["state"] == "visible" for n in ("tip", "knob"))]
    media = []
    for c in clips.values():
        path = Path(c["path"])
        digest = sha256(path) if path.exists() else None
        labels = [r for r in records if r["clipId"] == c["id"]]
        media.append({"clipId": c["id"], "title": c["title"], "session": c["session"],
                      "videoSHA256": c["sha256"], "exists": path.exists(), "hashMatches": digest == c["sha256"],
                      "catalogFrames": len(c["frames"]), "savedLabels": len(labels),
                      "statuses": dict(Counter(r["status"] for r in labels)),
                      "reviewedVisiblePairs": sum(r in complete for r in labels)})
    active = json.loads((args.labeler / "models" / "active.json").read_text())
    active_clips = set(active.get("clipCounts", {}))
    active_keys = set(active.get("trainingKeys", []))
    historical = []
    for manifest_path in sorted((args.labeler / "models").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        run = args.labeler / "training" / manifest["id"]
        split = {clip_id: "validation" if clip_id == manifest["holdoutClipId"] else "train"
                 for clip_id in manifest["clipCounts"]}
        snapshot = run / "snapshot.json"
        old_rows = json.loads(snapshot.read_text())["labels"] if snapshot.exists() else []
        evaluation = manifest.get("heldoutEvaluation", {})
        scored = None
        if old_rows and evaluation.get("details"):
            val_rows = [r for r in old_rows if r["clipId"] == manifest["holdoutClipId"]]
            # Historical opt-in drafts are never silently upgraded to reviewed truth.
            scored = score(val_rows, evaluation["details"])
            write_json(args.output / f"{manifest['id']}-historical-score.json", scored)
        historical.append({"id": manifest["id"], "usableLabels": manifest["usableLabels"],
                           "includedCompleteManualDrafts": manifest.get("includedCompleteManualDrafts", 0),
                           "holdoutClipId": manifest["holdoutClipId"],
                           "initialSplitSharedGroups": split_audit(old_rows, split, clips),
                           "finalRefitContainsFormerHoldout": any(k.startswith(manifest["holdoutClipId"]+":") for k in manifest.get("trainingKeys", [])),
                           "scorePrimary": scored.get("bothEndpointsPCK0.1BatLength") if scored else None,
                           "note": "Stored pre-refit checkpoint-validation predictions; not a new test or final-model accuracy estimate."})
    report = {"schema": "bat-label-audit-v1", "createdAt": datetime.now(timezone.utc).isoformat(),
              "sourceDatabase": str(db), "recordContentSHA256": hashlib.sha256(canonical).hexdigest(),
              "catalogSHA256": sha256(args.labeler / "clips.json"), "labels": len(records), "events": event_count,
              "statuses": dict(Counter(r["status"] for r in records)),
              "origins": dict(Counter(r.get("origin", "unknown") for r in records)),
              "pointStates": {n: dict(Counter(r["points"][n]["state"] for r in records)) for n in ("tip", "knob")},
              "reviewedVisiblePairs": len(complete),
              "reviewedVisiblePairsBySession": dict(Counter(r["session"] for r in complete)),
              "reviewedVisiblePairsByCamera": dict(Counter(r["camera"] for r in complete)),
              "reviewedAssisted": sum(bool(r.get("assistance")) for r in reviewed),
              "explicitAbsentNegatives": sum(r["status"] == "absent" for r in records),
              "bodyLabelsInLegacyDatabase": 0,
              "bodyLabelScope": "Legacy DB schema has only tip/knob. Browser IndexedDB/cloud reviewed body poses were not acquired by this audit.",
              "activeModel": {"id": active["id"], "recordedTrainingLabels": active["usableLabels"],
                              "currentReviewedPairsOnSeenVideos": sum(r["clipId"] in active_clips for r in complete),
                              "currentReviewedPairsOnUnseenVideos": sum(r["clipId"] not in active_clips for r in complete),
                              "currentReviewedPairsAtRecordedTrainingKeys": sum(f"{r['clipId']}:{r['frameIndex']}" in active_keys for r in complete)},
              "catalog": media, "historicalModels": historical, **result,
              "visualAdjudications": visual_review,
              "flagCounts": dict(Counter(f["reason"] for f in result["flags"])),
              "cautions": ["Flags are review candidates, never automatic deletions.",
                           "Video hashes prevent exact-file leakage, not adjacent excerpt or same-session leakage.",
                           "All labeled rows share one session: genuine unseen-session accuracy remains unmeasured.",
                           "Absent negatives, hidden endpoint localization and independent body labels are not available in this local DB.",
                           "Projected length changes and temporal acceleration may be genuine 3D swing motion."]}
    write_json(args.output / "label-audit.json", report)
    write_json(args.output / "labels-readonly-snapshot.json", {"recordContentSHA256": report["recordContentSHA256"], "labels": records})
    with (args.output / "review-queue.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["priority", "reason", "clipId", "title", "frameIndex", "status", "evidence"])
        writer.writeheader()
        writer.writerows(sorted(result["flags"], key=lambda f: (f["priority"], f["clipId"], f["frameIndex"])))
    summary = ["Fan iPhone label quality audit", "", f"Read-only record hash: {report['recordContentSHA256']}",
               f"Saved rows: {len(records)}; statuses: {report['statuses']}",
               f"Strict reviewed, visible tip+knob pairs: {len(complete)}", f"Origins: {report['origins']}",
               f"Reviewed model-assisted rows: {report['reviewedAssisted']}; unchanged suggestions: {len(result['acceptedPredictionsUnchanged'])}",
               f"Explicit absent negatives: {report['explicitAbsentNegatives']}; body label count in legacy DB: 0",
               f"Media: {len(media)}; all hash-verified: {all(m['hashMatches'] for m in media)}",
               f"Eligible sessions: {report['reviewedVisiblePairsBySession']}; cameras: {report['reviewedVisiblePairsByCamera']}",
               f"Flags: {report['flagCounts']}", "", "Audit boundaries:"] + report["cautions"]
    (args.output / "summary.txt").write_text("\n".join(summary) + "\n")
    print(json.dumps({k: report[k] for k in ("labels", "statuses", "reviewedVisiblePairs", "flagCounts", "activeModel")}, indent=2))


if __name__ == "__main__":
    main()
