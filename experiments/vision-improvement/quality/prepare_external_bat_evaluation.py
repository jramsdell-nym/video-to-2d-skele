"""Freeze existing reviewed November-9 bat pairs for evaluation, never training."""
import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from audit_labels import HERE, LEGACY, sha256, write_json
from audit_pose_export import audit_export, point_valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/tmp/bat-labeler-three-swing-training/dataset"))
    parser.add_argument("--sources", type=Path, default=Path("/tmp/bat-labeler-three-swing-training/training-sources.json"))
    parser.add_argument("--output", type=Path, default=HERE / "reports")
    parser.add_argument("--session", default="session_2026-11-09_13-19")
    args = parser.parse_args()
    audited = audit_export(args.dataset, args.sources)
    if any(f["priority"] == "block" for f in audited["flags"]):
        raise ValueError("Fix blocking export audit problems before freezing evaluation")
    source = json.loads((args.dataset / "annotations.json").read_text())
    records = [r for r in source["records"] if r["session"] == args.session and r["status"] == "reviewed"
               and all(point_valid(r.get(n)) and r[n].get("visibility") == 2 for n in ("batTip", "batKnob"))
               and r["batKnob"].get("reviewedAt")]
    if not records:
        raise ValueError("No eligible external-session pairs")
    with sqlite3.connect(f"file:{LEGACY / 'annotations.sqlite3'}?mode=ro", uri=True) as connection:
        legacy = [json.loads(r[0]) for r in connection.execute("SELECT record FROM labels")]
    overlap = {field: sorted({r[field] for r in records} & {r[field] for r in legacy}) for field in ("videoSHA256", "session", "trial")}
    if any(overlap.values()):
        raise ValueError(f"External evaluation overlaps legacy dataset: {overlap}")
    images = {(r["videoSHA256"], r["frameIndex"]): r for r in audited["imageChecksums"]}
    rows, image_paths = [], {}
    for r in records:
        row = {k: r[k] for k in ("videoSHA256", "frameIndex", "session", "trial", "camera", "status", "reviewedAt") if k in r}
        row.update(clipId=r["videoSHA256"][:20], imageWidth=r["width"], imageHeight=r["height"],
                   source=r["frame"], sourceVideo=r["source"], reviewMethod=r.get("reviewMethod", "unspecified"),
                   origin="native_export_" + r.get("reviewMethod", "unspecified"),
                   imagePath=str((args.dataset / r["image"]).resolve()),
                   imageSHA256=images[(r["videoSHA256"], r["frameIndex"])]["sha256"],
                   annotationProvenance={"tipSource": r["batTip"].get("source", "unspecified"),
                                         "knobSource": r["batKnob"].get("source", "unspecified"),
                                         "knobReviewedAt": r["batKnob"]["reviewedAt"],
                                         "reviewMethod": r.get("reviewMethod", "unspecified")}, points={})
        for endpoint, native in (("tip", "batTip"), ("knob", "batKnob")):
            p = r[native]
            row["points"][endpoint] = {**p, "state": "visible", "pixelX": p["x"]*r["width"], "pixelY": p["y"]*r["height"]}
        row["bothEndpointsManual"] = all(row["points"][n].get("source") == "manual" for n in ("tip", "knob"))
        row["eitherEndpointInterpolated"] = any(row["points"][n].get("source") == "interpolated" for n in ("tip", "knob"))
        image_paths[f"{row['clipId']}:{row['frameIndex']}"] = row["imagePath"]
        rows.append(row)
    manifest = {"schema": "external-session-bat-evaluation-v1", "purpose": "Frozen-model external-session development evaluation; never fit or tune on these rows.",
                "truthLimitations": "Human-reviewed export contains model-assisted/interpolated targets. This is provisional supervision, not independent blind gold. No negative examples.",
                "sourceManifest": str(args.dataset / "annotations.json"), "sourceManifestSHA256": audited["sourceManifestSHA256"],
                "sourceLinkageManifestSHA256": audited["sourceLinkageManifestSHA256"], "submissionSnapshotSHA256": audited["submissionSnapshotSHA256"],
                "session": args.session, "legacyOverlap": overlap, "endpointOrder": ["tip", "knob"],
                "counts": {"rows": len(rows), "videos": len({r['videoSHA256'] for r in rows}),
                           "reviewMethods": dict(Counter(r["reviewMethod"] for r in rows)),
                           "bothEndpointsManual": sum(r["bothEndpointsManual"] for r in rows),
                           "eitherEndpointInterpolated": sum(r["eitherEndpointInterpolated"] for r in rows),
                           "frameReviewedBothManual": sum(r["bothEndpointsManual"] and r["reviewMethod"] == "frame" for r in rows)},
                "rows": rows, "imagePaths": image_paths}
    args.output.mkdir(exist_ok=True, parents=True)
    write_json(args.output / "new-session-bat-truth.json", manifest)
    write_json(args.output / "new-session-bat-image-index.json", {f"{r['clipId']}:{r['frameIndex']}": {"path": r["imagePath"], "sha256": r["imageSHA256"]} for r in rows})
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
