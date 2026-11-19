"""Analyze three frozen external-session bat prediction files without model execution."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from analyze_temporal import analyze, distribution
from audit_labels import HERE, write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "reports" / "temporal")
    args = parser.parse_args()
    models = ("baseline", "candidate", "deployed")
    truth_path = args.experiment / "external-rows.json"
    paths = {m: args.experiment / f"{m}-external-predictions.json" for m in models}
    if not truth_path.exists() or not all(p.exists() for p in paths.values()):
        raise ValueError("All three frozen external prediction files and truth must exist; no inference is run here")
    rows = json.loads(truth_path.read_text())
    predictions = {m: json.loads(p.read_text()) for m, p in paths.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    comparison = {"schema": "temporal-bat-comparison-v1", "policies": {},
                  "note": "Development reference includes interpolation and bulk review. Confidence gates are not full iPhone policy parity. No fitting or GPU inference performed."}
    lines = ["Frozen external-session bat temporal diagnostics", "", comparison["note"], ""]
    for policy, thresholds in (("raw", (None, None)), ("confidence-gated", (.25, .25))):
        reports = {}
        for model in models:
            report = analyze(rows, predictions[model], *thresholds)
            report["inputs"] = {"truth": str(truth_path.resolve()), "truthSHA256": digest(truth_path),
                                "predictions": str(paths[model].resolve()), "predictionsSHA256": digest(paths[model]),
                                "analyzerSHA256": digest(HERE / "analyze_temporal.py")}
            write_json(args.output / f"{model}-{policy}.json", report)
            with (args.output / f"{model}-{policy}.review.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["videoSHA256", "clipId", "frameIndex", "endpoint", "reason", "evidence"])
                writer.writeheader()
                writer.writerows(report["reviewFlags"])
            reports[model] = report
        paired = {m: {(p["videoSHA256"], p["previousFrame"], p["frameIndex"], p["endpoint"]): p for p in r["residualPairs"]} for m, r in reports.items()}
        common = sorted(set.intersection(*(set(p) for p in paired.values())))
        summary = {"models": {m: {k: r[k] for k in ("frames", "consecutiveFramePairs", "missingPredictionRecords", "framesWithInterpolatedTruth", "dropouts", "residualChange", "flagCounts")} for m, r in reports.items()},
                   "commonObservedEndpointPairs": len(common),
                   "commonPairResidualDeltaPixels": {m: distribution([p[k]["residualDeltaPixels"] for k in common]) for m, p in paired.items()},
                   "commonPairResidualDeltaBatLengths": {m: distribution([p[k]["residualDeltaBatLengths"] for k in common if p[k]["residualDeltaBatLengths"] is not None]) for m, p in paired.items()}}
        comparison["policies"][policy] = summary
        lines.extend([policy, f"Common observed endpoint pairs across all models: {len(common)}"])
        for model, r in reports.items():
            drop = r["dropouts"]["eitherEndpoint"]
            p95 = summary["commonPairResidualDeltaPixels"][model]["p95"]
            lines.append(f"{model}: missing either-endpoint output {drop['missingFrames']}/{drop['eligibleFrames']}; longest observed dropout span={drop['longestObservedSpanSeconds']} s; common-pair residual-delta P95={p95} px; flags={r['flagCounts']}")
        lines.append("")
    lines.extend(["Dropout spans use first-to-last missing presentation PTS and are lower bounds; boundary/gap runs may be censored.",
                  "Residual changes compare prediction-minus-reference error; true correct rapid motion cancels. Low conditional jitter alone is not evidence of accuracy.",
                  "Review flags may reflect model errors, interpolation/reference errors, or both; they are never automatic corrections."])
    write_json(args.output / "comparison.json", comparison)
    (args.output / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
