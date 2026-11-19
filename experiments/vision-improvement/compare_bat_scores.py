"""Compare two miss-aware score_bat.py reports on exactly the same frame set.

This recommends additional evaluation only. It never publishes model weights.
Split integrity, independence, and inference-policy equality must be verified
from the associated experiment manifests, not inferred from these statistics.
"""
import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def identity(row):
    return row["clipId"], row["frameIndex"]


def indexed(report):
    result = {}
    for row in report["details"]:
        key = identity(row)
        if key in result:
            raise ValueError(f"Duplicate frame: {key}")
        result[key] = row
    return result


def eligible(row):
    return (row["kind"] == "positive" and row["bothTruthVisible"]
            and row["truthLengthPixels"] > 1e-8)


def passed(row):
    return (eligible(row) and len(row["errorsPixels"]) == 2
            and all(e is not None and math.isfinite(e) and e >= 0
                    and e <= .1 * row["truthLengthPixels"]
                    for e in row["errorsPixels"]))


def compare(baseline, candidate):
    a, b = indexed(baseline), indexed(candidate)
    if not a or a.keys() != b.keys():
        raise ValueError("Baseline and candidate must score the same nonempty frame set")
    groups = defaultdict(list)
    for key in sorted(a):
        for field in ("kind", "bothTruthVisible", "truthLengthPixels", "visibleTruthEndpoints", "videoSHA256", "session", "camera"):
            if a[key].get(field) != b[key].get(field):
                raise ValueError(f"Ground-truth mismatch for {key}: {field}")
        if eligible(a[key]):
            groups[a[key].get("videoSHA256") or key[0]].append(key)
    if not groups:
        raise ValueError("No complete nondegenerate visible pairs to compare")
    keys = [key for group in groups.values() for key in group]

    def summary(selected):
        wins = sum(not passed(a[k]) and passed(b[k]) for k in selected)
        losses = sum(passed(a[k]) and not passed(b[k]) for k in selected)
        na, nb = sum(passed(a[k]) for k in selected), sum(passed(b[k]) for k in selected)
        return {"frames": len(selected), "baselinePassed": na, "candidatePassed": nb,
                "baselineRate": na / len(selected), "candidateRate": nb / len(selected),
                "deltaPercentagePoints": 100 * (nb - na) / len(selected),
                "candidateWins": wins, "candidateLosses": losses,
                "bothPass": sum(passed(a[k]) and passed(b[k]) for k in selected),
                "bothFail": sum(not passed(a[k]) and not passed(b[k]) for k in selected)}

    overall = summary(keys)
    videos = {name: summary(group) for name, group in groups.items()}
    macro_a = statistics.mean(v["baselineRate"] for v in videos.values())
    macro_b = statistics.mean(v["candidateRate"] for v in videos.values())
    reasons = []
    if overall["candidateRate"] <= overall["baselineRate"]:
        reasons.append("No improvement in the all-frame primary success rate")
    if macro_b <= macro_a:
        reasons.append("No improvement in the equal-video primary success rate")
    for field in ("missedFrames", "missingEndpoints", "swapCandidates"):
        if candidate[field] > baseline[field]:
            reasons.append(f"Regression in {field}: {baseline[field]} to {candidate[field]}")
    if any(v["candidateRate"] < v["baselineRate"] for v in videos.values()):
        reasons.append("At least one video regressed on primary success")
    negatives = {"baseline": baseline.get("negativeFalsePositiveRate"),
                 "candidate": candidate.get("negativeFalsePositiveRate")}
    if (negatives["baseline"] is not None and negatives["candidate"] is not None
            and negatives["candidate"] > negatives["baseline"]):
        reasons.append("Regression on reviewed negative frames")
    return {"schema": "paired-bat-comparison-v1",
            "metric": "both ordered endpoints within 0.10 ground-truth projected bat length; misses fail",
            "overall": overall, "byVideo": videos,
            "macroVideo": {"baselineRate": macro_a, "candidateRate": macro_b,
                           "deltaPercentagePoints": 100 * (macro_b - macro_a)},
            "negativeFalsePositiveRate": negatives,
            "recommendation": "investigate_tradeoffs" if reasons else "advance_to_broader_evaluation",
            "reasons": reasons,
            "releaseAuthorized": False,
            "limitations": ["One comparison cannot establish statistical significance or a best model.",
                            "Adjacent frames are correlated; no frame-level confidence interval is reported.",
                            "Verify source hashes, split independence, model initialization and inference policies in manifests.",
                            "Unknown negative, occlusion, 3D and device behavior remain unevaluated; this is not a release gate."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy", required=True, help="Name of common inference policy from experiment manifests")
    args = parser.parse_args()
    report = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    report["declaredCommonInferencePolicy"] = args.policy
    report["inputs"] = {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for name, path in (("baseline", args.baseline), ("candidate", args.candidate))}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"recommendation": report["recommendation"], **report["overall"]}))


if __name__ == "__main__":
    main()
