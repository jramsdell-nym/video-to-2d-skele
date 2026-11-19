"""Ground-truth ORACLE selection among saved Vision people; never deploy these choices.

No inference or fitting. The oracle maximizes visible-joint PCK.05 at confidence
>= .3 using full-reference video scales. It may switch identity each frame.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_SCORER = HERE.parent / "research/body-experiment/score_body.py"


def load_scorer(path=DEFAULT_SCORER):
    spec = importlib.util.spec_from_file_location("body_scorer_for_oracle_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def passing_joints(row, candidate, scale, scorer):
    """Match the existing scorer's visible, finite, in-image, .3-confidence gate."""
    points, confidences = candidate.get("prediction"), candidate.get("keypointConfidences")
    if not isinstance(points, list) or len(points) != 12 or not isinstance(confidences, list) or len(confidences) != 12:
        raise ValueError("Each candidate requires twelve ordered points and confidences")
    passed = 0
    for truth, point, confidence in zip(row["points"], points, confidences):
        if truth["visibility"] != 2:
            continue
        usable = (scorer.xy(point) and scorer.finite(confidence) and confidence >= .3
                  and 0 <= point[0] <= row["width"] and 0 <= point[1] <= row["height"])
        if usable and math.dist(point, (truth["x"] * row["width"], truth["y"] * row["height"])) <= .05 * scale:
            passed += 1
    return passed


def choose_candidate(row, candidates, scale, scorer):
    counts = [passing_joints(row, c, scale, scorer) for c in candidates]
    # max retains the first candidate on ties, including all-zero candidates.
    index = max(range(len(counts)), key=counts.__getitem__) if counts else None
    return index, counts


def slices(score, scorer):
    visible = [x for x in score["jointDetails"] if x["visibility"] == 2]
    strict = [x for x in visible if x["reviewMethod"] == "frame" and x["noInterpolatedJointsInFrame"]]
    def describe(items):
        return {"frames": len({(x["videoSHA256"], x["frameIndex"]) for x in items}), **scorer.summarize(items)}
    return {"all": describe(visible), "wristsUnderSameBodySelection": describe([x for x in visible if x["jointIndex"] in (4, 5)]),
            "frameReviewedNoInterpolation": describe(strict),
            "frameReviewedNoInterpolationWrists": describe([x for x in strict if x["jointIndex"] in (4, 5)]),
            "byVideo": score["byVideo"]}


def analyze(rows, all_candidates, current, scorer):
    def lookup(records):
        out = {scorer.key(r): r for r in records}
        if len(out) != len(records):
            raise ValueError("Duplicate frame identities")
        return out
    truth_lookup, candidate_lookup, current_lookup = map(lookup, (rows, all_candidates, current))
    if truth_lookup.keys() != candidate_lookup.keys() or truth_lookup.keys() != current_lookup.keys():
        raise ValueError("Truth, candidate and current-policy frame identities must match exactly")
    current_score = scorer.score(rows, current, .3)  # Also validates reference schema.
    scales = current_score["bodyScalePixelsByVideo"]
    oracle_predictions, choices = [], []
    for row in rows:
        key = scorer.key(row)
        raw, existing = candidate_lookup[key], current_lookup[key]
        candidates = raw["candidates"]
        if raw["candidateCount"] != len(candidates):
            raise ValueError("Candidate count mismatch")
        if bool(existing.get("detected")) != (len(candidates) == 1):
            raise ValueError("Current record is not the declared exactly-one-person policy")
        if len(candidates) == 1 and any(existing.get(k) != candidates[0].get(k) for k in ("prediction", "keypointConfidences")):
            raise ValueError("Current single-person prediction differs from saved candidate")
        scale = scales.get(row["videoSHA256"])
        if scale is None or scale <= 0:
            raise ValueError("Oracle requires a valid full-reference video body scale")
        index, counts = choose_candidate(row, candidates, scale, scorer)
        prediction = {"videoSHA256": key[0], "frameIndex": key[1], "detected": index is not None}
        if index is not None:
            prediction.update({k: candidates[index][k] for k in ("prediction", "keypointConfidences")})
        oracle_predictions.append(prediction)
        choices.append({"videoSHA256": key[0], "frameIndex": key[1], "candidateCount": len(candidates),
                        "ORACLE_selectedCandidateIndex": index, "candidatePassingVisibleJoints": counts,
                        "ORACLE_passingVisibleJoints": counts[index] if index is not None else 0,
                        "currentPolicyAccepted": bool(existing.get("detected")),
                        "maximizerCount": counts.count(counts[index]) if index is not None else 0})
    oracle_score = scorer.score(rows, oracle_predictions, .3)
    if oracle_score["bodyScalePixelsByVideo"] != scales:
        raise AssertionError("Reference scales changed")
    if sum(c["ORACLE_passingVisibleJoints"] for c in choices) != oracle_score["within0.05BodyScale"]:
        raise AssertionError("Oracle selection and independent body scorer disagree")
    strict = [r for r in rows if r.get("reviewMethod") == "frame" and all(p.get("source") != "interpolated" for p in r["points"])]
    return {
        "schema": "vision-ORACLE-selection-diagnostic-v1", "ORACLE_only": True, "eligibleForPromotion": False,
        "selectionRule": "Per-frame ground-truth maximum visible-joint PCK0.05BodyScale; ties use first saved candidate; confidence >=0.3, finite in-image points only",
        "bodyScalePixelsByVideo": scales,
        "counts": {"frames": len(rows), "personCandidates": sum(len(r["candidates"]) for r in all_candidates),
                   "candidateCountHistogram": dict(sorted(Counter(len(r["candidates"]) for r in all_candidates).items())),
                   "currentAcceptedFrames": current_score["detectedFrames"], "oracleSelectedFrames": oracle_score["detectedFrames"],
                   "strictFrames": len(strict), "strictVideos": len({r["videoSHA256"] for r in strict}),
                   "strictPointSources": dict(Counter(p.get("source", "unspecified") for r in strict for p in r["points"]))},
        "currentExactlyOnePersonPolicy": slices(current_score, scorer),
        "ORACLE_groundTruthSelectedCeiling": slices(oracle_score, scorer),
        "limitations": [
            "Diagnostic upper bound over these saved candidates under provisional reference labels, not deployable output, measured accuracy, or evidence for model promotion.",
            "Selection consults reference coordinates and may switch person identity independently every frame; no feasible tracker is evaluated.",
            "Wrist results use the body-optimal choice, not a separately maximized wrist-specific ceiling.",
            "Strict-slice results use the same full-reference per-video scales, never recomputed subset scales; all slice frames remain in denominators.",
            "Reviewed labels include interpolation and model-derived coordinates. No-interpolation and manual source do not establish blind human localization: the H visibility toggle can mark source manual without changing coordinates.",
            "The current native reconstruction can reject an entire clip containing multiple people. This comparison reports the saved per-frame policy and does not remove that whole-clip guard.",
            "Mac CPU Vision results do not establish iPhone parity, latency, 3D accuracy, or unseen-session generalization."
        ],
        "ORACLE_choicesForAuditOnly": choices,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, default=DEFAULT_SCORER)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {k: getattr(args, k).resolve() for k in ("truth", "candidates", "current", "scorer")}
    sources["diagnosticScript"] = Path(__file__).resolve()
    hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in sources.items()}
    result = analyze(*(json.loads(sources[k].read_text()) for k in ("truth", "candidates", "current")), load_scorer(args.scorer))
    if hashes != {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in sources.items()}:
        raise RuntimeError("Source changed while computing diagnostic")
    result["readOnlySources"] = {k: {"path": str(p), "sha256": hashes[k]} for k, p in sources.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ORACLE-diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["ORACLE VISION CANDIDATE-SELECTION CEILING — NOT DEPLOYABLE", "", result["selectionRule"], "",
             f"Frames={result['counts']['frames']}; saved people={result['counts']['personCandidates']}; current accepted={result['counts']['currentAcceptedFrames']}",
             "All scores below retain every eligible visible reference joint, including missing output.", ""]
    for name in ("all", "wristsUnderSameBodySelection", "frameReviewedNoInterpolation", "frameReviewedNoInterpolationWrists"):
        a, b = (result[k][name] for k in ("currentExactlyOnePersonPolicy", "ORACLE_groundTruthSelectedCeiling"))
        percent = lambda x: f"{x:.2%}" if x is not None else "unknown"
        lines.append(f"{name} ({a['frames']} frames): current {a['within0.05BodyScale']}/{a['normalizableJoints']} ({percent(a['jointPCK0.05BodyScale'])}); ORACLE {b['within0.05BodyScale']}/{b['normalizableJoints']} ({percent(b['jointPCK0.05BodyScale'])})")
    a, b = (result[k]["all"] for k in ("currentExactlyOnePersonPolicy", "ORACLE_groundTruthSelectedCeiling"))
    lines.extend(["", "INTERPRETATION",
                  f"The ORACLE adds {b['within0.05BodyScale'] - a['within0.05BodyScale']} passing joints over the same {a['normalizableJoints']} targets. Single-person choices are unchanged; added passes come from frames currently rejected for multiple candidates.",
                  f"Even the ORACLE leaves {b['normalizableJoints'] - b['within0.05BodyScale']} reference joints failing. This does not bound a new model, only selection among these saved candidate outputs.",
                  f"Strict slice point provenance: {result['counts']['strictPointSources']}; these {result['counts']['strictFrames']} frames cover {result['counts']['strictVideos']} video(s).",
                  "Useful next question: can a non-oracle hitter association method recover the intended person consistently? This diagnostic neither builds nor evaluates such a method.", "", "BY SOURCE VIDEO"])
    for video, a in result["currentExactlyOnePersonPolicy"]["byVideo"].items():
        b = result["ORACLE_groundTruthSelectedCeiling"]["byVideo"][video]
        lines.append(f"{video}: current {a['within0.05BodyScale']}/{a['normalizableJoints']}; ORACLE {b['within0.05BodyScale']}/{b['normalizableJoints']}")
    lines.extend(["", "LIMITATIONS", *result["limitations"], "", "Reproduction and immutable input hashes: ORACLE-diagnostic.json readOnlySources.", "No model predictions were persisted or promoted; candidate indices are for audit only."])
    (args.output / "README.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
