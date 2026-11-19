"""Dependency-free, miss-aware bat endpoint scorer; never uses confidence as accuracy.

Predictions: list of {clipId, frameIndex, detected, prediction:[[tipX,tipY],
[knobX,knobY]], score?}. Coordinates must be original, unmirrored image pixels.
Truth: original SQLite records (reviewed positives or explicit absent negatives).
Missing prediction records count as misses. Extra/duplicate records fail loudly.
"""
import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ENDPOINTS = ("tip", "knob")
PIXEL_THRESHOLDS = (10, 25, 50)
NORMALIZED_THRESHOLDS = (0.05, 0.1, 0.2)


def key(row):
    return str(row.get("clipId") or row.get("videoSHA256")), int(row["frameIndex"])


def finite_xy(xy):
    return (isinstance(xy, (list, tuple)) and len(xy) == 2
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    and math.isfinite(x) for x in xy))


def truth_xy(row, endpoint):
    point = row.get("points", {}).get(endpoint, {})
    if point.get("state") != "visible":
        return None
    xy = [point.get("pixelX"), point.get("pixelY")]
    if not finite_xy(xy):
        raise ValueError(f"Visible truth without finite pixels: {key(row)} {endpoint}")
    return xy


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    i = (len(values) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return values[lo] + (values[hi] - values[lo]) * (i - lo)


def directed_angle(a, b):
    """Angle of knob-to-tip vectors in degrees; endpoint swaps produce 180."""
    va, vb = [a[0][i] - a[1][i] for i in (0, 1)], [b[0][i] - b[1][i] for i in (0, 1)]
    la, lb = math.hypot(*va), math.hypot(*vb)
    if min(la, lb) <= 1e-8:
        return None
    cosine = max(-1, min(1, sum(x*y for x, y in zip(va, vb)) / (la * lb)))
    return math.degrees(math.acos(cosine))


def summarize(details):
    positive = [x for x in details if x["kind"] == "positive"]
    negative = [x for x in details if x["kind"] == "negative"]
    evaluated_negative = [x for x in negative if x["predictionRecordPresent"]]
    missing_negative = len(negative) - len(evaluated_negative)
    false_positives = sum(x["detected"] for x in evaluated_negative)
    both = [x for x in positive if x["bothTruthVisible"]]
    normalizable = [x for x in both if x["truthLengthPixels"] > 1e-8]
    errors = [e for x in positive for e in x["errorsPixels"] if e is not None]
    labeled = sum(x["visibleTruthEndpoints"] for x in positive)
    angles = [x["directedAngleDegrees"] for x in both if x["directedAngleDegrees"] is not None]
    detected = sum(x["detected"] for x in positive)
    result = {
        "positiveFrames": len(positive), "detectedFrames": detected,
        "missedFrames": len(positive) - detected,
        "detectionRate": ratio(detected, len(positive)),
        "visibleTruthEndpoints": labeled, "localizedEndpoints": len(errors),
        "missingEndpoints": labeled - len(errors),
        "bothVisibleFrames": len(both), "normalizableFrames": len(normalizable),
        "meanEndpointErrorPixelsOnLocalizedEndpoints": statistics.mean(errors) if errors else None,
        "medianEndpointErrorPixelsOnLocalizedEndpoints": statistics.median(errors) if errors else None,
        "p95EndpointErrorPixelsOnLocalizedEndpoints": percentile(errors, .95),
        "meanDirectedAngleDegreesOnValidPredictions": statistics.mean(angles) if angles else None,
        "validAnglePredictions": len(angles),
        "swapCandidates": sum(x["swapCandidate"] for x in both),
        "swapCandidateRateAmongBothVisibleFrames": ratio(sum(x["swapCandidate"] for x in both), len(both)),
        "negativeFrames": len(negative),
        "evaluatedNegativeFrames": len(evaluated_negative),
        "missingNegativePredictions": missing_negative,
        "falsePositiveFrames": false_positives,
        "negativeFalsePositiveRate": ratio(false_positives, len(negative)) if not missing_negative else None,
        "negativeFalsePositiveRateOnEvaluatedFrames": ratio(false_positives, len(evaluated_negative)),
        "unscoredReviewedFrames": sum(x["kind"] == "unscored" for x in details),
    }
    for threshold in PIXEL_THRESHOLDS:
        passed = sum(e <= threshold for e in errors)
        both_passed = sum(all(e is not None and e <= threshold for e in x["errorsPixels"]) for x in both)
        result[f"endpointPCK{threshold}Pixels"] = ratio(passed, labeled)
        result[f"bothEndpointsWithin{threshold}Pixels"] = both_passed
        result[f"bothEndpointsPCK{threshold}Pixels"] = ratio(both_passed, len(both))
    for threshold in NORMALIZED_THRESHOLDS:
        tag = str(threshold)
        passed = sum(sum(e is not None and e / x["truthLengthPixels"] <= threshold for e in x["errorsPixels"]) for x in normalizable)
        both_passed = sum(all(e is not None and e / x["truthLengthPixels"] <= threshold for e in x["errorsPixels"]) for x in normalizable)
        result[f"endpointPCK{tag}BatLength"] = ratio(passed, 2 * len(normalizable))
        result[f"bothEndpointsWithin{tag}BatLength"] = both_passed
        result[f"bothEndpointsPCK{tag}BatLength"] = ratio(both_passed, len(normalizable))
    for threshold in (5, 10, 20):
        result[f"directedAngleWithin{threshold}DegreesRate"] = ratio(sum(a <= threshold for a in angles), len(normalizable))
    return result


def score(rows, predictions):
    """Score reviewed data. Missing endpoints/detections fail PCK thresholds."""
    truth_keys = [key(r) for r in rows]
    prediction_keys = [key(p) for p in predictions]
    for label, keys in (("truth", truth_keys), ("prediction", prediction_keys)):
        duplicates = [k for k, n in Counter(keys).items() if n > 1]
        if duplicates:
            raise ValueError(f"Duplicate {label} identities: {duplicates}")
    extras = set(prediction_keys) - set(truth_keys)
    if extras:
        raise ValueError(f"Predictions without ground truth: {sorted(extras)}")
    prediction_map = dict(zip(prediction_keys, predictions))
    details, excluded = [], Counter()
    for row in rows:
        if row.get("status") not in ("reviewed", "absent"):
            excluded[row.get("status", "missing_status")] += 1
            continue
        identity = key(row)
        pred = prediction_map.get(identity, {})
        detected = bool(pred.get("detected", False))
        original = [truth_xy(row, name) for name in ENDPOINTS]
        count = sum(x is not None for x in original)
        kind = "negative" if row["status"] == "absent" else "positive" if count else "unscored"
        xy = pred.get("prediction")
        valid = (isinstance(xy, (list, tuple)) and len(xy) == 2)
        predicted = [p if detected and finite_xy(p) else None for p in xy] if valid else [None, None]
        errors = [math.dist(t, p) if t is not None and p is not None else None for t, p in zip(original, predicted)]
        length = math.dist(*original) if count == 2 else None
        complete = count == 2 and all(p is not None for p in predicted)
        angle = directed_angle(original, predicted) if complete else None
        swapped_error = sum(math.dist(original[i], predicted[1-i]) for i in (0, 1)) if complete else None
        # Review candidate, not an assertion: require reversal plus substantial gain.
        swap = bool(complete and angle is not None and angle > 90 and swapped_error + 1e-8 < .5 * sum(errors))
        details.append({"clipId": identity[0], "frameIndex": identity[1],
                        "videoSHA256": row.get("videoSHA256"), "session": row.get("session"),
                        "camera": row.get("camera"), "origin": row.get("origin", "unknown"),
                        "kind": kind, "predictionRecordPresent": identity in prediction_map,
                        "detected": detected, "visibleTruthEndpoints": count,
                        "bothTruthVisible": count == 2, "truthLengthPixels": length,
                        "errorsPixels": errors, "directedAngleDegrees": angle,
                        "swapCandidate": swap})
    result = {"schema": "bat-quality-score-v1", "endpointOrder": list(ENDPOINTS),
              "coordinateSpace": "original unmirrored image pixels",
              "excludedByStatus": dict(excluded),
              "notes": ["PCK and angle pass rates include misses in their denominators.",
                        "Pixel error and angle means are conditional; never compare alone.",
                        "Bat-length normalization uses ground-truth projected 2D length.",
                        "No absent labels means unknown false-positive rate, not zero.",
                        "No metrics establish unseen-session performance without that split."],
              **summarize(details)}
    for field, name in (("videoSHA256", "byVideo"), ("session", "bySession"), ("camera", "byCamera"), ("origin", "byOrigin")):
        groups = defaultdict(list)
        for detail in details:
            group = detail.get(field) or (detail["clipId"] if field == "videoSHA256" else "unknown")
            groups[group].append(detail)
        result[name] = {k: summarize(v) for k, v in sorted(groups.items())}
    videos = [v for v in result["byVideo"].values() if v["bothEndpointsPCK25Pixels"] is not None]
    result["macroVideoBothEndpointsPCK25Pixels"] = statistics.mean(v["bothEndpointsPCK25Pixels"] for v in videos) if videos else None
    for threshold in NORMALIZED_THRESHOLDS:
        metric = f"bothEndpointsPCK{threshold}BatLength"
        values = [v[metric] for v in result["byVideo"].values() if v[metric] is not None]
        result[f"macroVideoBothEndpointsPCK{threshold}BatLength"] = statistics.mean(values) if values else None
    result["details"] = details
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    truth, predictions = json.loads(args.truth.read_text()), json.loads(args.predictions.read_text())
    if isinstance(truth, dict):
        truth = truth.get("labels", truth.get("annotations", truth.get("records", truth.get("rows"))))
    if isinstance(predictions, dict):
        predictions = predictions.get("details", predictions.get("predictions"))
    args.output.write_text(json.dumps(score(truth, predictions), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
