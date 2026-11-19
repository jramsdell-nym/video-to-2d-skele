"""CPU-only temporal output audit using consecutive frames and exact presentation PTS.

Residual change is (prediction motion minus reference motion), so correct rapid
motion is not labeled jitter. All flags require visual review, especially where
reference labels came from interpolation. No source files are modified.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from score_bat import ENDPOINTS, directed_angle, finite_xy, key, percentile, truth_xy


def exact_time(row):
    source = row.get("source", {})
    try:
        pts = source["pts"]
        if not isinstance(pts, int) or isinstance(pts, bool):
            return None
        scale = Fraction(source["timeBase"])
        if scale <= 0 or source.get("index", row["frameIndex"]) != row["frameIndex"]:
            return None
        value = pts * scale
        if "time" in source and (not math.isfinite(source["time"]) or abs(float(value) - source["time"]) > 2e-6):
            return None
        return value
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        return None


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def distribution(values):
    return {"count": len(values), "mean": statistics.mean(values) if values else None,
            "median": percentile(values, .5), "p95": percentile(values, .95),
            "maximum": max(values) if values else None}


def apply_policy(pred, box_threshold=None, keypoint_threshold=None):
    detected = bool(pred.get("detected", False))
    if detected and box_threshold is not None:
        if not numeric(pred.get("score")):
            raise ValueError("Box gating requested, but a detected prediction has no finite score")
        detected = pred["score"] >= box_threshold
    raw = pred.get("prediction")
    points = list(raw) if isinstance(raw, (list, tuple)) and len(raw) == 2 else [None, None]
    if detected and keypoint_threshold is not None:
        confidences = pred.get("endpointConfidences")
        if not isinstance(confidences, (list, tuple)) or len(confidences) != 2 or not all(numeric(c) for c in confidences):
            raise ValueError("Endpoint gating requested, but a detected prediction has no two finite endpoint confidences")
        points = [p if confidence >= keypoint_threshold else None for p, confidence in zip(points, confidences)]
    return detected, [p if detected and finite_xy(p) else None for p in points]


def dropout_runs(segment, kind):
    """Durations are observed first-to-last-missing PTS spans, not assumed frame time."""
    runs, current = [], None
    previous_available = False
    for node in segment:
        eligible = node["dropoutEligible"][kind]
        missing = node["missing"][kind] if eligible else False
        if eligible and missing:
            if current is None:
                current = {"videoSHA256": node["videoSHA256"], "clipId": node["clipId"],
                           "kind": kind, "firstFrame": node["frameIndex"], "lastFrame": node["frameIndex"],
                           "firstTimeSeconds": node["timeSeconds"], "lastTimeSeconds": node["timeSeconds"],
                           "missingFrames": 0, "unexecutedPredictionRecords": 0,
                           "leftCensored": not previous_available}
            current["lastFrame"], current["lastTimeSeconds"] = node["frameIndex"], node["timeSeconds"]
            current["missingFrames"] += 1
            current["unexecutedPredictionRecords"] += not node["predictionRecordPresent"]
        elif current is not None:
            current["resumedAtFrame"] = node["frameIndex"] if eligible else None
            current["resumedAtTimeSeconds"] = node["timeSeconds"] if eligible else None
            current["rightCensored"] = not eligible
            runs.append(current)
            current = None
        previous_available = eligible and not missing
    if current is not None:
        current.update(resumedAtFrame=None, resumedAtTimeSeconds=None, rightCensored=True)
        runs.append(current)
    for run in runs:
        start, end, resumed = run["firstTimeSeconds"], run["lastTimeSeconds"], run["resumedAtTimeSeconds"]
        run["observedSpanSeconds"] = end - start if start is not None and end is not None else None
        run["timeUntilResumeSeconds"] = resumed - start if resumed is not None and start is not None else None
    return runs


def summarize(nodes, residual_pairs, runs, eligible_pairs):
    out = {"frames": len(nodes), "missingPredictionRecords": sum(not n["predictionRecordPresent"] for n in nodes),
           "invalidTimingFrames": sum(n["timeSeconds"] is None for n in nodes),
           "consecutiveFramePairs": len(eligible_pairs), "dropouts": {}, "residualChange": {}}
    for kind in ("detection", "tip", "knob", "eitherEndpoint"):
        selected = [r for r in runs if r["kind"] == kind]
        spans = [r["observedSpanSeconds"] for r in selected if r["observedSpanSeconds"] is not None]
        resumes = [r["timeUntilResumeSeconds"] for r in selected if r["timeUntilResumeSeconds"] is not None]
        eligible = sum(n["dropoutEligible"][kind] for n in nodes)
        missing = sum(n["missing"][kind] for n in nodes if n["dropoutEligible"][kind])
        out["dropouts"][kind] = {"eligibleFrames": eligible, "missingFrames": missing,
                                  "missingFraction": missing/eligible if eligible else None,
                                  "runs": len(selected), "longestRunFrames": max((r["missingFrames"] for r in selected), default=0),
                                  "longestObservedSpanSeconds": max(spans) if spans else None,
                                  "longestTimeUntilResumeSeconds": max(resumes) if resumes else None,
                                  "rightCensoredRuns": sum(r["rightCensored"] for r in selected)}
    for endpoint in ("tip", "knob", "all"):
        selected = [p for p in residual_pairs if endpoint == "all" or p["endpoint"] == endpoint]
        eligible = sum(pair["eligibleEndpoints"][endpoint] if endpoint != "all" else sum(pair["eligibleEndpoints"].values()) for pair in eligible_pairs)
        out["residualChange"][endpoint] = {"eligibleEndpointPairs": eligible,
                                           "measuredEndpointPairs": len(selected),
                                           "unmeasuredEndpointPairs": eligible-len(selected),
                                           "deltaPixels": distribution([p["residualDeltaPixels"] for p in selected]),
                                           "deltaBatLengths": distribution([p["residualDeltaBatLengths"] for p in selected if p["residualDeltaBatLengths"] is not None]),
                                           "pixelsPerPresentationSecond": distribution([p["pixelsPerPresentationSecond"] for p in selected]),
                                           "pairsWithInterpolatedReference": sum(p["containsInterpolatedTruth"] for p in selected)}
    return out


def analyze(rows, predictions, box_threshold=None, keypoint_threshold=None, jump_pixels=25, jump_bat_lengths=.15):
    for value in (box_threshold, keypoint_threshold):
        if value is not None and (not numeric(value) or not 0 <= value <= 1):
            raise ValueError("Confidence thresholds must be in [0,1]")
    for label, records in (("truth", rows), ("prediction", predictions)):
        if len({key(r) for r in records}) != len(records):
            raise ValueError(f"Duplicate {label} identities")
    if len({(r.get("videoSHA256") or key(r)[0], r["frameIndex"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate truth source video/frame under different clip aliases")
    if {key(r) for r in predictions} - {key(r) for r in rows}:
        raise ValueError("Predictions have no matching truth")
    prediction_map = {key(p): p for p in predictions}
    groups, excluded = defaultdict(list), Counter()
    for row in rows:
        if row.get("status") != "reviewed":
            excluded[row.get("status", "unknown")] += 1
            continue
        truth = [truth_xy(row, endpoint) for endpoint in ENDPOINTS]
        if not any(p is not None for p in truth):
            excluded["no_visible_truth_endpoints"] += 1
            continue
        identity = key(row)
        pred = prediction_map.get(identity, {})
        detected, points = apply_policy(pred, box_threshold, keypoint_threshold)
        time = exact_time(row)
        eligible = {"detection": True, "tip": truth[0] is not None, "knob": truth[1] is not None,
                    "eitherEndpoint": all(p is not None for p in truth)}
        missing = {"detection": not detected, "tip": points[0] is None, "knob": points[1] is None,
                   "eitherEndpoint": any(p is None for p in points)}
        video = row.get("videoSHA256") or identity[0]
        groups[video].append({"clipId": identity[0], "videoSHA256": video, "frameIndex": identity[1],
                              "timeSeconds": float(time) if time is not None else None, "exactTime": time,
                              "truth": truth, "prediction": points, "detected": detected,
                              "predictionRecordPresent": identity in prediction_map,
                              "truthSources": [row.get("points", {}).get(n, {}).get("source", row.get("origin", "unspecified")) for n in ENDPOINTS],
                              "reviewMethod": row.get("reviewMethod", "unspecified"),
                              "dropoutEligible": eligible, "missing": missing})
    all_nodes, residual_pairs, runs, eligible_pairs, breaks, flags = [], [], [], [], [], []
    for video, nodes in sorted(groups.items()):
        nodes.sort(key=lambda n: n["frameIndex"])
        segments, segment = [], []
        for node in nodes:
            previous = segment[-1] if segment else None
            reason = None
            if previous is not None:
                if node["frameIndex"] != previous["frameIndex"] + 1:
                    reason = "nonconsecutive_source_frame"
                elif node["exactTime"] is None or previous["exactTime"] is None:
                    reason = "invalid_presentation_timestamp"
                elif node["exactTime"] <= previous["exactTime"]:
                    reason = "nonpositive_presentation_delta"
                if reason:
                    breaks.append({"videoSHA256": video, "previousFrame": previous["frameIndex"],
                                   "nextFrame": node["frameIndex"], "reason": reason})
                    segments.append(segment)
                    segment = []
            segment.append(node)
            truth, pred = node["truth"], node["prediction"]
            if all(p is not None for p in truth+pred):
                angle = directed_angle(truth, pred)
                direct = sum(math.dist(t, p) for t, p in zip(truth, pred))
                reverse = sum(math.dist(truth[i], pred[1-i]) for i in (0, 1))
                if angle is not None and angle > 90 and reverse + 1e-8 < .5*direct:
                    flags.append({"videoSHA256": video, "clipId": node["clipId"], "frameIndex": node["frameIndex"],
                                  "endpoint": "both", "reason": "possible_endpoint_swap",
                                  "evidence": f"Directed angle {angle:.1f}deg; reversed assignment error {reverse:.1f}px vs direct {direct:.1f}px. Inspect prediction and reference."})
        if segment:
            segments.append(segment)
        for segment in segments:
            for kind in ("detection", "tip", "knob", "eitherEndpoint"):
                runs.extend(dropout_runs(segment, kind))
            for previous, node in zip(segment, segment[1:]):
                dt = float(node["exactTime"] - previous["exactTime"])
                eligible_pairs.append({"videoSHA256": video, "previousFrame": previous["frameIndex"], "frameIndex": node["frameIndex"],
                                       "eligibleEndpoints": {name: previous["truth"][i] is not None and node["truth"][i] is not None for i, name in enumerate(ENDPOINTS)}})
                lengths = [math.dist(*n["truth"]) for n in (previous, node) if all(p is not None for p in n["truth"])]
                scale = sum(lengths)/2 if len(lengths) == 2 and min(lengths) > 1e-8 else None
                for i, endpoint in enumerate(ENDPOINTS):
                    if any(n[field][i] is None for n in (previous, node) for field in ("truth", "prediction")):
                        continue
                    residuals = [[n["prediction"][i][d] - n["truth"][i][d] for d in (0, 1)] for n in (previous, node)]
                    delta = math.dist(*residuals)
                    normalized = delta/scale if scale is not None else None
                    interpolated = any(n["truthSources"][i] == "interpolated" for n in (previous, node))
                    item = {"videoSHA256": video, "clipId": node["clipId"], "previousFrame": previous["frameIndex"],
                            "frameIndex": node["frameIndex"], "endpoint": endpoint, "deltaSeconds": dt,
                            "residualDeltaPixels": delta, "residualDeltaBatLengths": normalized,
                            "pixelsPerPresentationSecond": delta/dt, "meanTruthBatLengthPixels": scale,
                            "containsInterpolatedTruth": interpolated,
                            "previousTruthSource": previous["truthSources"][i], "truthSource": node["truthSources"][i]}
                    residual_pairs.append(item)
                    if delta > jump_pixels and (normalized is None or normalized > jump_bat_lengths):
                        flags.append({"videoSHA256": video, "clipId": node["clipId"], "frameIndex": node["frameIndex"],
                                      "endpoint": endpoint, "reason": "large_residual_change",
                                      "evidence": f"Previous frame {previous['frameIndex']}; residual delta {delta:.1f}px over {dt:.6f}s; interpolated reference={interpolated}. Inspect reference and prediction, not motion speed alone."})
        all_nodes.extend(nodes)
    result = {"schema": "temporal-bat-quality-v1", "policy": {"boxThreshold": box_threshold, "keypointThreshold": keypoint_threshold,
              "jumpPixels": jump_pixels, "jumpBatLengths": jump_bat_lengths}, "excludedByStatus": dict(excluded),
              "notes": ["Adjacent pairs require consecutive original frame indices, valid exact PTS/time base and strictly positive time difference. Gaps break every run.",
                        "Residual delta is the change in prediction-minus-reference error; true rapid motion cancels when prediction is correct.",
                        "Dropout span is first-to-last missing presentation timestamp, a lower bound. Single-frame spans are zero; no frame duration is invented.",
                        "Time-until-resume includes the next available observation only within the same consecutive segment. Censored runs have unknown recovery.",
                        "Missing prediction records are separately reported and must not be mistaken for executed inference failures.",
                        "Residual distributions are conditional on observed endpoints. Compare them together with dropout and unmeasured-pair coverage.",
                        "Interpolated or model-derived reference errors can look like model jitter. All flags require visual review, never automatic correction.",
                        "Optional confidence gates analyze raw model coordinates; they do not replay native candidate association or reject finite out-of-image coordinates. These are not full iPhone policy metrics.",
                        "Pixels/second use presentation PTS, not verified capture time or physical bat speed."],
              **summarize(all_nodes, residual_pairs, runs, eligible_pairs),
              "framesWithInterpolatedTruth": sum("interpolated" in n["truthSources"] for n in all_nodes),
              "byVideo": {}, "breaks": breaks, "flagCounts": dict(Counter(f["reason"] for f in flags)),
              "dropoutRuns": runs, "residualPairs": residual_pairs, "reviewFlags": flags}
    for video, nodes in sorted(groups.items()):
        result["byVideo"][video] = summarize(nodes, [p for p in residual_pairs if p["videoSHA256"] == video],
                                           [r for r in runs if r["videoSHA256"] == video],
                                           [p for p in eligible_pairs if p["videoSHA256"] == video])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--box-threshold", type=float)
    parser.add_argument("--keypoint-threshold", type=float)
    args = parser.parse_args()
    rows, predictions = json.loads(args.truth.read_text()), json.loads(args.predictions.read_text())
    if isinstance(rows, dict):
        rows = next(rows[k] for k in ("rows", "labels", "annotations", "records") if k in rows)
    if isinstance(predictions, dict):
        predictions = next(predictions[k] for k in ("predictions", "details") if k in predictions)
    result = analyze(rows, predictions, args.box_threshold, args.keypoint_threshold)
    result["inputs"] = {"truth": str(args.truth.resolve()), "predictions": str(args.predictions.resolve()),
                        "truthSHA256": hashlib.sha256(args.truth.read_bytes()).hexdigest(),
                        "predictionsSHA256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
                        "analyzerSHA256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with args.output.with_suffix(".review.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["videoSHA256", "clipId", "frameIndex", "endpoint", "reason", "evidence"])
        writer.writeheader()
        writer.writerows(result["reviewFlags"])
    print(json.dumps({"frames": result["frames"], "consecutiveFramePairs": result["consecutiveFramePairs"],
                      "dropouts": result["dropouts"], "flagCounts": result["flagCounts"]}, indent=2))


if __name__ == "__main__":
    main()
