"""Dependency-free scorer for exactly twelve ordered body landmarks.

Primary: visible-joint PCK at 0.05 clip-median reference body scale. The scale
is torso-midpoint length plus mean left/right hip-knee-ankle chain in pixels.
Missing predictions remain failures; interpolated reviewed labels are not gold.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

JOINTS = [f"{side} {joint}" for joint in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle") for side in ("Left", "Right")]


def key(r):
    return r["videoSHA256"], r["frameIndex"]


def finite(x):
    return isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)


def xy(p):
    return isinstance(p, (tuple, list)) and len(p) == 2 and all(finite(x) for x in p)


def reference_scales(rows):
    samples = defaultdict(list)
    for r in rows:
        p = r["points"]
        required = (0, 1, 6, 7, 8, 9, 10, 11)
        if not all(p[i].get("visibility") == 2 and all(finite(p[i].get(a)) for a in ("x", "y")) for i in required):
            continue
        pts = [(q["x"] * r["width"], q["y"] * r["height"]) if q.get("visibility") else None for q in p]
        midpoint = lambda a, b: tuple((a[i] + b[i]) / 2 for i in (0, 1))
        torso = math.dist(midpoint(pts[0], pts[1]), midpoint(pts[6], pts[7]))
        legs = sum(math.dist(pts[i], pts[i+2]) + math.dist(pts[i+2], pts[i+4]) for i in (6, 7)) / 2
        if torso + legs > 1e-8:
            samples[r["videoSHA256"]].append(torso + legs)
    return {video: statistics.median(values) for video, values in samples.items()}


def fraction(n, d):
    return n/d if d else None


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    x = (len(values)-1)*q
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi]-values[lo])*(x-lo)


def summarize(items):
    errors = [i["errorPixels"] for i in items if i["errorPixels"] is not None]
    normalized = [i for i in items if i["bodyScalePixels"] is not None]
    out = {"referenceJoints": len(items), "localizedJoints": len(errors),
           "missingJoints": len(items)-len(errors), "normalizableJoints": len(normalized),
           "meanErrorPixelsOnLocalized": statistics.mean(errors) if errors else None,
           "medianErrorPixelsOnLocalized": statistics.median(errors) if errors else None,
           "p95ErrorPixelsOnLocalized": percentile(errors, .95)}
    for t in (10, 25, 50):
        n = sum(e <= t for e in errors)
        out[f"within{t}Pixels"] = n
        out[f"jointPCK{t}Pixels"] = fraction(n, len(items))
    for t in (.025, .05, .1):
        n = sum(i["errorPixels"] is not None and i["errorPixels"] <= t*i["bodyScalePixels"] for i in normalized)
        out[f"within{t}BodyScale"] = n
        out[f"jointPCK{t}BodyScale"] = fraction(n, len(normalized))
    return out


def score(rows, predictions, keypoint_threshold=.3):
    if not finite(keypoint_threshold) or not 0 <= keypoint_threshold <= 1:
        raise ValueError("Invalid keypoint threshold")
    for label, values in (("truth", rows), ("prediction", predictions)):
        duplicates = [k for k, n in Counter(key(r) for r in values).items() if n > 1]
        if duplicates:
            raise ValueError(f"Duplicate {label} identities: {duplicates}")
    if set(map(key, predictions)) - set(map(key, rows)):
        raise ValueError("Prediction has no matching reference frame")
    for r in rows:
        if r.get("status") != "reviewed" or len(r.get("points", [])) != 12:
            raise ValueError("Expected reviewed twelve-point reference records")
        for p in r["points"]:
            if p.get("visibility") not in (0, 1, 2):
                raise ValueError("Invalid reference visibility")
            if p["visibility"] and not all(finite(p.get(a)) and 0 <= p[a] <= 1 for a in ("x", "y")):
                raise ValueError("Invalid reference coordinates")
    scales = reference_scales(rows)
    lookup = {key(p): p for p in predictions}
    details, frames = [], []
    for r in rows:
        p = lookup.get(key(r), {})
        points, confs = p.get("prediction"), p.get("keypointConfidences")
        if p.get("detected") and (not isinstance(points, list) or len(points) != 12 or not isinstance(confs, list) or len(confs) != 12):
            raise ValueError("Detected predictions require exactly twelve points and confidences")
        clean_frame = all(q.get("source") != "interpolated" for q in r["points"])
        frame_items = []
        for j, truth in enumerate(r["points"]):
            if truth["visibility"] == 0:
                continue
            pred = points[j] if p.get("detected") else None
            conf = confs[j] if p.get("detected") else None
            usable = (xy(pred) and finite(conf) and conf >= keypoint_threshold
                      and 0 <= pred[0] <= r["width"] and 0 <= pred[1] <= r["height"])
            expected = (truth["x"]*r["width"], truth["y"]*r["height"])
            item = {"videoSHA256": r["videoSHA256"], "frameIndex": r["frameIndex"],
                    "session": r["session"], "joint": JOINTS[j], "jointIndex": j,
                    "visibility": truth["visibility"], "source": truth.get("source", "unspecified"),
                    "reviewMethod": r.get("reviewMethod", "legacy_or_missing"),
                    "noInterpolatedJointsInFrame": clean_frame,
                    "bodyScalePixels": scales.get(r["videoSHA256"]),
                    "errorPixels": math.dist(expected, pred) if usable else None}
            details.append(item)
            if truth["visibility"] == 2:
                frame_items.append(item)
        frames.append({"videoSHA256": r["videoSHA256"], "frameIndex": r["frameIndex"],
                       "predictionRecordPresent": bool(p), "detected": bool(p.get("detected")),
                       "visibleReferenceJoints": len(frame_items), "localizedVisibleJoints": sum(x["errorPixels"] is not None for x in frame_items)})
    visible = [x for x in details if x["visibility"] == 2]
    result = {"schema": "body-quality-v1", "jointOrder": JOINTS,
              "primary": "visible jointPCK0.05BodyScale; misses count as failures",
              "referenceBodyScale": "Per-video median of visible shoulder-midpoint to hip-midpoint distance + mean hip-knee-ankle chain, original-image pixels",
              "bodyScalePixelsByVideo": scales, "keypointConfidenceThreshold": keypoint_threshold,
              "frames": len(rows), "detectedFrames": sum(f["detected"] for f in frames),
              "missingPredictionRecords": sum(not f["predictionRecordPresent"] for f in frames),
              **summarize(visible)}
    for field, name in (("videoSHA256", "byVideo"), ("session", "bySession"), ("joint", "byJoint"), ("source", "byPointSource"), ("reviewMethod", "byReviewMethod")):
        groups = defaultdict(list)
        for item in visible:
            groups[item[field]].append(item)
        result[name] = {k: summarize(v) for k, v in sorted(groups.items())}
    result["wrists"] = summarize([x for x in visible if x["jointIndex"] in (4, 5)])
    result["frameReviewedSubset"] = summarize([x for x in visible if x["reviewMethod"] == "frame"])
    result["noInterpolationFrameSubset"] = summarize([x for x in visible if x["noInterpolatedJointsInFrame"]])
    result["frameReviewedNoInterpolationSubset"] = summarize([x for x in visible if x["reviewMethod"] == "frame" and x["noInterpolatedJointsInFrame"]])
    result["hiddenEstimatedTargets"] = summarize([x for x in details if x["visibility"] == 1])
    macro = [v["jointPCK0.05BodyScale"] for v in result["byVideo"].values() if v["jointPCK0.05BodyScale"] is not None]
    result["macroVideoJointPCK0.05BodyScale"] = statistics.mean(macro) if macro else None
    result["notes"] = ["Reviewed/interpolated reference labels are not independent gold measurements.",
                       "Threshold policy is common across YOLO models, not confidence calibration across model families.",
                       "Detection uses a declared non-oracle candidate policy. Wrong-person selection is not corrected with reference boxes.",
                       "Conditional error cannot replace miss-aware PCK. No body score measures 3D accuracy."]
    result["frameDetails"], result["jointDetails"] = frames, details
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keypoint-threshold", type=float, default=.3)
    args = parser.parse_args()
    rows = json.loads(args.truth.read_text())
    predictions = json.loads(args.predictions.read_text())
    args.output.write_text(json.dumps(score(rows, predictions, args.keypoint_threshold), indent=2, allow_nan=False)+"\n")
