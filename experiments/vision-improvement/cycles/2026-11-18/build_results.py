"""Summarize the completed fixed-recipe evaluation without selecting checkpoints."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BODY = "jointPCK0.05BodyScale"
BAT = "bothEndpointsPCK0.1BatLength"


def read(path):
    return json.loads(path.read_text())


def body_summary(score):
    return {"bodyPCK": score[BODY], "wristsPCK": score["wrists"][BODY],
            "macroVideoBodyPCK": score["macroVideoJointPCK0.05BodyScale"],
            "visibleReferenceJoints": score["referenceJoints"], "missingJoints": score["missingJoints"],
            "visibleReferenceWrists": score["wrists"]["referenceJoints"],
            "bodySuccesses": score["within0.05BodyScale"],
            "wristSuccesses": score["wrists"]["within0.05BodyScale"],
            "perJointPCK": {k: v[BODY] for k, v in score["byJoint"].items()},
            "frameReviewedPCK": score["frameReviewedSubset"][BODY],
            "frameReviewedNoInterpolationPCK": score["frameReviewedNoInterpolationSubset"][BODY],
            "frameReviewedNoInterpolationReferenceJoints": score["frameReviewedNoInterpolationSubset"]["referenceJoints"]}


def bat_summary(score):
    return {"pairedEndpointsPCK": score[BAT], "macroVideoPairedEndpointsPCK": score["macroVideoBothEndpointsPCK0.1BatLength"],
            "pairedSuccesses": score["bothEndpointsWithin0.1BatLength"],
            "pairedReferenceFrames": score["normalizableFrames"], "missingEndpoints": score["missingEndpoints"],
            "missedFrames": score["missedFrames"], "swapCandidates": score["swapCandidates"]}


if __name__ == "__main__":
    sources = {"body": {name: read(ROOT / f"training/body/{name}-primary-score.json") for name in ("generic", "previous", "candidate")},
               "bat": {name: read(ROOT / f"training/bat/{name}-primary-score.json") for name in ("current", "candidate")}}
    result = {"dataset": read(ROOT / "dataset/inventory.json"), "split": read(ROOT / "split-receipt.json")["tasks"],
              "body": {name: body_summary(score) for name, score in sources["body"].items()},
              "bat": {name: bat_summary(score) for name, score in sources["bat"].items()},
              "evaluation": "Five whole swings from three sessions excluded from candidate training/validation and the compared checkpoints' recorded training trials.",
              "limitations": ["Reference labels include interpolation/model assistance; these scores measure agreement with reviewed labels, not independent gold-standard accuracy.",
                              "Five swings are a small correlated sample; athlete independence is not established.",
                              "Body videos compare generic YOLO11n-pose with the new YOLO candidate, not with the phone's Apple Vision body provider.",
                              "2D detector-only evaluation; no claim of improved 3D reconstruction and no deployed model replaced."],
              "perVideo": []}
    for clip in read(ROOT / "videos/frozen-selection.json")["clips"]:
        sha = clip["source"]["sha256"]
        item = {"slug": clip["slug"], "title": clip["title"], "videoSHA256": sha, "sourceFrames": clip["frameCount"],
                "body": {name: score["byVideo"][sha][BODY] for name, score in sources["body"].items()},
                "bat": {name: score["byVideo"][sha][BAT] for name, score in sources["bat"].items()}}
        result["perVideo"].append(item)
    result["body"]["changeVsGenericPercentagePoints"] = 100 * (result["body"]["candidate"]["bodyPCK"] - result["body"]["generic"]["bodyPCK"])
    result["body"]["wristChangeVsGenericPercentagePoints"] = 100 * (result["body"]["candidate"]["wristsPCK"] - result["body"]["generic"]["wristsPCK"])
    result["body"]["changeVsPreviousPercentagePoints"] = 100 * (result["body"]["candidate"]["bodyPCK"] - result["body"]["previous"]["bodyPCK"])
    result["body"]["wristChangeVsPreviousPercentagePoints"] = 100 * (result["body"]["candidate"]["wristsPCK"] - result["body"]["previous"]["wristsPCK"])
    result["bat"]["changeVsCurrentPercentagePoints"] = 100 * (result["bat"]["candidate"]["pairedEndpointsPCK"] - result["bat"]["current"]["pairedEndpointsPCK"])
    output = ROOT / "results.json"
    assert not output.exists(), "Do not overwrite a published result summary"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("body", "bat", "perVideo")}, indent=2))
