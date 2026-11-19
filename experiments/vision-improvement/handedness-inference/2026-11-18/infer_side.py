"""Experimental, label-derived batting-side proposals; never edit source labels."""
from collections import defaultdict
from pathlib import Path
from statistics import median
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1] / "cycles/2026-11-18/dataset/body-rows.json"


def main():
    raw = SOURCE.read_bytes()
    groups = defaultdict(list)
    for row in json.loads(raw):
        groups[row["videoSHA256"]].append(row)
    proposals = []
    for sha, rows in sorted(groups.items(), key=lambda item: item[1][0]["trial"]):
        rows.sort(key=lambda row: row["frameIndex"])
        evidence = []
        for row in rows[:max(1, len(rows) // 3)]:
            # Body12 retains COCO slots5:17: shoulders0/1, wrists4/5.
            points = [row["points"][i] for i in (0, 1, 4, 5)]
            if any(point.get("visibility") != 2 for point in points):
                continue
            xy = [(p["x"] * row["width"], p["y"] * row["height"]) for p in points]
            hands = tuple((xy[2][i] + xy[3][i]) / 2 for i in range(2))
            left, right = math.dist(hands, xy[0]), math.dist(hands, xy[1])
            if left + right <= 1e-9:
                continue
            # Positive means hands closer to the anatomical right shoulder.
            evidence.append({"frameIndex": row["frameIndex"],
                             "rightShoulderProximity": (left - right) / (left + right)})
        signal = median(item["rightShoulderProximity"] for item in evidence) if evidence else None
        candidate = ("R" if signal > 0 else "L") if signal else None
        consistency = (sum((item["rightShoulderProximity"] > 0) == (signal > 0)
                           for item in evidence) / len(evidence)) if candidate else None
        eligible = candidate is not None and len(evidence) >= 5 and abs(signal) >= .15 and consistency >= .85
        proposals.append({"videoSHA256": sha, "trial": rows[0]["trial"],
                          "camera": rows[0]["camera"], "existingSide": rows[0]["handedness"],
                          "candidateSide": candidate, "proposedSide": candidate if eligible else None,
                          "reviewStatus": "pending-human-review", "source": "derived-from-body-labels",
                          "medianProximity": signal, "signAgreement": consistency,
                          "usableFrames": len(evidence), "evidence": evidence,
                          "eligibleForProposal": eligible})
    known = [p for p in proposals if p["existingSide"] in ("L", "R")]
    unknown = [p for p in proposals if p["existingSide"] not in ("L", "R")]
    result = {"source": str(SOURCE), "sourceSHA256": hashlib.sha256(raw).hexdigest(),
              "method": "Hands midpoint relative to anatomical shoulders over first third of available labeled frames; positive median proximity means R.",
              "gates": {"minimumUsableFrames": 5, "minimumAbsoluteMedianProximity": .15,
                        "minimumSignAgreement": .85},
              "summary": {"knownClips": len(known),
                          "knownCandidateMatches": sum(p["candidateSide"] == p["existingSide"] for p in known),
                          "unknownClips": len(unknown),
                          "unknownProposals": sum(p["proposedSide"] is not None for p in unknown)},
              "limitations": ["Exploratory rule checked against these six known tags after a wrist-to-bat projection shortcut disagreed with five. This is not independent validation.",
                              "Sign agreement is temporal consistency, not a calibrated probability. Interpolated/model-assisted frames are correlated.",
                              "Early frames are not verified setup-phase labels. Cross-body hand positions, poor anatomical labels, occlusion and camera perspective can break this rule.",
                              "Batting side and camera view are distinct. No camera-view classification is produced.",
                              "No source labels, benchmark groups, model weights, or previous evaluation results are changed."],
              "proposals": proposals}
    assert SOURCE.read_bytes() == raw
    (ROOT / "proposals.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]))
    for p in proposals:
        print(p["trial"], "existing=", p["existingSide"], "proposal=", p["proposedSide"],
              "frames=", p["usableFrames"], "agreement=", p["signAgreement"])


if __name__ == "__main__":
    main()
