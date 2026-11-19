"""Score paired experiments and render deterministic, labeled audit figures."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "quality"))
from score_bat import score


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def filtered(predictions):
    result = deepcopy(predictions)
    for item in result:
        item["detected"] = item["detected"] and item.get("score", 0) >= .25
        if item["detected"]:
            item["prediction"] = [xy if conf >= .25 else None for xy, conf in zip(item["prediction"], item["endpointConfidences"])]
    return result


def render(run, rows, predictions, keys, name, subtitle, title="BAT ENDPOINTS | Held-out Randy videos", recipes=("baseline", "candidate")):
    from PIL import Image, ImageDraw, ImageFont
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    font = ImageFont.truetype(font_path, 26)
    small = ImageFont.truetype(font_path, 21)
    title_font = ImageFont.truetype(font_path, 34)
    truth = {(r["clipId"], r["frameIndex"]): r for r in rows}
    maps = {recipe: {(r["clipId"], r["frameIndex"]): r for r in data} for recipe, data in predictions.items()}
    # Full original frame kept visible. Colored tip/knob endpoints and connecting
    # lines are rendered in source pixels before the 50% display resize.
    cell_w, cell_h = 960, 590
    image = Image.new("RGB", (1920, 145 + cell_h * len(keys)), "#101827")
    draw = ImageDraw.Draw(image)
    draw.text((24, 15), title, font=title_font, fill="white")
    draw.text((24, 61), subtitle, font=font, fill="#cbd5e1")
    draw.text((24, 101), "Green = reviewed truth; pink = model | T = tip, K = knob | raw box threshold 0.15", font=small, fill="#b8dcc7")
    details = []
    for ri, identity in enumerate(keys):
        row = truth[identity]
        item_detail = {"clipId": identity[0], "frameIndex": identity[1], "recipes": {}}
        for ci, recipe in enumerate(recipes):
            pred = maps[recipe][identity]
            original = Image.open(row.get("imagePath") or run / "dataset" / "images" / f"{identity[0]}-{identity[1]:04d}.jpg").convert("RGB")
            overlay = ImageDraw.Draw(original)
            actual = [(row["points"][p]["pixelX"], row["points"][p]["pixelY"]) for p in ("tip", "knob")]
            predicted = pred.get("prediction") if pred["detected"] else None
            for pts, color in ((actual, "#20ff9b"), (predicted, "#ff62ce")):
                if not pts:
                    continue
                overlay.line([tuple(p) for p in pts], fill=color, width=5)
                for label, (x, y) in zip(("T", "K"), pts):
                    overlay.ellipse((x - 10, y - 10, x + 10, y + 10), outline=color, width=4)
                    overlay.text((x + 15, y - 30), label, font=title_font, fill=color, stroke_width=3, stroke_fill="black")
            x0, y0 = ci * cell_w, 145 + ri * cell_h
            image.paste(original.resize((960, 540)), (x0, y0 + 50))
            import math
            errors = [math.dist(a, b) for a, b in zip(actual, predicted)] if predicted else None
            label = f"{recipe.upper()} | {identity[0][:6]} frame {identity[1]} | "
            label += f"tip {errors[0]:.1f}px / knob {errors[1]:.1f}px" if errors else "MISSED"
            draw.text((x0 + 15, y0 + 12), label, font=small, fill="white")
            item_detail["recipes"][recipe] = {"detected": pred["detected"], "errorsPixels": errors}
        details.append(item_detail)
    image.save(run / name, quality=92, subsampling=0)
    save(run / (Path(name).stem + "-frames.json"), details)


def main(run):
    results = {}
    summaries = {}
    important = ["positiveFrames", "detectedFrames", "missedFrames", "missingEndpoints", "bothEndpointsWithin25Pixels", "bothEndpointsPCK25Pixels", "bothEndpointsPCK50Pixels", "bothEndpointsPCK0.1BatLength", "directedAngleWithin10DegreesRate", "meanEndpointErrorPixelsOnLocalizedEndpoints", "medianEndpointErrorPixelsOnLocalizedEndpoints", "p95EndpointErrorPixelsOnLocalizedEndpoints", "swapCandidates", "macroVideoBothEndpointsPCK25Pixels", "negativeFrames"]
    splits = ("val", "test", "external") if (run / "candidate-external-predictions.json").exists() else ("val", "test")
    for split in splits:
        rows = json.loads((run / f"{split}-rows.json").read_text())
        recipes = ("baseline", "candidate", "deployed") if split == "external" and (run / "deployed-external-predictions.json").exists() else ("baseline", "candidate")
        for recipe in recipes:
            preds = json.loads((run / f"{recipe}-{split}-predictions.json").read_text())
            for policy, data in (("raw015", preds), ("confidence_filtered025", filtered(preds))):
                result = score(rows, data)
                result["policy"] = {"name": policy, "boxThreshold": .15 if policy == "raw015" else .25, "endpointThreshold": None if policy == "raw015" else .25, "top1Only": True, "exactAppParity": False}
                tag = f"{recipe}-{split}-{policy}"
                save(run / f"{tag}-score.json", result)
                results[tag] = result
                summaries[tag] = {k: result[k] for k in important}
                if split == "external":
                    strata = {
                        "both_manual": [r for r in rows if r.get("bothEndpointsManual")],
                        "frame_reviewed_both_manual": [r for r in rows if r.get("bothEndpointsManual") and r.get("reviewMethod") == "frame"],
                        "interpolated": [r for r in rows if r.get("eitherEndpointInterpolated")],
                    }
                    for stratum, truth_rows in strata.items():
                        identities = {(r["clipId"], r["frameIndex"]) for r in truth_rows}
                        subresult = score(truth_rows, [d for d in data if (d["clipId"], d["frameIndex"]) in identities])
                        save(run / f"{tag}-{stratum}-score.json", subresult)
                        summaries[f"{tag}-{stratum}"] = {k: subresult[k] for k in important}
    rows = json.loads((run / "test-rows.json").read_text())
    preds = {recipe: json.loads((run / f"{recipe}-test-predictions.json").read_text()) for recipe in ("baseline", "candidate")}
    selected = json.loads((run / "visual-selection.json").read_text())
    fixed = [(r["clipId"], r["frameIndex"]) for r in selected["frames"]]
    render(run, rows, preds, fixed, "fixed-heldout-comparison.jpg", "Six frames chosen before inference: first, middle and last labeled frame in each video")
    candidate_details = results["candidate-test-raw015"]["details"]
    worst = sorted(candidate_details, key=lambda d: max((x if x is not None else float("inf")) for x in d["errorsPixels"]), reverse=True)[:3]
    worst_keys = [(r["clipId"], r["frameIndex"]) for r in worst]
    render(run, rows, preds, worst_keys, "worst-candidate-errors.jpg", "Error review: three worst candidate frames selected AFTER scoring; not a representative sample")
    if "external" in splits:
        external_rows = json.loads((run / "external-rows.json").read_text())
        external_preds = {recipe: json.loads((run / f"{recipe}-external-predictions.json").read_text()) for recipe in ("baseline", "candidate")}
        external_keys = [(r["clipId"], r["frameIndex"]) for r in json.loads((run / "external-visual-selection.json").read_text())["frames"]]
        render(run, external_rows, external_preds, external_keys, "fixed-external-comparison.jpg", "Preselected first/middle/last frames; labels include interpolation and bulk review", "BAT ENDPOINTS | New-session provisional labels")
        if (run / "deployed-external-predictions.json").exists():
            external_preds["deployed"] = json.loads((run / "deployed-external-predictions.json").read_text())
            render(run, external_rows, external_preds, external_keys, "deployed-vs-candidate-external.jpg", "Preselected first/middle/last frames; labels include interpolation and bulk review", "BAT ENDPOINTS | Current deployed source vs candidate", ("deployed", "candidate"))
    manifest = json.loads((run / "manifest.json").read_text())
    test_delta = {}
    for policy in ("raw015", "confidence_filtered025"):
        test_delta[policy] = {k: summaries[f"candidate-test-{policy}"][k] - summaries[f"baseline-test-{policy}"][k] for k in ("bothEndpointsPCK25Pixels", "bothEndpointsPCK50Pixels", "bothEndpointsPCK0.1BatLength", "directedAngleWithin10DegreesRate", "meanEndpointErrorPixelsOnLocalizedEndpoints")}
    output = {"createdAt": datetime.now(timezone.utc).isoformat(), "summary": summaries, "testCandidateMinusBaseline": test_delta,
              "scoreSHA256": hashlib.sha256((ROOT.parent / "quality" / "score_bat.py").read_bytes()).hexdigest(),
              "selection": manifest["selectionPolicy"], "splits": {k: {f: v[f] for f in ("count", "players", "clips")} for k, v in manifest["splits"].items()},
              "limitations": manifest["limitations"] + ["Seeded MPS scatter_reduce and index_put kernels warn of nondeterminism; identical seeds do not guarantee bitwise repeatability.", "Confidence-filtered 0.25 scoring approximates endpoint gates only; it does not replay the app's wrist association, multi-person selection, out-of-frame handling, temporal pipeline, or 3D solver."],
              "figures": ["fixed-heldout-comparison.jpg", "worst-candidate-errors.jpg"]}
    if "external" in splits:
        output["externalEvaluation"] = json.loads((run / "external-provenance.json").read_text())
        output["limitations"].append("External-session scores measure agreement with provisional reviewed labels, including interpolation and model assistance. Separate both-manual and frame-reviewed+both-manual strata are reported; none establishes independent gold accuracy.")
        output["figures"].append("fixed-external-comparison.jpg")
        if (run / "deployed-vs-candidate-external.jpg").exists():
            output["figures"].append("deployed-vs-candidate-external.jpg")
    save(run / "comparison.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=ROOT / "player-holdout-clean-20261111")
    main(p.parse_args().run.resolve())
