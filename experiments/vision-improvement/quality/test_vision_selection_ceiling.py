import copy
import unittest

from vision_selection_ceiling import analyze, choose_candidate, load_scorer


SCORER = load_scorer()


def row(frame=0, scale=1):
    coordinates = [(40, 20), (60, 20), (35, 30), (65, 30), (30, 35), (70, 35),
                   (40, 40), (60, 40), (40, 60), (60, 60), (40, 80), (60, 80)]
    return {"videoSHA256": "test", "frameIndex": frame, "session": "test", "width": 100, "height": 100,
            "status": "reviewed", "reviewMethod": "frame", "points": [
                {"x": x * scale / 100, "y": y * scale / 100, "visibility": 2, "source": "manual"} for x, y in coordinates]}


def candidate(r, offset=0):
    return {"prediction": [[p["x"] * 100 + offset, p["y"] * 100] for p in r["points"]], "keypointConfidences": [.9] * 12}


class VisionOracleTest(unittest.TestCase):
    def test_objective_and_deterministic_ties(self):
        r = row()
        c = candidate(r)
        index, counts = choose_candidate(r, [candidate(r, 20), c, copy.deepcopy(c)], 60, SCORER)
        self.assertEqual((index, counts), (1, [0, 12, 12]))

    def test_gate_visibility_and_bounds_match_scorer(self):
        r = row()
        r["points"][3]["visibility"] = 1
        c = candidate(r)
        c["keypointConfidences"][0] = .299
        c["keypointConfidences"][1] = .3
        c["prediction"][2] = [float("nan"), 0]
        c["prediction"][4] = [-.1, 0]
        c["prediction"][5] = [100.1, 0]
        # Huge scale admits all distances, but invalid/out-of-image/low confidence stay excluded.
        self.assertEqual(choose_candidate(r, [c], 10000, SCORER)[1], [7])

    def test_no_candidates_are_failures_not_removed(self):
        r = row()
        raw = {"videoSHA256": "test", "frameIndex": 0, "candidateCount": 0, "candidates": []}
        current = {"videoSHA256": "test", "frameIndex": 0, "detected": False}
        out = analyze([r], [raw], [current], SCORER)
        self.assertEqual(out["ORACLE_groundTruthSelectedCeiling"]["all"]["within0.05BodyScale"], 0)
        self.assertEqual(out["ORACLE_groundTruthSelectedCeiling"]["all"]["normalizableJoints"], 12)

    def test_strict_slice_retains_full_reference_scale(self):
        rows = [row(0, .25), row(1, 1), row(2, 1)]
        for r in rows[1:]:
            r["reviewMethod"] = "smoothed-batch"
        candidates = [candidate(r, 2) for r in rows]
        raw = [{"videoSHA256": "test", "frameIndex": i, "candidateCount": 1, "candidates": [c]} for i, c in enumerate(candidates)]
        current = [{"videoSHA256": "test", "frameIndex": i, "detected": True, **c} for i, c in enumerate(candidates)]
        snapshot = copy.deepcopy((rows, raw, current))
        out = analyze(rows, raw, current, SCORER)
        strict = out["ORACLE_groundTruthSelectedCeiling"]["frameReviewedNoInterpolation"]
        self.assertEqual(strict["within0.05BodyScale"], 12)  # .05*60=3; subset-only .05*15=.75 would fail.
        self.assertEqual(out["bodyScalePixelsByVideo"]["test"], 60)
        self.assertEqual((rows, raw, current), snapshot)

    def test_missing_and_duplicate_records_rejected(self):
        with self.assertRaises(ValueError):
            analyze([row()], [], [], SCORER)
        with self.assertRaises(ValueError):
            analyze([row(), row()], [], [], SCORER)


if __name__ == "__main__":
    unittest.main()
