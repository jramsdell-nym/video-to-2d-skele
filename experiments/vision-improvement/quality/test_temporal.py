import copy
import unittest

from analyze_temporal import analyze


def truth(index, pts=None, tip=100, knob=0):
    pts = index*10 if pts is None else pts
    return {"clipId": "a", "videoSHA256": "hash-a", "frameIndex": index, "status": "reviewed",
            "source": {"index": index, "pts": pts, "timeBase": "1/1000", "time": pts/1000},
            "points": {"tip": {"state": "visible", "pixelX": tip, "pixelY": 20, "source": "manual"},
                       "knob": {"state": "visible", "pixelX": knob, "pixelY": 20, "source": "manual"}}}


def prediction(row, offset=0, detected=True):
    return {"clipId": row["clipId"], "frameIndex": row["frameIndex"], "detected": detected,
            "score": .9, "endpointConfidences": [.9, .9],
            "prediction": [[row["points"][n]["pixelX"]+offset, row["points"][n]["pixelY"]] for n in ("tip", "knob")]}


class TemporalTest(unittest.TestCase):
    def test_irregular_exact_timing_controls_rate(self):
        rows = [truth(0, 0), truth(1, 10), truth(2, 40)]
        preds = [prediction(r, offset) for r, offset in zip(rows, [0, 1, 4])]
        out = analyze(rows, preds)
        tip = [p for p in out["residualPairs"] if p["endpoint"] == "tip"]
        self.assertEqual([p["deltaSeconds"] for p in tip], [.01, .03])
        self.assertEqual([p["pixelsPerPresentationSecond"] for p in tip], [100, 100])

    def test_sparse_frame_gap_breaks_residuals_and_dropout_runs(self):
        rows = [truth(0), truth(2)]
        out = analyze(rows, [prediction(r, detected=False) for r in rows])
        self.assertEqual(out["consecutiveFramePairs"], 0)
        self.assertEqual(out["residualPairs"], [])
        self.assertEqual(out["dropouts"]["detection"]["runs"], 2)
        self.assertEqual(out["dropouts"]["detection"]["longestRunFrames"], 1)
        self.assertEqual(out["dropouts"]["detection"]["longestObservedSpanSeconds"], 0)
        self.assertEqual(out["breaks"][0]["reason"], "nonconsecutive_source_frame")

    def test_correct_rapid_motion_cancels(self):
        rows = [truth(0), truth(1, tip=1000, knob=850), truth(2, tip=-400, knob=-500)]
        out = analyze(rows, [prediction(r) for r in rows])
        self.assertEqual(out["residualChange"]["all"]["deltaPixels"]["maximum"], 0)
        self.assertEqual(out["reviewFlags"], [])

    def test_miss_resume_does_not_bridge_residual(self):
        rows = [truth(0, 0), truth(1, 10), truth(2, 30), truth(3, 60)]
        preds = [prediction(r, detected=i not in (1, 2)) for i, r in enumerate(rows)]
        out = analyze(rows, preds)
        detection = [r for r in out["dropoutRuns"] if r["kind"] == "detection"][0]
        self.assertEqual(detection["missingFrames"], 2)
        self.assertAlmostEqual(detection["observedSpanSeconds"], .02)
        self.assertAlmostEqual(detection["timeUntilResumeSeconds"], .05)
        self.assertFalse(detection["leftCensored"])
        self.assertFalse(detection["rightCensored"])
        self.assertEqual(out["residualChange"]["tip"]["measuredEndpointPairs"], 0)
        self.assertEqual(out["residualChange"]["tip"]["unmeasuredEndpointPairs"], 3)

    def test_invalid_pts_breaks_timing(self):
        rows = [truth(0), truth(1), truth(2)]
        rows[1]["source"]["time"] = 99
        out = analyze(rows, [prediction(r) for r in rows])
        self.assertEqual(out["invalidTimingFrames"], 1)
        self.assertEqual(out["consecutiveFramePairs"], 0)

    def test_endpoint_gate_is_distinct_from_detection(self):
        rows = [truth(0), truth(1)]
        preds = [prediction(r) for r in rows]
        preds[1]["endpointConfidences"][0] = .1
        out = analyze(rows, preds, .25, .25)
        self.assertEqual(out["dropouts"]["detection"]["missingFrames"], 0)
        self.assertEqual(out["dropouts"]["tip"]["missingFrames"], 1)
        self.assertEqual(out["dropouts"]["knob"]["missingFrames"], 0)

    def test_missing_inference_and_interpolated_reference_are_explicit(self):
        rows = [truth(0), truth(1), truth(2)]
        rows[1]["points"]["tip"]["source"] = "interpolated"
        out = analyze(rows, [prediction(rows[0]), prediction(rows[1], offset=40)])
        self.assertEqual(out["framesWithInterpolatedTruth"], 1)
        self.assertEqual(out["missingPredictionRecords"], 1)
        self.assertTrue(any("interpolated reference=True" in f["evidence"] for f in out["reviewFlags"]))

    def test_reversed_endpoints_are_review_candidates(self):
        row = truth(0)
        p = prediction(row)
        p["prediction"].reverse()
        out = analyze([row], [p])
        self.assertEqual(out["flagCounts"], {"possible_endpoint_swap": 1})

    def test_inputs_are_not_mutated(self):
        rows = [truth(0), truth(1)]
        preds = [prediction(r) for r in rows]
        original = copy.deepcopy((rows, preds))
        analyze(rows, preds, .25, .25)
        self.assertEqual((rows, preds), original)


if __name__ == "__main__":
    unittest.main()
