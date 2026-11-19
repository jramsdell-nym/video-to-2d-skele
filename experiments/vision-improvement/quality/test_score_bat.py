import copy
import unittest

from score_bat import score


def truth(i=0, clip="a", status="reviewed"):
    return {"clipId": clip, "videoSHA256": clip * 64, "frameIndex": i,
            "status": status, "session": "s", "camera": "cam5", "points": {
                "tip": {"state": "visible", "pixelX": 100, "pixelY": 0},
                "knob": {"state": "visible", "pixelX": 0, "pixelY": 0}}}


def prediction(i=0, clip="a", xy=None):
    return {"clipId": clip, "frameIndex": i, "detected": True,
            "prediction": xy or [[100, 0], [0, 0]], "score": .0001}


class ScorerTest(unittest.TestCase):
    def test_missing_detection_stays_in_denominator(self):
        r = score([truth(), truth(1)], [prediction()])
        self.assertEqual(r["missedFrames"], 1)
        self.assertEqual(r["meanEndpointErrorPixelsOnLocalizedEndpoints"], 0)
        self.assertEqual(r["bothEndpointsPCK25Pixels"], .5)
        self.assertEqual(r["endpointPCK0.1BatLength"], .5)
        self.assertEqual(r["directedAngleWithin5DegreesRate"], .5)

    def test_swap_is_not_axis_alignment_success(self):
        r = score([truth()], [prediction(xy=[[0, 0], [100, 0]])])
        self.assertEqual(r["swapCandidates"], 1)
        self.assertEqual(r["meanDirectedAngleDegreesOnValidPredictions"], 180)
        self.assertEqual(r["bothEndpointsPCK50Pixels"], 0)

    def test_negative_not_inferred_from_skipped(self):
        absent, skipped = truth(1, status="absent"), truth(2, status="skipped")
        for row in (absent, skipped):
            row["points"] = {}
        r = score([truth(), absent, skipped], [prediction(), prediction(1), prediction(2)])
        self.assertEqual(r["negativeFrames"], 1)
        self.assertEqual(r["negativeFalsePositiveRate"], 1)
        self.assertEqual(r["positiveFrames"], 1)
        self.assertEqual(r["excludedByStatus"], {"skipped": 1})
        self.assertIsNone(score([truth()], [prediction()])["negativeFalsePositiveRate"])

    def test_missing_negative_inference_is_not_true_negative(self):
        row = truth(status="absent")
        row["points"] = {}
        r = score([row], [])
        self.assertEqual(r["missingNegativePredictions"], 1)
        self.assertEqual(r["evaluatedNegativeFrames"], 0)
        self.assertIsNone(r["negativeFalsePositiveRate"])
        verified = prediction()
        verified["detected"] = False
        self.assertEqual(score([row], [verified])["negativeFalsePositiveRate"], 0)

    def test_partial_and_nan_are_not_success(self):
        row = truth()
        row["points"]["knob"] = {"state": "occluded"}
        r = score([row], [prediction(xy=[[float("nan"), 0], [0, 0]])])
        self.assertEqual(r["visibleTruthEndpoints"], 1)
        self.assertEqual(r["missingEndpoints"], 1)
        self.assertEqual(r["endpointPCK25Pixels"], 0)
        self.assertIsNone(r["bothEndpointsPCK25Pixels"])

    def test_degenerate_length_never_divides_by_zero(self):
        row = truth()
        row["points"]["tip"]["pixelX"] = 0
        r = score([row], [prediction(xy=[[0, 0], [0, 0]])])
        self.assertEqual(r["bothEndpointsPCK25Pixels"], 1)
        self.assertEqual(r["normalizableFrames"], 0)
        self.assertIsNone(r["bothEndpointsPCK0.1BatLength"])

    def test_confidence_has_no_accuracy_credit(self):
        p = prediction(xy=[[300, 0], [200, 0]])
        p["score"] = .99999
        self.assertEqual(score([truth()], [p])["bothEndpointsPCK25Pixels"], 0)
        self.assertEqual(score([truth()], [prediction()])["bothEndpointsPCK25Pixels"], 1)

    def test_video_macro_is_not_dominated_by_frame_count(self):
        rows = [truth(i) for i in range(4)] + [truth(clip="b")]
        preds = [prediction(i) for i in range(4)]
        r = score(rows, preds)
        self.assertEqual(r["bothEndpointsPCK25Pixels"], .8)
        self.assertEqual(r["macroVideoBothEndpointsPCK25Pixels"], .5)
        self.assertEqual(r["macroVideoBothEndpointsPCK0.1BatLength"], .5)
        self.assertEqual(r["bothEndpointsWithin0.1BatLength"], 4)

    def test_duplicate_or_unknown_keys_fail(self):
        with self.assertRaises(ValueError):
            score([truth(), truth()], [])
        with self.assertRaises(ValueError):
            score([truth()], [prediction(), prediction()])
        with self.assertRaises(ValueError):
            score([truth()], [prediction(9)])

    def test_draft_truth_never_scores(self):
        r = score([truth(status="draft")], [prediction()])
        self.assertEqual(r["positiveFrames"], 0)
        self.assertIsNone(r["bothEndpointsPCK25Pixels"])

    def test_thresholds_are_inclusive(self):
        r = score([truth()], [prediction(xy=[[125, 0], [25, 0]])])
        self.assertEqual(r["bothEndpointsPCK25Pixels"], 1)
        self.assertEqual(r["bothEndpointsPCK10Pixels"], 0)


if __name__ == "__main__":
    unittest.main()
