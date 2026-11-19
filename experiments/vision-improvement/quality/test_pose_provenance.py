import unittest

from audit_pose_export import point_valid, summarize


def record(method="frame"):
    p = {"x": .5, "y": .5, "visibility": 2, "source": "manual"}
    return {"status": "reviewed", "reviewMethod": method, "points": [p.copy() for _ in range(12)],
            "batTip": p.copy(), "batKnob": {**p, "reviewedAt": "2026-11-10T00:00:00Z"}}


class ProvenanceTest(unittest.TestCase):
    def test_frame_review_does_not_erase_interpolation(self):
        r = record()
        r["points"][3]["source"] = "interpolated"
        result = summarize([r])
        self.assertEqual(result["explicitFrameReview"]["frames"], 1)
        self.assertEqual(result["explicitFrameReview"]["noInterpolatedJoints"], 0)
        self.assertEqual(result["explicitFrameReview"]["allManualJoints"], 0)

    def test_body_review_does_not_accept_knob(self):
        r = record()
        del r["batKnob"]["reviewedAt"]
        self.assertEqual(summarize([r])["batEligibleVisibleReviewedPairs"], 0)

    def test_manual_batch_not_a_frame_review(self):
        self.assertEqual(summarize([record("smoothed-batch")])["explicitFrameReview"]["allManualJoints"], 0)

    def test_invalid_point_rejected(self):
        self.assertFalse(point_valid({"x": True, "y": .5}))
        self.assertFalse(point_valid({"x": float("nan"), "y": .5}))


if __name__ == "__main__":
    unittest.main()
