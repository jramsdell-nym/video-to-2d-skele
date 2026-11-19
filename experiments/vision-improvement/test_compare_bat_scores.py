import copy
import unittest
from compare_bat_scores import compare


def report(errors, clips=None):
    clips = clips or ["a"] * len(errors)
    return {"details": [{"clipId": clip, "frameIndex": i, "kind": "positive",
                         "bothTruthVisible": True, "truthLengthPixels": 100,
                         "visibleTruthEndpoints": 2, "errorsPixels": pair}
                        for i, (clip, pair) in enumerate(zip(clips, errors))],
            "missedFrames": sum(all(e is None for e in p) for p in errors),
            "missingEndpoints": sum(e is None for p in errors for e in p),
            "swapCandidates": 0, "negativeFalsePositiveRate": None}


class ComparisonTests(unittest.TestCase):
    def test_misses_stay_in_denominator_and_improvement_is_not_release(self):
        result = compare(report([[1, 1], [None, None]]), report([[1, 1], [10, 10]]))
        self.assertEqual(result["overall"]["baselineRate"], .5)
        self.assertEqual(result["overall"]["candidateRate"], 1)
        self.assertEqual(result["overall"]["candidateWins"], 1)
        self.assertEqual(result["recommendation"], "advance_to_broader_evaluation")
        self.assertFalse(result["releaseAuthorized"])

    def test_one_good_endpoint_is_not_a_pair(self):
        result = compare(report([[1, None]]), report([[1, 11]]))
        self.assertEqual(result["overall"]["bothFail"], 1)

    def test_mismatched_truth_fails(self):
        first = report([[1, 1]])
        second = copy.deepcopy(first)
        second["details"][0]["truthLengthPixels"] = 101
        with self.assertRaises(ValueError):
            compare(first, second)

    def test_mismatched_frames_and_duplicates_fail(self):
        first = report([[1, 1]])
        with self.assertRaises(ValueError):
            compare(first, report([[1, 1], [2, 2]]))
        second = copy.deepcopy(first)
        second["details"] *= 2
        with self.assertRaises(ValueError):
            compare(first, second)

    def test_same_frame_alias_different_video_fails(self):
        first = report([[1, 1]])
        second = copy.deepcopy(first)
        first["details"][0]["videoSHA256"] = "first-source"
        second["details"][0]["videoSHA256"] = "different-source"
        with self.assertRaises(ValueError):
            compare(first, second)

    def test_video_regression_is_not_hidden_by_more_frames_elsewhere(self):
        clips = ["a", "a", "a", "b"]
        result = compare(report([[20, 20], [20, 20], [20, 20], [1, 1]], clips),
                         report([[1, 1], [1, 1], [1, 1], [20, 20]], clips))
        self.assertGreater(result["overall"]["deltaPercentagePoints"], 0)
        self.assertEqual(result["recommendation"], "investigate_tradeoffs")
        self.assertIn("At least one video regressed on primary success", result["reasons"])


if __name__ == "__main__":
    unittest.main()
