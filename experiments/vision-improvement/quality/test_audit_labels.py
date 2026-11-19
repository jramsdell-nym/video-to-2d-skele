import copy
import unittest

from audit_labels import audit, split_audit


def clip():
    return {"id": "a", "sha256": "hash-a", "title": "Player A", "session": "s", "trial": "t", "width": 200, "height": 100,
            "frames": [{"index": i, "pts": i, "timeBase": "1/30", "time": i/30} for i in range(3)]}


def rows():
    c = clip()
    return [{"clipId": "a", "videoSHA256": "hash-a", "session": "s", "trial": "t", "frameIndex": i,
             "source": c["frames"][i], "status": "reviewed", "imageWidth": 200, "imageHeight": 100,
             "points": {"tip": {"state": "visible", "x": .5, "y": .5, "pixelX": 100, "pixelY": 50},
                        "knob": {"state": "visible", "x": .25, "y": .5, "pixelX": 50, "pixelY": 50}}}
            for i in range(3)]


class AuditTest(unittest.TestCase):
    def test_stationary_valid_rows_have_no_flags(self):
        self.assertEqual(audit(rows(), {"a": clip()})["flags"], [])

    def test_single_frame_reversal_flags_both_transitions_without_mutation(self):
        r = rows()
        r[1]["points"]["tip"], r[1]["points"]["knob"] = r[1]["points"]["knob"], r[1]["points"]["tip"]
        before = copy.deepcopy(r)
        flags = audit(r, {"a": clip()})["flags"]
        self.assertEqual(sum(f["reason"] == "possible_temporal_endpoint_swap" for f in flags), 2)
        self.assertEqual(r, before)

    def test_invalid_time_prevents_temporal_judgment(self):
        r = rows()
        for row in r:
            row["source"] = {"index": row["frameIndex"]}
        result = audit(r, {"a": clip()})
        self.assertEqual(result["temporalPairsChecked"], 0)
        self.assertEqual(result["temporalTriplesChecked"], 0)

    def test_shared_hash_across_clip_ids_is_split_leakage(self):
        r = rows()[:2]
        r[1]["clipId"] = "b"
        result = split_audit(r, {"a": "train", "b": "test"}, {"a": clip(), "b": clip()})
        self.assertEqual(result["videoSHA256"], {"hash-a": ["test", "train"]})
        self.assertEqual(result["session"], {"s": ["test", "train"]})

    def test_coordinate_mismatch_is_blocking(self):
        r = rows()[:1]
        r[0]["points"]["tip"]["pixelX"] = 99
        flags = audit(r, {"a": clip()})["flags"]
        self.assertEqual(flags[0]["reason"], "pixel_normalized_mismatch")
        self.assertEqual(flags[0]["priority"], "block")


if __name__ == "__main__":
    unittest.main()
