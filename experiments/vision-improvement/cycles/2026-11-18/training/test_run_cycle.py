import importlib.util
from pathlib import Path
import unittest
import contextlib
import io
import tempfile

spec = importlib.util.spec_from_file_location("cycle", Path(__file__).with_name("run_cycle.py"))
cycle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cycle)


class LabelContractTests(unittest.TestCase):
    def body(self):
        return {"status": "reviewed", "box": [.1, .2, .7, .9], "points": [{"x": .1 + i / 100, "y": .3 + i / 100, "visibility": 2} for i in range(12)]}

    def bat(self):
        return {"status": "reviewed", "imageWidth": 1920, "imageHeight": 1080, "points": {name: {"x": x, "y": y, "pixelX": x * 1920, "pixelY": y * 1080, "state": "visible"} for name, x, y in (("tip", .4, .3), ("knob", .5, .6))}}

    def test_body_preserves_head_and_order(self):
        row = self.body()
        values = list(map(float, cycle.label(row, "body").split()))
        self.assertEqual(len(values), 56)
        self.assertEqual(values[5:20], [0] * 15)
        self.assertEqual(values[20:23], [.1, .3, 2])
        self.assertEqual(values[-3:], [row["points"][-1]["x"], row["points"][-1]["y"], 2])

    def test_missing_body_point_cannot_train_unknown_coordinate(self):
        row = self.body()
        row["points"][0] = {"visibility": 0}
        self.assertEqual(list(map(float, cycle.label(row, "body").split()))[20:23], [0, 0, 0])

    def test_invisible_but_localized_point_preserves_visibility_one(self):
        row = self.body()
        row["points"][0]["visibility"] = 1
        self.assertEqual(list(map(float, cycle.label(row, "body").split()))[20:23], [.1, .3, 1])

    def test_body_rejects_outside_normalized_space(self):
        row = self.body()
        row["points"][0]["x"] = 1920
        with self.assertRaises(AssertionError):
            cycle.label(row, "body")

    def test_bat_order_and_bbox_padding(self):
        values = list(map(float, cycle.label(self.bat(), "bat").split()))
        self.assertEqual(values[5:], [.4, .3, 2, .5, .6, 2])
        self.assertAlmostEqual(values[3], .1 + 48 / 1920)
        self.assertAlmostEqual(values[4], .3 + 48 / 1080)

    def test_bat_rejects_pixel_mismatch(self):
        row = self.bat()
        row["points"]["tip"]["pixelX"] += 1
        with self.assertRaises(AssertionError):
            cycle.label(row, "bat")

    def test_bat_incomplete_is_not_negative(self):
        row = self.bat()
        row["points"]["tip"]["state"] = "unknown"
        with self.assertRaises(AssertionError):
            cycle.label(row, "bat")

    def test_localized_occluded_bat_preserves_visibility_one(self):
        row = self.bat()
        row["points"]["tip"].update(state="occluded", visibility=1)
        self.assertEqual(list(map(float, cycle.label(row, "bat").split()))[5:8], [.4, .3, 1])

    def test_only_explicit_reviewed_absence_is_negative(self):
        with self.assertRaises(AssertionError):
            cycle.label({"status": "absent"}, "bat")
        self.assertEqual(cycle.label({"status": "absent", "explicitAbsenceReviewed": True}, "bat"), "")


class PreparationContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        from PIL import Image
        self.rows = {}
        for split, tag in zip(("train", "val", "evaluation"), ("a", "b", "c")):
            image = self.root / f"{split}.png"
            Image.new("RGB", (100, 80), (40, 60, 90)).save(image)
            row = LabelContractTests().body()
            row.update(videoSHA256=tag * 64, frameIndex=0, session="session", trial=tag, canonicalGroup=f"session/{tag}", width=100, height=80, imagePath=str(image), imageSHA256=cycle.sha(image))
            self.rows[split] = [row]
        weights = self.root / "weights.pt"
        weights.write_bytes(b"fixture-not-used-for-inference")
        model = {"path": str(weights), "sha256": cycle.sha(weights)}
        self.protocol = {"approvedForTraining": True, "task": "body", "initial": model, "baselines": {"generic": model}, "evaluationExclusionVideos": [], "evaluationExclusionGroups": [], "rows": {split: str(self.root / f"{split}.json") for split in self.rows}}

    def prepare(self):
        for split, rows in self.rows.items():
            cycle.save(self.protocol["rows"][split], rows)
        path = self.root / "protocol.json"
        cycle.save(path, self.protocol)
        with contextlib.redirect_stdout(io.StringIO()):
            cycle.prepare(self.root / "run", path)

    def test_evaluation_is_absent_from_training_yaml_and_inputs_are_frozen(self):
        self.prepare()
        run = self.root / "run"
        self.assertNotIn("evaluation", (run / "dataset/data.yaml").read_text())
        self.assertFalse((run / "dataset/images/evaluation").exists())
        cycle.validate(run)
        (run / "dataset/labels/train" / ("a" * 64 + "_0.txt")).write_text("changed\n")
        with self.assertRaisesRegex(AssertionError, "Input changed"):
            cycle.validate(run)

    def test_validation_evaluation_same_trial_different_video_is_rejected(self):
        self.rows["evaluation"][0]["canonicalGroup"] = "session/b"
        with self.assertRaisesRegex(AssertionError, "val/evaluation leakage"):
            self.prepare()

    def test_baseline_training_trial_is_rejected_even_with_new_video_hash(self):
        self.protocol["evaluationExclusionGroups"] = ["session/c"]
        with self.assertRaisesRegex(AssertionError, "synchronized evaluation trial"):
            self.prepare()

    def test_image_geometry_mismatch_is_rejected(self):
        self.rows["train"][0]["width"] = 200
        with self.assertRaisesRegex(AssertionError, "geometry mismatch"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
