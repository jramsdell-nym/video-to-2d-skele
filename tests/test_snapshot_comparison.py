import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("snapshot_wrapper", Path(__file__).resolve().parents[1] / "tools/collect_training_snapshot.py")
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)


class SnapshotComparisonTests(unittest.TestCase):
    def test_private_output_cannot_target_tracked_archive(self):
        repo = Path(__file__).resolve().parents[1]
        with self.assertRaises(ValueError):
            snapshot.private_output(repo / "experiments/new-data", repo)
        self.assertEqual(snapshot.private_output(repo / "outputs/new-data", repo), repo / "outputs/new-data")

    def test_new_removed_and_revised_labels_are_distinguished(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, new = root / "old", root / "new"
            old.mkdir()
            new.mkdir()
            for task in ("body", "bat"):
                p = {"x": .2, "y": .3, "visibility": 2, "source": "manual"}
                points = [p] if task == "body" else {"tip": p, "knob": p}
                rows = [{"videoSHA256": "a" * 64, "frameIndex": i, "frame": {"index": i},
                         "width": 100, "height": 80, "points": copy.deepcopy(points)} for i in range(3)]
                changed = copy.deepcopy(rows[:2])
                # A changed review timestamp/source is not a changed target.
                changed[0]["reviewedAt"] = "new-review"
                point0 = changed[0]["points"][0] if task == "body" else changed[0]["points"]["tip"]
                point0["source"] = "reviewed-model"
                point1 = changed[1]["points"][0] if task == "body" else changed[1]["points"]["tip"]
                point1["x"] = .4
                added = copy.deepcopy(rows[2])
                added["videoSHA256"] = "b" * 64
                changed.append(added)
                (old / f"{task}-rows-unresolved.json").write_text(json.dumps(rows))
                (new / f"{task}-rows-unresolved.json").write_text(json.dumps(changed))
            report = snapshot.compare_snapshots(new, old)
            for result in report.values():
                self.assertEqual(result["newFrameIdentities"], 1)
                self.assertEqual(result["removedFrameIdentities"], 1)
                self.assertEqual(result["revisedExistingTargets"], 1)
                self.assertEqual(result["netAddedFrames"], 0)
                self.assertFalse(result["larger"])
                self.assertEqual(result["currentVideos"], 2)


if __name__ == "__main__":
    unittest.main()
