import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("fresh_training", Path(__file__).resolve().parents[1] / "tools/prepare_fresh_training.py")
fresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fresh)


def row(tag, session, frame=0, task="body"):
    value = {"videoSHA256": tag * 64, "frameIndex": frame, "session": session, "trial": "trial-" + tag,
             "canonicalGroup": session + "/trial-" + tag, "trialKeys": ["trial:" + tag], "status": "reviewed",
             "frame": {"index": frame, "pts": frame * 256, "timeBase": "1/15360", "time": frame / 60},
             "width": 100, "height": 80, "box": [.1, .1, .9, .9]}
    if task == "body":
        value["points"] = [{"x": .2 + i / 100, "y": .3, "visibility": 2} for i in range(12)]
    else:
        value.update(imageWidth=100, imageHeight=80, points={name: {"x": x, "y": .3, "pixelX": x * 100, "pixelY": 24, "state": "visible", "visibility": 2} for name, x in (("tip", .2), ("knob", .6))})
    return value


class FreshSelectionTests(unittest.TestCase):
    def test_all_new_frames_in_old_validation_and_evaluation_sessions_are_excluded(self):
        protected = fresh.protected_identity([row("a", "validation"), row("b", "evaluation")])
        rows = [row("c", "validation", 100), row("d", "evaluation", 50), row("e", "new-session")]
        selected, excluded = fresh.choose_training(rows, protected)
        self.assertEqual([r["session"] for r in selected], ["new-session"])
        self.assertEqual([r["reason"] for r in excluded], ["protected_session", "protected_session"])

    def test_cross_camera_alias_is_excluded_by_trial_keys_even_after_session_rename(self):
        protected = fresh.protected_identity([row("a", "evaluation")])
        alias = row("b", "renamed-session")
        alias["trialKeys"] = ["trial:a", "other-key"]
        selected, excluded = fresh.choose_training([alias, row("c", "new")], protected)
        self.assertEqual(len(selected), 1)
        self.assertEqual(excluded[0]["reason"], "protected_trial_key")

    def test_video_identity_stays_excluded_despite_metadata_rename(self):
        protected = fresh.protected_identity([row("a", "evaluation")])
        _, excluded = fresh.choose_training([row("a", "renamed"), row("c", "new")], protected)
        self.assertEqual(excluded[0]["reason"], "protected_video")

    def test_unreviewed_and_duplicates_fail_instead_of_silently_becoming_training_data(self):
        valid = row("a", "train")
        bad = copy.deepcopy(valid)
        bad["status"] = "draft"
        for rows in ([bad], [valid, valid]):
            with self.assertRaises(ValueError):
                fresh.choose_training(rows, fresh.protected_identity([]))

    def test_latest_training_set_counts_added_removed_and_changed_supervision(self):
        old = [row("a", "train"), row("b", "train")]
        new = [row("a", "train"), row("c", "train")]
        new[0]["points"][0]["x"] = .4
        changes = fresh.supervised_changes(old, new, "body")
        self.assertEqual(changes["addedFrameIdentities"], 1)
        self.assertEqual(changes["removedFrameIdentities"], 1)
        self.assertEqual(changes["retainedFramesWithChangedSupervision"], 1)


class FreshFreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.source = self.root / "historical"
        self.snapshot = self.repo / "outputs/fresh/dataset"
        self.output = self.repo / "outputs/fresh"
        self.defaults, self.current = {}, {}
        for task in ("body", "bat"):
            model = self.repo / f"models/{task}.pt"
            model.parent.mkdir(parents=True, exist_ok=True)
            model.write_bytes(task.encode())
            self.defaults[task] = {"checkpoint": str(model.relative_to(self.repo)), "sha256": fresh.sha(model)}
            run = self.source / "training" / task
            run.mkdir(parents=True)
            inputs = []
            for split, tag in (("train", "a"), ("val", "b"), ("evaluation", "c")):
                path = run / f"{split}-rows.json"
                # Deliberately nonstandard whitespace tests exact byte retention.
                path.write_text(json.dumps([row(tag, split, task=task)], separators=(",", ":")) + "\n\n")
                inputs.append({"path": str(path), "sha256": fresh.sha(path)})
            fresh.write(run / "inputs.json", inputs)
            fresh.write(run / "manifest.json", {"inputManifestSHA256": fresh.sha(run / "inputs.json")})
            fresh.write(run / "training-status.json", {"checkpointSHA256": fresh.sha(model)})
            fresh.write(run / "protocol.json", {"evaluationExclusionVideos": ["d" * 64], "evaluationExclusionGroups": ["legacy/trial-d"], "limits": ["Provisional labels"]})
            self.current[task] = [row("a", "train", task=task), row("e", "new", task=task), row("f", "val", 88, task)]
        fresh.write(self.repo / "models/manifest.json", {"defaults": self.defaults})

    def freeze(self):
        with patch.object(fresh, "validate_snapshot", return_value=(self.current, {"verifiedImages": 3, "verifiedSourceVideos": 3})):
            return fresh.prepare(self.source, self.snapshot, self.output, self.repo)

    def test_freezes_exact_benchmark_bytes_and_all_lineage_without_approving_fit(self):
        receipt = self.freeze()
        for task in ("body", "bat"):
            protocol = fresh.read(self.output / f"{task}-protocol.json")
            self.assertFalse(protocol["approvedForTraining"])
            self.assertEqual(protocol["recipe"]["lr0"], .00005)
            self.assertIn("a" * 64, protocol["evaluationExclusionVideos"])
            self.assertIn("d" * 64, protocol["evaluationExclusionVideos"])
            self.assertIn("e" * 64, protocol["evaluationExclusionVideos"])
            for split in ("val", "evaluation"):
                old = self.source / f"training/{task}/{split}-rows.json"
                new = Path(protocol["rows"][split])
                self.assertEqual(old.read_bytes(), new.read_bytes())
            self.assertEqual(receipt["tasks"][task]["trainingChanges"]["freshTrainingFrames"], 2)
            self.assertEqual(receipt["tasks"][task]["excludedFreshFrames"], 1)
        with self.assertRaises(FileExistsError):
            self.freeze()

    def test_changed_historical_split_is_rejected_before_writing_outputs(self):
        path = self.source / "training/body/evaluation-rows.json"
        path.write_text("[]")
        with self.assertRaisesRegex(ValueError, "frozen hashes"):
            self.freeze()
        self.assertFalse((self.output / "body-protocol.json").exists())

    def test_scored_run_continues_its_own_final_checkpoints(self):
        previous = self.source / "training"
        for task in ("body", "bat"):
            run = previous / task
            weights = run / "candidate/weights/last.pt"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"distinct final " + task.encode())
            status = {"state": "complete", "checkpointSHA256": fresh.sha(weights)}
            (run / "training-status.json").write_text(json.dumps(status))
            fresh.write(run / "candidate-prediction-manifest.json", {
                "checkpointSHA256": fresh.sha(weights), "rowsSHA256": fresh.sha(run / "evaluation-rows.json")})
            inputs = fresh.read(run / "inputs.json")
            inputs.append({"path": str(run / "protocol.json"), "sha256": fresh.sha(run / "protocol.json")})
            (run / "inputs.json").write_text(json.dumps(inputs))
            (run / "manifest.json").write_text(json.dumps({"inputManifestSHA256": fresh.sha(run / "inputs.json")}))
        with patch.object(fresh, "validate_snapshot", return_value=(self.current, {})):
            fresh.prepare(previous, self.snapshot, self.output, self.repo, prior_run=True,
                          recipe_profile="short-continuation")
        for task in ("body", "bat"):
            protocol = fresh.read(self.output / f"{task}-protocol.json")
            self.assertEqual(protocol["initial"]["sha256"], fresh.sha(previous / task / "candidate/weights/last.pt"))
            self.assertEqual(protocol["initial"], protocol["baselines"]["previous"])
            self.assertIn("legacy/trial-d", protocol["evaluationExclusionGroups"])
            self.assertEqual(Path(protocol["rows"]["evaluation"]).read_bytes(),
                             (previous / task / "evaluation-rows.json").read_bytes())
            self.assertEqual((protocol["recipe"]["epochs"], protocol["recipe"]["lr0"]), (10, .000025))

    def test_scored_run_refuses_a_different_fit_or_unfinished_checkpoint(self):
        run = self.root / "run"
        weights = run / "candidate/weights/last.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"final")
        fresh.write(run / "training-status.json", {"state": "training", "checkpointSHA256": fresh.sha(weights)})
        with self.assertRaisesRegex(ValueError, "completed final"):
            fresh.previous_checkpoint(run)
        (run / "training-status.json").write_text(json.dumps({"state": "complete", "checkpointSHA256": fresh.sha(weights)}))
        fresh.write(run / "candidate-prediction-manifest.json", {"checkpointSHA256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "used for evaluation"):
            fresh.previous_checkpoint(run)

    def test_unknown_baseline_checkpoint_is_rejected(self):
        (self.repo / "models/body.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Nov18 checkpoint"):
            self.freeze()


if __name__ == "__main__":
    unittest.main()
