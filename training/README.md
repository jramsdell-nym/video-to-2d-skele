# Retrain and compare the batting detectors

The continuation runner fine-tunes one body pose model and one bat-endpoint model. Run these commands from the repository root with the Python environment from the main README. Training uses PyTorch MPS on an Apple silicon Mac. ONNX export also needs the packages in `requirements-onnx.txt`.

The team dataset and prior scored runs are private. Set `SWING_LABELER_PREFIX` to the approved Cloud Storage labeler prefix and use an authorized `gcloud` installation. Set `PRIOR_RUN` to a local completed run with its original checkpoints, frozen split rows, predictions and source snapshot. Set `BASELINE` to the corresponding public release ID. Set `RELEASE` to a new ID such as `2026-11-19-r2`. Each run writes to a new directory under `outputs/`.

```sh
RUN="outputs/training-$RELEASE"
python tools/collect_training_snapshot.py fetch --output "$RUN/dataset"
python tools/collect_training_snapshot.py audit --output "$RUN/dataset" --compare-to "$PRIOR_RUN/dataset"
python tools/collect_training_snapshot.py images --output "$RUN/dataset"
python tools/prepare_fresh_training.py --prior-run "$PRIOR_RUN" --baseline-release "$BASELINE" \
  --snapshot "$RUN/dataset" --output "$RUN" --recipe-profile short-continuation --approve-for-training
for task in body bat; do
  python training/run_cycle.py prepare --run "$RUN/$task" --protocol "$RUN/$task-protocol.json"
  python training/run_cycle.py train --run "$RUN/$task"
  python training/run_cycle.py predict --run "$RUN/$task" --model previous
  python training/run_cycle.py predict --run "$RUN/$task" --model candidate
done
python tools/make_package_spec.py --run "$RUN" --release "$RELEASE" --output "$RUN/package-spec.json"
python tools/package_training_run.py --spec "$RUN/package-spec.json" --output-root "outputs/releases/$RELEASE"
```

The audit validates reviewed targets and original video timing. The split builder keeps validation and evaluation rows fixed, excludes their sessions from training, and verifies the warm-start checkpoint. The short continuation recipe uses ten epochs at learning rate 0.000025. The scorer compares both final detectors with the previous checkpoint on the same five evaluation swings. Results include missed predictions in the denominator and report wrists separately.

The release builder creates a ZIP with PyTorch and static ONNX models, a local inference runner, aggregate scores and checksums. It checks the trained tensors, compares native and ONNX predictions on 15 frames, and tests inference from an extracted copy. Commit the four sanitized model files through Git LFS together with their hashes and aggregate release records. Raw annotations, source video and run directories stay in private storage.
