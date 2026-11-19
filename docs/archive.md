# Using the source archive

The root video runner and current model inventory are independent of the original local workspace. The imported experiments preserve their source layout so training, scoring, and export logic remain inspectable.

Original source hashes in `SOURCE_INDEX.json` describe the imported versions. Public-copy portability changes are listed below; the original workspace is untouched.

- Environment-specific data preparation and cloud scripts need your own source files and authorization. They do not ship with recordings, labels, original manifests, golden image tensors, or cloud configuration.
- `experiments/vision-improvement/cycles/2026-11-18/training/run_cycle.py` accepts `prepare`, `train`, and `predict` subcommands. Start with `--help` and provide your own protocol/data paths. Its immutable-input checks and fixed-recipe logic are preserved.
- Older training paths under `experiments/2d-tracking/labeler/training/` contain only checkpoints, not their private datasets. Do not interpret a `best.pt` filename as a recommendation; use the explicit defaults in the model manifest.
- `experiments/2d-tracking/models/` retains the original generic YOLO11s-pose and YOLO11m baselines. MediaPipe comparison source is preserved, but MediaPipe model bundles are not YOLO and are not included.
- Apple Vision Swift comparison files require macOS and an appropriate Swift toolchain. Compiled binaries are excluded.
- The original browser verification harness imports services from a separate bat-data-labeler application and requires withheld image-derived test vectors. It documents an export check; it is not the root runner or a self-contained browser app.
- `swing-review/` is the local annotation application source. Its service is local and its cloud integration is read-only. Use your own data directory, cloud prefix, and gcloud installation. It is not populated with the original player's records.
- `exports/yolo-bat-model-2026-11-12/` remains a self-contained image-inference package with its original instructions, requirements, and checksum verification.

Public-copy changes remove hard-coded account identity and named-player assumptions from preparation utilities; cloud paths and executable locations are configurable. These changes do not alter any model bytes. Local source-path defaults remaining in archival scripts identify missing research inputs and are not needed by `predict.py`.
