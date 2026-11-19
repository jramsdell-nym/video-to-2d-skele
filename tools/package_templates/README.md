# Fan Lab body and BAT models — $release

This package contains the $release **2D** body-pose and bat-endpoint candidates, their portable ONNX exports, and a local video inference runner. It is an evaluation handoff. It does not change the iPhone app's selected models or estimate 3D motion, physical bat speed, or biomechanics.

The exact completed training settings and dataset allocation are recorded below and in `evaluation-summary.json`.

$dataset_note

## Results against $baseline

$results_table

These are agreements with reviewed, partly interpolated/model-assisted references on the same frozen evaluation set. Missing predictions count as failures. Body/wrist agreement uses a tolerance of 5% of reference body scale; paired BAT agreement requires **both** endpoints within 10% of the projected reference bat length. The evaluation has been used in previous development, so it is not a new blind production test. Frames from one swing are correlated, and athlete independence is not established.

$allocation_table

Only the training allocation updates weights. `evaluation-summary.json` contains the numeric counts, exact metric names, and training settings. Neither training images nor identifiable evaluation records are included.

The fixed training recipe uses a **50% probability of horizontal reflection**. Reflected body labels swap the anatomical COCO left/right slots; BAT tip and knob keep their physical identities. Validation and evaluation images are not reflected by this augmentation. The inference runner does not automatically mirror or normalize a hitter's handedness.

## Quick start on your own video

Tested using $python_version on $platform_name with CPU. Other operating systems and accelerators require their own check. Unzip the entire folder, then run these commands from inside it:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_checksums.py
python predict.py /path/to/swing.mp4 --output outputs/swing.jsonl --overlay outputs/swing-overlay.mp4
```

On Windows, activate with `.venv\Scripts\activate`. Add `--max-frames 12` for a quick check. CPU is the default; the runner also accepts `--device mps` or `--device 0`, but this handoff does not claim those backends were validated. Use new output paths: the runner refuses to overwrite existing files. Dependency installation needs package access; inference itself uses only the local files and needs no service account. PyTorch `.pt` files must come from a trusted source.

`predict.py` runs both packaged PT models. It produces one JSONL metadata record, one record per decoded frame, and a final `summary` with `complete: true`. Incomplete files must not be treated as completed runs. The optional silent MP4 has the source-relative presentation timestamps, with at most one black padding pixel at the right/bottom for odd dimensions.

## Output and selection

Coordinates use the upright, unmirrored source image: origin top left, X right, Y down. The runner applies right-angle video orientation metadata. It records the original `pts`, rational `timeBase`, source `timeSeconds`, and `relativeTimeSeconds` for every frame; it refuses missing or non-increasing timestamps.

Body output contains the 12 limb joints below. Each point has pixel coordinates, normalized coordinates, confidence and `usable`. The initial body is the largest usable joint box, followed by gated spatial association; a missing or ambiguous body has `points: null`. This heuristic can select a spectator.

BAT output uses the highest-scoring bat independently of the body. `bat.points` is null unless both endpoints pass confidence and image-boundary checks. Rejected raw endpoints remain in `bat.candidate` for diagnosis. No wrist association, smoothing, interpolation or gap filling is applied. Default thresholds are detection 0.25, body joints 0.30 and BAT endpoints 0.25; they are operating thresholds, not calibrated accuracy probabilities. The runner's association policy differs from the evaluation's highest-confidence-person selection, so coverage on an arbitrary video is not the table's agreement metric.

## Model contracts

| Model | Input | Raw ONNX output | Keypoints |
|---|---|---|---|
| Body | float32 `[1,3,$body_size,$body_size]` | float32 `[1,56,$body_candidates]` | 17 COCO slots; use supervised slots 5–16 |
| BAT | float32 `[1,3,$bat_size,$bat_size]` | float32 `[1,11,$bat_candidates]` | 0 = physical barrel tip; 1 = physical terminal knob |

Both ONNX graphs have a fixed batch of 1, fixed square resolution, FP32 operations and **opset 17**, with no NMS in the graph. Names and exact dimensions are recorded in `models/manifest.json`. The native runner performs Ultralytics postprocessing/NMS. Direct ONNX integration must implement its own candidate filtering and selection/NMS; simply treating all raw candidates as detections is incorrect.

Prepare an upright RGB image, scale proportionally by `gain = min(size / width, size / height)`, bilinear-resize to `round(width * gain)` by `round(height * gain)`, then center on a square filled with RGB `(114,114,114)`. Left/top padding uses `round((remaining_pixels / 2) - 0.1)`; right/bottom uses `round((remaining_pixels / 2) + 0.1)`. Convert to contiguous NCHW float32 and divide by 255. Do not stretch or use minimum rectangular padding. In the PT API use `imgsz=size, rect=False` to match this export.

Raw channels 0–3 contain box center X, center Y, width and height in padded-image pixels. Channel 4 is the single class score (there is **no separate objectness channel**). Remaining channels repeat keypoint X, Y and confidence. Scores are already sigmoid outputs: do not apply sigmoid again. Rounding the resized dimensions can make the actual X and Y scales slightly different: use `scale_x = resized_width / original_width` and `scale_y = resized_height / original_height`. Undo the transform with `x = (model_x - left_padding) / scale_x`, `y = (model_y - top_padding) / scale_y`; reject unusable or outside-image points rather than inventing observations.

COCO slot order: nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle. The batting data supervised **only the 12 shoulder-through-ankle slots**; the five face slots are not evaluated as trained outputs and the runner omits them. Anatomical left/right belongs to the hitter, not the viewer. BAT tip and knob are physical identities and must not be exchanged when mirroring an image.

## Checks and limitations

`validation.json` records native-versus-ONNX comparisons on 15 private samples across five evaluation swings, with aggregate errors only. Export parity demonstrates numerical consistency, **not pose accuracy**. A separate clean-extraction smoke test runs the packaged video runner and verifies timestamps, model hashes and a completely decoded overlay; its report is included. Source footage, labels, private review reports, paths and account details are omitted.

Fast motion, blur, occlusion, foreshortening, dark/thin bats, multiple people/bats and unusual camera angles still produce misses or wrong points. The physical knob convention needs consistent human review where gloves hide the handle end; a grip/shaft-entry point is not the same target. False positives on absent-bat footage, calibrated uncertainty, broad handedness performance and downstream 3D accuracy have not been established by this run. Review the overlay before any downstream use.

The `.pt` files are sanitized inference checkpoints reconstructed with exactly matching state tensors. Training optimizer state, original file paths, account metadata and dataset references have been removed. Their file hashes therefore differ from the original training checkpoints; original SHA-256 identities are retained in `package-manifest.json`. They can initialize further training, but do not resume its optimizer state. ONNX metadata is also reduced to the public model contract.

## Files and terms

- `models/body.pt`, `models/bat.pt`: sanitized inference checkpoints.
- `models/body.onnx`, `models/bat.onnx`: portable exports.
- `models/manifest.json`: package-local paths, hashes and input/output contract.
- `predict.py`, `batting/`: local video runner and association implementation.
- `requirements.txt`: exact direct dependencies for the PT runner; `requirements-export.txt`: export/parity tool versions.
- `package-manifest.json`, `evaluation-summary.json`, `validation.json`, `clean-extraction-test.json`: provenance and aggregate validation.
- `SHA256SUMS.txt`, `verify_checksums.py`: integrity checks for every distributed file (except the checksum list itself).
- `LICENSE`, `THIRD_PARTY_NOTICE.txt`: retained upstream terms and attribution.

Upstream YOLO software and model metadata identify Ultralytics AGPL-3.0. Its license text is included as `LICENSE`; this does not assign a new license to original Fan Lab code, grant additional rights, or grant rights to omitted training data. See `THIRD_PARTY_NOTICE.txt` and the upstream license for the applicable terms. The package contains no proprietary service credentials and does not require a repository checkout or the historical experiments to run.
