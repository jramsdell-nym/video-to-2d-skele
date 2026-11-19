# November 19, 2026 — default batting 2D models

This release is the default body-pose and bat-endpoint pair used by the repository's `predict.py`. Run it from the repository root using the [quick start](../../../README.md#quick-start). The native iPhone app has a separate model integration and is not changed by this release.

## Models and provenance

| Model | Files | Input | Raw ONNX output |
| --- | --- | --- | --- |
| Body | `body.pt`, `body.onnx` | float32 `[1,3,640,640]` | float32 `[1,56,8400]` |
| Bat | `bat.pt`, `bat.onnx` | float32 `[1,3,768,768]` | float32 `[1,11,12096]` |

Both models completed 20 continuation epochs from the November 18 weights with a fresh optimizer, learning rate 0.00005, seed 42, and 10 frozen layers. The body allocation contained 2,161 training frames across 15 swings; the bat allocation contained 2,128 across 15 swings. Each used 241 validation frames across two swings. Aggregate training and evaluation counts are in [evaluation-summary.json](evaluation-summary.json).

The `.pt` files are the sanitized inference checkpoints from the completed training handoff. The packaging check verified all 541 state tensors per model against the original training checkpoint and measured zero raw-output difference on its 15 reference samples. Private paths, dataset references, account metadata, and optimizer state were removed; these files can initialize training but cannot resume the original optimizer. Original and sanitized checkpoint SHA-256 values are preserved in [validation.json](validation.json). The four distributed files are copied byte-for-byte from that handoff and identified in [SHA256SUMS.txt](SHA256SUMS.txt) and the repository [model manifest](../../manifest.json).

## Evaluation against November 18

| Reference agreement | November 18 | November 19 |
| --- | --- | --- |
| Body joints | 88.61% (6,143 / 6,933) | 88.78% (6,155 / 6,933) |
| Wrists | 88.46% (966 / 1,092) | 87.55% (956 / 1,092) |
| Both bat endpoints | 75.04% (436 / 581) | 78.49% (456 / 581) |

The evaluation uses 601 frames from five swings across three sessions, with partly interpolated/model-assisted references. Missing predictions count as failures. Body and wrist agreement requires error within 5% of reference body scale; bat-pair agreement requires both endpoints within 10% of the projected reference bat length. This evaluation has been reused during development, frames within a swing are correlated, and athlete independence is not established. These numbers are not blind production accuracy or biomechanical validation. Wrist agreement regressed; retain that limitation when choosing this release.

The evaluation selects the highest-confidence person; the video runner uses spatial association after choosing an initial body. Runner coverage is therefore a separate measure from the agreement table.

## Integration contract

Body output has the 17 COCO slots, but only the 12 shoulder-through-ankle slots were supervised by the batting labels and are emitted by the runner. The five face slots must not be treated as validated batting outputs. Bat keypoint 0 is the physical barrel tip; keypoint 1 is the physical terminal knob. Their identities do not swap with handedness or image reflection.

The ONNX graphs use FP32, opset 17, fixed batch/resolution, input `images`, and output `output0`. NMS is not included. Letterbox upright RGB to the input square with fill 114, convert to NCHW float32, and divide by 255. Use `rect=False` for matching PT preprocessing. Raw channels 0–3 are box center X/Y and width/height in padded-image pixels, channel 4 is the single class score, and remaining channels repeat keypoint X/Y/confidence. Scores are already sigmoid outputs, with no separate objectness channel. A direct ONNX consumer must implement candidate filtering, NMS/selection, and reversal of the actual resize and padding.

Packaging compared native and ONNX results on 15 samples per model from five private swings. There were zero top-candidate mismatches. Maximum selected source-pixel differences were 0.000366211 for body and 0.000076294 for bat. Full aggregate values and tool versions are in [validation.json](validation.json); this demonstrates numerical export consistency, not pose accuracy. Source footage and labels are excluded.

## November 18 rollback

All earlier checkpoints remain in the repository. To run the previous default with its matching manifest and runner, create a separate checkout from the repository root:

```sh
git worktree add --detach ../video-to-2d-nov18 058336c2917ddfe02d6dc2d99d08eeae18e0714d
cd ../video-to-2d-nov18
git lfs pull --include='experiments/vision-improvement/releases/2026-11-18-labeler/*.pt,experiments/vision-improvement/releases/2026-11-18-labeler/*.onnx'
python3 verify_models.py --defaults-only
```

Follow that checkout's README to install dependencies and run inference.

The upstream Ultralytics terms remain in [the retained AGPL-3.0 license](../../../licenses/ULTRALYTICS-AGPL-3.0.txt). No rights to omitted private training data are granted.
