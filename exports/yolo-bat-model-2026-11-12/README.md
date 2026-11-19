# YOLO bat tip + knob model

Packaged November 12, 2026. This is the selected **YOLO11s-pose bat checkpoint**, fine-tuned to locate two physical bat endpoints in an image. It outputs **tip first, knob second**. It does not estimate the hitter's body or reconstruct 3D.

## Contents

| File | Use |
|---|---|
| `models/bat-tip-knob.onnx` | Portable FP32 model; used by the included Python example. |
| `models/bat-tip-knob.pt` | Original Ultralytics checkpoint for Python/PyTorch or further training. |
| `models/BatTipKnobFP32.mlpackage/` | FP32 Core ML package for Apple integration. Keep this directory intact. |
| `predict.py` | Single-image ONNX inference, JSON coordinates and optional overlay. |
| `model-card.json` | Checkpoint identity, input/output contract, provenance and evaluation summary. |
| `validation.json` | Packaging checks and numerical export comparison. |
| `SHA256SUMS.txt`, `verify_checksums.py` | File integrity verification. |
| `THIRD_PARTY_LICENSE.txt` | Upstream Ultralytics AGPL-3.0 license text. |

Training images, player records, annotation exports and credentials are not included. Existing checkpoint/export metadata can retain the original local training-path string. Model files are copied byte-for-byte.

## Quick start: ONNX / CPU

Tested with Python 3.12 on macOS Apple silicon. Other operating systems and acceleration providers were not tested for this handoff. Unzip, open a terminal in this folder, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_checksums.py
python predict.py /path/to/swing-frame.jpg --output prediction.json --annotated overlay.png
```

On Windows, activate with `.venv\Scripts\activate` instead. Dependencies require an initial installation; inference then uses the bundled local model without a service or account.

The JSON contains `accepted`, `points`, and a diagnostic `candidate`. Accepted `points.tip` and `points.knob` are `[x, y]` pixels in the original image after EXIF orientation is applied, with the origin at top-left, X right and Y down. If either endpoint is unusable, `accepted` is false and `points` is null; the candidate is retained for debugging. Do not treat a rejected candidate as a measured position. An optional overlay is drawn only for an accepted pair.

Defaults require a bat score of at least **0.25**, each endpoint score at least **0.25**, and both points inside the source image. These are operating thresholds, not calibrated accuracy probabilities. Flags `--confidence` and `--keypoint-confidence` change them. The example selects the highest-scoring bat candidate; it does not implement hitter/wrist association, temporal tracking, smoothing or gap filling. For video integration, decode frames in timestamp order and retain their actual presentation timestamps alongside each result.

## Input/output contract

**Preprocessing:** Convert to upright RGB. Scale proportionally to fit 768 × 768, using bilinear resize; center with gray `(114,114,114)` padding. Do not stretch the image or use rectangular/minimal padding. For ONNX, divide by 255 and use float32 NCHW `[1,3,768,768]`, input name `images`.

**ONNX output:** `output0`, float32 `[1,11,12096]`, batch/channel/candidate order. ONNX opset 17; fixed batch and resolution; no NMS in the graph.

| Channels | Meaning |
|---|---|
| 0–3 | Bounding-box center X, center Y, width, height in padded-image pixels |
| 4 | Bat class score |
| 5, 6, 7 | Tip X, tip Y, tip score |
| 8, 9, 10 | Knob X, knob Y, knob score |

There is one class, `0 = bat`. Scores are already sigmoid probabilities: **do not apply sigmoid again** and do not assume a separate objectness channel. Reverse the image transform with `x = (model_x - left_padding) / gain` and `y = (model_y - top_padding) / gain`. The example implements the exact rounding used for resize and padding. Never swap tip and knob when mirroring an image; they are physical identities, not left/right joints.

## PyTorch use

The original checkpoint was packaged and checked using **Ultralytics 8.4.208, PyTorch 2.15.0** in a separate Python 3.13 environment. These packages are optional and are not in the ONNX requirements.

```python
from ultralytics import YOLO

model = YOLO("models/bat-tip-knob.pt")
result = model.predict(
    "swing-frame.jpg", imgsz=768, rect=False,
    conf=0.25, max_det=1, device="cpu", verbose=False,
)[0]
if len(result.boxes):
    # Original-image coordinates. Order: tip, knob.
    xy = result.keypoints.xy[0].cpu().numpy()
    scores = result.keypoints.conf[0].cpu().numpy()
    # Apply per-endpoint >=0.25 and in-image checks before accepting the pair.
```

Ultralytics handles coordinate unpadding in `result.keypoints.xy`. Preserve the `rect=False` setting for comparison with the exported square-input models. Only load checkpoints from a trusted source.

## Core ML use

Add `models/BatTipKnobFP32.mlpackage` to Xcode. Its input is `image`, a **768 × 768 RGB image with byte-range values**. Apply the same centered letterbox first; normalization by 1/255 is already inside the model. The output is `var_1347`, float32 `[1,11,12096]`, with the same channel layout and decoding as ONNX. There is no NMS or hitter selection in the package.

The included package uses full precision because the earlier half-precision export changed some selected candidates. Its existing 36-frame CPU comparison against PyTorch had no top-candidate mismatches and a maximum endpoint discrepancy of about **0.000171 original-image pixels**. This is numerical export agreement, not ground-truth detection accuracy or an iPhone latency measurement. Measure runtime and accuracy on the intended device.

## Training and accuracy context

Checkpoint: `bat-20261109T164108Z`. Initialized from `yolo11s-pose.pt`, fine-tuned on **129 labeled frames from six videos**, including 20 complete manually labeled drafts with explicit approval. Only complete visible endpoint pairs were used; absent-bat examples were not part of this fit. The final fit includes the earlier validation video, so its old pre-refit validation score is not an independent final-model result.

On the November 11 development comparison using **207 complete labeled pairs from two other-session videos**, this checkpoint localized both endpoints within 10% of the reference projected bat length on **150/207 (72.46%)** frames, and both within 25 original-image pixels on **166/207 (80.19%)** frames. Missing predictions count as failures. Both newly trained challengers scored worse, so this checkpoint was retained.

Those reference labels include interpolation and model assistance; they are provisional development evidence. This is **not a 72% production-accuracy guarantee**. Motion blur, occlusion, dark/thin bats and other people/bats can cause mistakes, swaps and missing detections. False-positive performance on absent-bat footage and downstream 3D accuracy were not established by this evaluation. The standalone example uses top-score selection; the phone's wrist association is a separate integration policy.

## Provenance and references

The source checkpoint and both export formats are identified by SHA-256 in `model-card.json` and `SHA256SUMS.txt`. The ONNX export metadata records Ultralytics 8.3.162, opset 17, FP32, no graph simplification, and no NMS. The Core ML export used Ultralytics 8.4.208 and coremltools 9.0.

Upstream model metadata identifies AGPL-3.0; its license text is included. This handoff does not grant additional rights to the underlying software, model or training data. See the upstream [licensing information](https://www.ultralytics.com/license), [pose documentation](https://docs.ultralytics.com/tasks/pose/) and [ONNX integration documentation](https://docs.ultralytics.com/integrations/onnx/).
