# Batting detectors, November 19 round 2

The second continuation run started from the previous [November 19 checkpoints](../2026-11-19/README.md). The two final-epoch models are available in PyTorch and ONNX. The standalone runner currently selects body `2026-11-19`, bat `2026-11-19`.

Use both round 2 models on a video with the repository's [quick start](../../../README.md#quick-start) and `--model-release 2026-11-19-r2`. Use `--model-release 2026-11-19` to run the preceding pair on the same input. The [training guide](../../../training/README.md) covers dataset auditing, fitting, scoring and export.

## Training data and recipe

The latest labeler export had 32 completed assignments and 4,107 submitted frames. 4,083 body frames and 4,050 bat frames qualified across 31 swings. Actual training used 2,641 body frames and 2,608 bat frames across 19 training swings. Validation retained 241 frames from 2 swings.

Both models ran 10 epochs with a fresh AdamW optimizer, learning rate 2.5e-05, seed 42, and 10 frozen layers. Body input resolution was 640 px and bat resolution was 768 px. Training applied horizontal reflection with probability 0.5. Reflections swap anatomical left and right body slots and preserve the physical bat tip and knob.

## Fixed evaluation

The same 601 evaluation frames from 5 swings and 3 sessions were scored for both releases. A visible body point must fall within 5% of the reference body scale. A bat pair passes when both endpoints fall within 10% of projected bat length. Missing detections count as failures.

| Agreement with reviewed labels | November 19 | Round 2 | Change |
| --- | ---: | ---: | ---: |
| Body joints | 88.78% (6,155/6,933) | 89.30% (6,191/6,933) | +0.52 pp |
| Wrists | 87.55% (956/1,092) | 86.63% (946/1,092) | -0.92 pp |
| Bat endpoint pairs | 78.49% (456/581) | 76.25% (443/581) | -2.24 pp |

## Additional newly submitted evaluation swings

The new-swings comparison covers 5 previously unseen swings in 3 held-out capture sessions. The body and bat scoring inputs contain 600 and 600 eligible frames respectively. The scorer excluded the training and validation sessions and linked trial identities.

| Agreement with reviewed labels | November 19 | Round 2 | Change |
| --- | ---: | ---: | ---: |
| Body joints | 87.95% (5,851/6,653) | 88.61% (5,895/6,653) | +0.66 pp |
| Wrists | 60.06% (588/979) | 62.61% (613/979) | +2.55 pp |
| Bat endpoint pairs | 56.53% (316/559) | 53.67% (300/559) | -2.86 pp |

The references include reviewed interpolation and model-assisted labels. Frames within a swing are correlated. Results describe these two samples. Standalone default selection required nondecreasing fixed-set agreement for the task (including wrists for body) and less than one percentage point of regression on each corresponding supplemental metric. The aggregate records are in [evaluation-summary.json](evaluation-summary.json) and [new-swings-summary.json](new-swings-summary.json).

## Integration and verification

Both ONNX graphs accept an upright letterboxed RGB FP32 tensor named `images`. Body uses `[1,3,640,640]` and returns `output0 [1,56,8400]`. Bat uses `[1,3,768,768]` and returns `output0 [1,11,12096]`. Opset is 17. Candidate selection and NMS run in the consumer. Body uses COCO slots 5–16 (left/right shoulders, elbows, wrists, hips, knees, ankles); facial slots are unsupervised. Bat slot 0 is the barrel tip and slot 1 is the physical terminal knob.

Both sanitized PT files preserve every trained tensor and dtype from the final checkpoint. Numerical comparisons covered 15 samples per model, three per swing across five swings. Source-pixel selected-point differences were at most 0.0002747 px for body and 0.0001526 px for bat. [Validation and checkpoint identities](validation.json) · [Distributed file checksums](SHA256SUMS.txt) · [Extraction test](clean-extraction-test.json)

Upstream license terms are available in the [retained Ultralytics license](../../../licenses/ULTRALYTICS-AGPL-3.0.txt).
