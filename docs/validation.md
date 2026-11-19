# Validation records

## November 19 default-model promotion

- Promoted the completed November 19 body and bat retrain in the root model manifest. Added the two sanitized PT checkpoints and two ONNX exports through Git LFS; retained all 51 earlier model/package files.
- SHA-256 and byte-size verification passed for all 55 model/package files. All four default paths and hashes match the inventory and the completed training handoff. New release text and checkpoint metadata passed a private-path/account-string audit.
- All 6 standalone-runner contract tests passed. Inference code and dependency versions are unchanged.
- A complete local 120-frame batting clip ran on CPU using the new manifest defaults. Its metadata identifies both November 19 checkpoint paths and hashes. All 120 frames had an observed body and accepted bat endpoints; these are coverage counts, not accuracy results.
- Independently decoded the source and overlay: both contain 120 frames. All JSON source presentation timestamps match exactly, and all overlay timestamps exactly match source timestamps relative to the first frame.
- The private clip and generated JSONL/MP4 remain excluded from Git. The smoke test used the existing pinned inference environment; a fresh dependency installation, CUDA/MPS, and phone integration were not tested in this promotion.

See the [release notes](../models/releases/2026-11-19/README.md) for aggregate reference agreement, the wrist regression, original/sanitized checkpoint provenance, existing 15-sample-per-model ONNX parity checks, and rollback instructions. Numerical export parity is not pose accuracy.

## Initial import — November 19, 2026 (November 18 defaults)

- SHA-256 and byte-size verification passed for all 51 model/package files.
- All imported Python source passed syntax compilation.
- 109 unit tests passed: 6 standalone-runner contracts, 34 quality/scoring tests, 19 body-experiment tests, 17 dataset tests, 13 training-cycle tests, and 20 local-review tests. The training-cycle tests require Pillow and were run with the inference environment.
- The November 18 body and bat checkpoints processed a complete local 120-frame batting clip on CPU. The runner emitted 120 frame records, 120 observed-body frames, and 119 accepted-bat frames. These are coverage counts, not accuracy results.
- The exported MP4 independently decoded to 120 frames. JSON source timestamps agreed exactly with input presentation timestamps, and decoded overlay timestamps agreed exactly after shifting the origin to zero.
- A separate 12-frame run verified the optional frame limit.
- The private validation clip and generated outputs are excluded from Git.

Runtime: Python 3.13.14, Ultralytics 8.4.208, PyTorch 2.15.0, PyAV 19.0.2, OpenCV 5.0.0, NumPy 2.5.4, macOS. The existing environment reports duplicate AVFoundation class warnings when importing PyAV and OpenCV together; the decode, inference, encode, and independent decode checks completed successfully.

The historical ONNX release manifest recorded 15 native/export comparison frames per model. Only aggregate numerical agreement and model hashes are retained here; private source-frame payloads and image tensors are excluded. No new ground-truth accuracy, real-time throughput, CUDA/MPS, or phone-performance acceptance is claimed by this import.
