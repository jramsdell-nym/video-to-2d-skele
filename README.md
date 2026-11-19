# Batting video → 2D skeleton

Local **baseball batting** video processing with YOLO body-pose and bat-tip/knob models. This repository contains the current models, earlier YOLO checkpoints, export formats, and the code used to train, evaluate, label, and review them.

Start with `predict.py`: it processes your own video into timestamped 2D body joints and bat endpoints, with an optional overlay MP4. No cloud account is needed for inference.

This is the usable 2D foundation for a future batting performance-analysis version. It does not currently estimate physical bat speed, reconstruct 3D motion, or provide validated performance metrics.

## Quick start

Install Python 3.13 and [Git LFS](https://git-lfs.com/), then:

```sh
git lfs install
git clone https://github.com/jramsdell-nym/video-to-2d-skele.git
cd video-to-2d-skele
git lfs pull
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_models.py
python predict.py /path/to/batting.mp4 \
  --output outputs/batting.jsonl \
  --overlay outputs/batting-overlay.mp4
```

On Windows, activate with `.venv\Scripts\activate`. CPU is the default. Apple silicon can use `--device mps`; a compatible CUDA installation can use `--device 0`. The initial validation was on macOS with CPU; other devices/platforms need their own validation.

For a short check, add `--max-frames 12`. Outputs must use new paths, so previous results and source files cannot be overwritten. The overlay is silent and applies the source video's right-angle orientation metadata. No mirroring is applied. Its presentation timestamps match the input relative to the first frame; odd image dimensions receive at most one black pixel of padding on the right/bottom.

All historical models are included through Git LFS, totaling approximately 1 GB of unique model/package content. To fetch only the four current inference model files after a clone made with LFS download disabled:

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/jramsdell-nym/video-to-2d-skele.git
cd video-to-2d-skele
git lfs pull --include='models/releases/2026-11-19/body.pt,models/releases/2026-11-19/body.onnx,models/releases/2026-11-19/bat.pt,models/releases/2026-11-19/bat.onnx'
python3 verify_models.py --defaults-only
```

## Current models

The latest published candidates are the [November 19 round 2 body and bat detectors](models/releases/2026-11-19-r2/README.md). The tested standalone defaults are body `2026-11-19`, bat `2026-11-19`. To run both candidates, select the release ID explicitly:

```sh
git lfs pull --include='models/releases/2026-11-19-r2/*.pt,models/releases/2026-11-19-r2/*.onnx'
python predict.py /path/to/batting.mp4 --model-release 2026-11-19-r2 \
  --output outputs/round2.jsonl --overlay outputs/round2.mp4
```

Use `--model-release 2026-11-19` to compare with the [November 19 first run](models/releases/2026-11-19/README.md). Both body checkpoints use a 640 px input and provide the 12 anatomical shoulder-through-ankle landmarks. Both bat checkpoints use a 768 px input and return the physical barrel tip and terminal knob.

The [round 2 report](models/releases/2026-11-19-r2/README.md) compares all body joints, wrists and both bat endpoints on the original five evaluation swings and on newly submitted swings. The [model manifest](models/manifest.json) records every file's size and SHA-256 plus the selected model for each task. See the [training guide](training/README.md) for the commands used to prepare private reviewed data, train the two detectors, evaluate them and package their exports. The native [iPhone app](https://github.com/jramsdell-nym/Fan-Lab-Iphone-App) owns its detector selection separately.

## Output contract

JSONL contains one `metadata` record, one `frame` record for every decoded frame, then a `summary` record. A successful output ends with `summary.complete = true`; a file without it is incomplete. A requested `--max-frames` limit is recorded in metadata and summary.

Each frame includes its zero-based index, original integer `pts`, rational `timeBase`, absolute source `timeSeconds`, and `relativeTimeSeconds` from the first frame. Coordinates are in the upright source image, with origin at the top left, X to the right, and Y down. Points also include normalized coordinates, model confidence, and `usable`.

Body selection starts from the largest visible body and then uses spatial association. Missing or ambiguous people produce `body.points = null`. Individual body points below the threshold or outside the image have `usable = false`; do not interpret their raw coordinates as accepted observations. This heuristic can initialize on a spectator and needs review in crowded scenes.

Bat selection uses the highest-scoring detection independently of body selection. `bat.points` is null unless **both** endpoints pass the confidence and image-boundary checks; rejected raw endpoints remain under `bat.candidate`. There is no wrist association in this entry point, so another visible bat can be selected. There is no smoothing, extrapolation, or gap filling.

Defaults: detection confidence 0.25, body keypoint confidence 0.30, bat endpoint confidence 0.25. These are operating thresholds, not calibrated probabilities of accuracy. Inspect the overlay before downstream analysis; motion blur, occlusion, multiple people, and thin/dark bats remain difficult.

## Source map

| Path | Purpose |
| --- | --- |
| `training/`, `tools/` | Continuation training, evaluation and verified export |
| `batting/`, `predict.py` | Standalone local video inference and timestamped export |
| `experiments/2d-tracking/src/` | Original YOLO, Apple Vision, MediaPipe comparison, association, rendering, and validation code |
| `experiments/2d-tracking/labeler/` | Local labeling UI, suggestions, and bat training |
| `experiments/vision-improvement/` | Data checks, scoring, training cycles, body/bat evaluation, and export code |
| `experiments/bat-ios-export/` | Core ML conversion and numerical comparison |
| `experiments/bat-failure-audit/` | Bat failure analysis code |
| `swing-review/` | Local annotation review application and tests |
| `models/manifest.json` | Complete model inventory and default model contract |
| `SOURCE_INDEX.json` | Original imported source paths and hashes |

The historical experiment scripts retain their original directory structure. They require their own datasets, manifests, and sometimes a separate application checkout or macOS Apple Vision. They are research/source archives, not all standalone commands. See [the archive notes](docs/archive.md) for setup boundaries. `predict.py` is the supported path for using the bundled current models on a new video.

Private recordings, player annotations, cloud snapshots, databases, reference tensors derived from images, generated reports, credentials, virtual environments, and caches are excluded. The separate 3D lifter and KinaTrax audit projects are outside this 2D repository.

## Development

```sh
python -m unittest discover -s tests -v
python verify_models.py
```

The [validation record](docs/validation.md) describes the import checks and runtime tests. Native/ONNX numerical agreement checks do not establish pose accuracy. Prior evaluations used interpolated and model-assisted references and should not be presented as independent production accuracy.

## Future performance version

Keep the timestamped 2D output as the boundary for a future performance application. Planned work includes reviewed hitter/bat association, calibrated motion metrics with an explicit camera model, confidence/quality reporting, broader held-out data, and measured device/runtime performance. These are future directions, not delivered features or release commitments.

## Third-party terms

The bundled YOLO checkpoints and exports identify Ultralytics AGPL-3.0 in their upstream metadata. The supplied license text is retained in [licenses/ULTRALYTICS-AGPL-3.0.txt](licenses/ULTRALYTICS-AGPL-3.0.txt) and the original bat export package. No new permissive license is assigned to the repository's original code by this import, and no rights to the omitted training data are granted. Review the upstream terms for your intended distribution or integration.
