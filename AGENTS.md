# Repository scope

This is the canonical repository for Fan Lab batting video-to-2D processing, YOLO models, training/evaluation, and local review tools.

Use this checkout for new work. The old FanIphone experiment folders are historical source material; do not continue development there.

The companion native app is https://github.com/jramsdell-nym/Fan-Lab-Iphone-App (local sibling `../Fan-Lab-Iphone-App`). Keep native app changes there.

Keep private recordings, labels, annotation databases, cloud snapshots, credentials, and generated outputs out of Git. Track model binaries with Git LFS. Update the model manifest and verify hashes when changing model files. Keep actual frame timestamps and explicit missing observations in inference outputs.

Run `python -m unittest discover -s tests -v` for runner changes and `python verify_models.py` for model packaging changes. Do not claim physical biomechanics or 3D accuracy from 2D coverage or numerical export parity.
