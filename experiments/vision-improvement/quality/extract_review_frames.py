"""Decode exact, unaltered source frames for label review (requires PyAV + Pillow)."""
import argparse
import json
from pathlib import Path

import av

from audit_labels import HERE, LEGACY, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeler", type=Path, default=LEGACY)
    parser.add_argument("--output", type=Path, default=HERE / "reports" / "inspection")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--frames", required=True, help="Comma separated original zero-based decode indices")
    args = parser.parse_args()
    clips = {c["id"]: c for c in json.loads((args.labeler / "clips.json").read_text())["clips"]}
    clip = clips[args.clip]
    wanted = set(map(int, args.frames.split(",")))
    if any(i < 0 or i >= len(clip["frames"]) for i in wanted):
        raise ValueError("Requested frame is outside catalog")
    if sha256(Path(clip["path"])) != clip["sha256"]:
        raise ValueError("Source video checksum changed")
    args.output.mkdir(parents=True, exist_ok=True)
    output = []
    with av.open(clip["path"]) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index not in wanted:
                continue
            expected = clip["frames"][index]
            if frame.pts != expected["pts"] or str(frame.time_base) != expected["timeBase"]:
                raise ValueError("Decoded frame PTS differs from catalog")
            path = args.output / f"{clip['id']}-{index}.png"
            frame.to_image().save(path)
            output.append({"clipId": clip["id"], "videoSHA256": clip["sha256"],
                           "frameIndex": index, "source": expected,
                           "image": str(path), "pngSHA256": sha256(path)})
    if len(output) != len(wanted):
        raise ValueError("Decoder did not return every requested frame")
    write_json(args.output / f"{clip['id']}-extraction.json", output)
    print(f"Decoded {len(output)} source frames; no source/label writes.")


if __name__ == "__main__":
    main()
