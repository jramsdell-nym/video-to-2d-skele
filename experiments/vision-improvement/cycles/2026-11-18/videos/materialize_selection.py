"""Decode every frame of a frozen source selection for shared model inference."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import re

import av

from render_comparisons import dump, sha256, verified_file


def materialize(selection_path, output):
    selection_path = selection_path.resolve()
    original = json.loads(selection_path.read_text())
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    materialized = {**original, 'clips': []}
    rows = []
    identities = set()
    for clip in original['clips']:
        slug = clip['slug']
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', slug):
            raise ValueError('Unsafe clip slug')
        source = verified_file(selection_path.parent, clip['source'])
        video_sha = clip['source']['sha256']
        if video_sha in identities:
            raise ValueError('Duplicate selected video')
        identities.add(video_sha)
        directory = output / slug
        directory.mkdir()
        frames = []
        last_time = None
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            size = (stream.width, stream.height)
            if 'width' in clip and size != (clip['width'], clip['height']):
                raise ValueError('Frozen source dimensions do not match')
            rate = str(stream.average_rate)
            for index, frame in enumerate(container.decode(video=0)):
                if frame.pts is None or frame.time_base is None:
                    raise ValueError('Source frame is missing exact presentation time')
                time = Fraction(frame.pts) * frame.time_base
                if last_time is not None and time <= last_time:
                    raise ValueError('Non-increasing source presentation timestamps')
                last_time = time
                if (frame.width, frame.height) != size:
                    raise ValueError('Variable source dimensions are unsupported')
                image_path = directory / f'frame-{index:05d}.png'
                frame.to_image().save(image_path, compress_level=3)
                record = {'index': index, 'pts': frame.pts, 'timeBase': str(frame.time_base), 'time': float(time)}
                frames.append(record)
                rows.append({'videoSHA256': video_sha, 'frameIndex': index,
                             'imagePath': str(image_path), 'imageSHA256': sha256(image_path),
                             'width': frame.width, 'height': frame.height,
                             'sourceFrame': record,
                             **({'clipId': clip['clipId']} if 'clipId' in clip else {})})
        if not frames:
            raise ValueError('Selected video has no frames')
        if 'frameCount' in clip and len(frames) != clip['frameCount']:
            raise ValueError('Decoded frame count differs from selection')
        frame_path = directory / 'source-frames.json'
        dump(frame_path, frames)
        materialized['clips'].append({**clip, 'source': {'path': str(source), 'sha256': video_sha},
                                      'frames': {'path': str(frame_path), 'sha256': sha256(frame_path)},
                                      'width': size[0], 'height': size[1], 'frameCount': len(frames),
                                      'sourceRate': rate})
        print(json.dumps({'clip': slug, 'frames': len(frames), 'sourceRate': rate}), flush=True)
    rows_path = output / 'inference-rows.json'
    dump(rows_path, rows)
    selection_output = output / 'materialized-selection.json'
    dump(selection_output, materialized)
    receipt = {'selection': str(selection_path), 'selectionSHA256': sha256(selection_path),
               'sourceCodeSHA256': sha256(Path(__file__)), 'inferenceRows': str(rows_path),
               'inferenceRowsSHA256': sha256(rows_path), 'frameCount': len(rows),
               'materializedSelection': str(selection_output),
               'materializedSelectionSHA256': sha256(selection_output),
               'allSourceFramesIncluded': True, 'inferencePerformed': False}
    dump(output / 'materialization-receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('selection', type=Path)
    parser.add_argument('output', type=Path)
    arguments = parser.parse_args()
    print(json.dumps(materialize(arguments.selection, arguments.output)), flush=True)
