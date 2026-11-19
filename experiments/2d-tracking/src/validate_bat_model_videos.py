"""Verify all encoded frames and timestamps in the bat model review exports."""
import json
from fractions import Fraction
from pathlib import Path
import av

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'labeler/models/active.json').read_text())
    output = ROOT / 'outputs/bat-model-review' / manifest['id']
    review = json.loads((output / 'review.json').read_text())
    assert review['modelId'] == manifest['id']
    assert review['weightsSHA256'] == manifest['weightsSHA256']
    assert review['trainingLabels'] == manifest['usableLabels']
    clips = {c['id']: c for c in json.loads((ROOT / 'labeler/clips.json').read_text())['clips']}
    assert len(review['clips']) == 5
    results = []
    for item in review['clips']:
        original = clips[item['clipId']]
        predictions = json.loads((ROOT / item['predictions']).read_text())
        assert predictions['modelId'] == manifest['id']
        rows = predictions['frames']
        assert item['clipId'] not in manifest['clipCounts']
        assert len(rows) == item['sourceFrames'] == len(original['frames'])
        assert all(r['frameIndex'] == i and r['source'] == original['frames'][i] for i, r in enumerate(rows))
        assert sum(r['points'] is not None for r in rows) == item['usablePredictions']
        t0 = Fraction(original['frames'][0]['pts']) * Fraction(original['frames'][0]['timeBase'])
        for name, factor in [('realtime', 1), ('quarter', 4)]:
            path = ROOT / item['videos'][name]
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                assert stream.codec_context.name == 'h264'
                assert (stream.width, stream.height) == (1920, 1200)
                frames = list(container.decode(video=0))
                assert len(frames) == len(rows)
                max_error = 0
                for frame, source in zip(frames, original['frames']):
                    expected = (Fraction(source['pts']) * Fraction(source['timeBase']) - t0) * factor
                    error = abs(float(Fraction(frame.pts) * frame.time_base - expected))
                    max_error = max(max_error, error)
                    assert error < .001, (path, frame.pts, expected, error)
                duration = float(stream.duration * stream.time_base)
                expected_duration = (float(Fraction(original['frames'][-1]['pts']) * Fraction(original['frames'][-1]['timeBase']) - t0) + 1 / float(Fraction(item['sourceRate']))) * factor
                assert abs(duration - expected_duration) < .003, (path, duration, expected_duration)
                result = {'clip': item['slug'], 'version': name, 'frames': len(frames),
                          'durationSeconds': duration, 'maxTimestampErrorSeconds': max_error,
                          'fileBytes': path.stat().st_size, 'passed': True}
                results.append(result)
    validation = {'passed': True, 'modelId': manifest['id'], 'selectedVideos': len(review['clips']),
                  'sourceFrames': sum(c['sourceFrames'] for c in review['clips']), 'exports': results}
    (output / 'validation.json').write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
