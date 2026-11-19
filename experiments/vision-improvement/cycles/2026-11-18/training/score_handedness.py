"""Derived catalog-handedness slices; never rewrites primary scores or labels."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hand(row):
    return row.get('handedness') if row.get('handedness') in ('L', 'R') else 'unknown'


def key(row):
    return row['videoSHA256'], row['frameIndex']


def counts(rows):
    return {'frames': len(rows), 'videos': len({row['videoSHA256'] for row in rows}),
            'sessions': len({row.get('session') for row in rows}),
            'knownAthletes': len({row['athleteId'] for row in rows if row.get('athleteId')})}


def main():
    output = {'schema': 'catalog-handedness-slices-v1', 'labels': 'Unmodified catalog L/R; missing and unrecognized values remain unknown.',
              'limits': ['Evaluation has only one labeled left-handed video and two right-handed videos; these slices do not establish broad handedness generalization.',
                         'Frames from a video are correlated, not independent trials.',
                         'Unknown is a missing metadata category, not ambidextrous or an inferred side.',
                         'Random training reflections do not increase the count of independent labeled athletes or swings.'], 'tasks': {}}
    for task, model_names in (('body', ('generic', 'previous', 'candidate')), ('bat', ('current', 'candidate'))):
        run = ROOT / task
        rows = read(run / 'evaluation-rows.json')
        spec = importlib.util.spec_from_file_location(f'{task}_scorer', run / 'scorer.py')
        scorer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scorer)
        info = {'partitions': {}, 'models': {}, 'rowsSHA256': sha(run / 'evaluation-rows.json'), 'scorerSHA256': sha(run / 'scorer.py')}
        for partition in ('train', 'val', 'evaluation'):
            partition_rows = read(run / f'{partition}-rows.json')
            info['partitions'][partition] = {h: counts([row for row in partition_rows if hand(row) == h]) for h in ('L', 'R', 'unknown')}
        for model in model_names:
            path = run / f'{model}-predictions.json'
            manifest = read(run / f'{model}-prediction-manifest.json')
            assert sha(path) == manifest['predictionsSHA256']
            assert info['rowsSHA256'] == manifest['rowsSHA256']
            predictions = {key(p): p for p in read(path)}
            assert set(predictions) == set(map(key, rows))
            model_info = {'predictionsSHA256': manifest['predictionsSHA256'], 'checkpointSHA256': manifest['checkpointSHA256'], 'slices': {}}
            for h in ('L', 'R', 'unknown'):
                truth = [row for row in rows if hand(row) == h]
                predicted = [predictions[key(row)] for row in truth]
                if task == 'body':
                    score = scorer.score(truth, predicted, .3)
                    values = {field: score[field] for field in ('referenceJoints', 'normalizableJoints', 'missingJoints', 'within0.05BodyScale', 'jointPCK0.05BodyScale', 'jointPCK25Pixels', 'byJoint')}
                    values['wrists'] = score['wrists']
                else:
                    gated = []
                    for record in predicted:
                        item = dict(record)
                        item['detected'] = record['detected'] and record.get('score', 0) >= .25
                        if item['detected']:
                            item['prediction'] = [point if confidence >= .25 else None for point, confidence in zip(record['prediction'], record['endpointConfidences'])]
                        gated.append(item)
                    score = scorer.score(truth, gated)
                    values = {field: score[field] for field in ('positiveFrames', 'detectedFrames', 'missingEndpoints', 'bothVisibleFrames', 'normalizableFrames', 'bothEndpointsWithin0.1BatLength', 'bothEndpointsPCK0.1BatLength', 'bothEndpointsWithin25Pixels', 'bothEndpointsPCK25Pixels', 'swapCandidates')}
                model_info['slices'][h] = {**counts(truth), **values}
            primary = read(run / f'{model}-primary-score.json')
            reconcile_fields = ('normalizableJoints', 'within0.05BodyScale') if task == 'body' else ('normalizableFrames', 'bothEndpointsWithin0.1BatLength')
            for field in reconcile_fields:
                assert sum(s[field] for s in model_info['slices'].values()) == primary[field], f'Handedness slice does not reconcile: {model} {field}'
            model_info['reconcilesWithPrimaryScore'] = True
            info['models'][model] = model_info
        output['tasks'][task] = info
    target = ROOT / 'handedness-scores.json'
    assert not target.exists(), 'Refusing to overwrite a derived report'
    target.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(target)


if __name__ == '__main__':
    main()
