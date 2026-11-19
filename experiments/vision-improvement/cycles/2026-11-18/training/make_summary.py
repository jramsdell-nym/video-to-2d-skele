"""Summarize completed, unchanged primary score reports for this cycle."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def body_summary(score):
    keys = ['frames', 'detectedFrames', 'referenceJoints', 'normalizableJoints', 'missingJoints',
            'within0.05BodyScale', 'jointPCK0.05BodyScale', 'jointPCK25Pixels',
            'medianErrorPixelsOnLocalized', 'p95ErrorPixelsOnLocalized',
            'macroVideoJointPCK0.05BodyScale']
    result = {key: score.get(key) for key in keys}
    result['wrists'] = score['wrists']
    for field in ('byVideo', 'byJoint', 'byPointSource', 'byReviewMethod',
                  'frameReviewedSubset', 'frameReviewedNoInterpolationSubset',
                  'noInterpolationFrameSubset', 'hiddenEstimatedTargets'):
        result[field] = score[field]
    return result


def bat_summary(score):
    keys = ['positiveFrames', 'detectedFrames', 'missedFrames', 'visibleTruthEndpoints',
            'missingEndpoints', 'bothVisibleFrames', 'normalizableFrames',
            'bothEndpointsWithin0.1BatLength', 'bothEndpointsPCK0.1BatLength',
            'bothEndpointsWithin25Pixels', 'bothEndpointsPCK25Pixels',
            'medianEndpointErrorPixelsOnLocalizedEndpoints',
            'p95EndpointErrorPixelsOnLocalizedEndpoints',
            'directedAngleWithin10DegreesRate', 'swapCandidates', 'negativeFrames',
            'macroVideoBothEndpointsPCK0.1BatLength', 'byVideo', 'bySession']
    return {key: score.get(key) for key in keys}


def main():
    result = {'body': {}, 'bat': {}, 'training': {}, 'interpretation': [
        'All scores use the same five recording-session-held-out video sources.',
        'Reference labels include interpolation and model assistance; these are agreement scores, not independent gold-standard accuracy.',
        'Body has twelve supervised limb joints; the original seventeen-output head is preserved.',
        'Primary body score uses visible joints and confidence>=0.3; primary BAT uses box/endpoint confidence>=0.25 and both-visible pairs.',
        'Missing predictions remain in primary accuracy denominators.',
        'Fixed final checkpoints; no selection or recipe tuning using these evaluation scores.',
        'No app models were deployed or replaced.'
    ]}
    for task, models, summarize in (('body', ['generic', 'previous', 'candidate'], body_summary),
                                    ('bat', ['current', 'candidate'], bat_summary)):
        result['training'][task] = read(ROOT / task / 'training-status.json')
        result['training'][task]['split'] = read(ROOT / task / 'manifest.json')['splits']
        for model in models:
            result[task][model] = summarize(read(ROOT / task / f'{model}-primary-score.json'))
            result[task][model]['predictionManifest'] = read(ROOT / task / f'{model}-prediction-manifest.json')
    result['changes'] = {
        'bodyCandidateMinusGenericPCKPercentagePoints': 100 * (result['body']['candidate']['jointPCK0.05BodyScale'] - result['body']['generic']['jointPCK0.05BodyScale']),
        'bodyCandidateMinusPreviousPCKPercentagePoints': 100 * (result['body']['candidate']['jointPCK0.05BodyScale'] - result['body']['previous']['jointPCK0.05BodyScale']),
        'batCandidateMinusCurrentPCKPercentagePoints': 100 * (result['bat']['candidate']['bothEndpointsPCK0.1BatLength'] - result['bat']['current']['bothEndpointsPCK0.1BatLength']),
    }
    (ROOT / 'comparison-summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['changes'], indent=2))


if __name__ == '__main__':
    main()
