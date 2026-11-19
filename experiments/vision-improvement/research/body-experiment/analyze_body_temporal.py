"""Compare body error changes on contiguous original frames; no inference.

Residual change subtracts reference motion before measuring displacement. It is
conditional on available points, and interpolated references can bias it.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics

from score_body import JOINTS, finite, key, percentile, xy


def presentation_time(row):
    frame = row.get('frame', {})
    try:
        if (frame['index'] != row['frameIndex'] or not isinstance(frame['pts'], int)
                or isinstance(frame['pts'], bool)):
            return None
        base = Fraction(frame['timeBase'])
        if base <= 0:
            return None
        value = frame['pts'] * base
        if not finite(frame['time']) or abs(float(value)-frame['time']) > 2e-6:
            return None
        return value
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None


def usable_points(pred, row, threshold):
    if not pred.get('detected'):
        return [None]*12
    points, confidences = pred.get('prediction'), pred.get('keypointConfidences')
    if not isinstance(points, list) or not isinstance(confidences, list) or len(points) != 12 or len(confidences) != 12:
        raise ValueError('Detected predictions require exactly twelve ordered joints')
    return [point if xy(point) and finite(confidence) and confidence >= threshold
            and 0 <= point[0] <= row['width'] and 0 <= point[1] <= row['height'] else None
            for point, confidence in zip(points, confidences)]


def distribution(values):
    return {'count': len(values), 'mean': statistics.mean(values) if values else None,
            'median': percentile(values, .5), 'p95': percentile(values, .95),
            'maximum': max(values) if values else None}


def pair_summary(pairs, name):
    measured = [p for p in pairs if p['models'][name] is not None]
    return {'eligibleJointPairs': len(pairs), 'measuredJointPairs': len(measured),
            'missingJointPairs': len(pairs)-len(measured),
            'measuredPairsWithInterpolatedReference': sum(p['interpolatedReference'] for p in measured),
            'residualDeltaPixels': distribution([p['models'][name]['deltaPixels'] for p in measured]),
            'pixelsPerPresentationSecond': distribution([p['models'][name]['pixelsPerPresentationSecond'] for p in measured])}


def analyze(rows, model_predictions, threshold=.3):
    if not model_predictions or not finite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Require models and a finite confidence threshold in [0,1]')
    truth_keys = set(map(key, rows))
    if len(truth_keys) != len(rows):
        raise ValueError('Duplicate reference frame identities')
    lookup, nodes, missingness = {}, defaultdict(list), {}
    for name, predictions in model_predictions.items():
        pred_keys = set(map(key, predictions))
        if len(pred_keys) != len(predictions) or pred_keys-truth_keys:
            raise ValueError(f'Duplicate or unmatched prediction identity: {name}')
        lookup[name] = {key(p): p for p in predictions}
        missingness[name] = {'frames': len(rows), 'missingPredictionRecords': 0,
                             'detectedFrames': 0, 'visibleReferenceJoints': 0,
                             'usableJoints': 0, 'missingJoints': 0,
                             'visibleReferenceWrists': 0, 'usableWrists': 0, 'missingWrists': 0}
    for row in rows:
        if row.get('status') != 'reviewed' or len(row.get('points', [])) != 12:
            raise ValueError('Expected reviewed twelve-point reference frames')
        truth = []
        for point in row['points']:
            if point.get('visibility') not in (0,1,2):
                raise ValueError('Invalid reference visibility')
            if point['visibility'] and not all(finite(point.get(a)) and 0 <= point[a] <= 1 for a in ('x','y')):
                raise ValueError('Invalid reference coordinates')
            truth.append([point['x']*row['width'],point['y']*row['height']] if point['visibility']==2 else None)
        predictions = {}
        for name, items in lookup.items():
            pred = items.get(key(row), {})
            points = usable_points(pred, row, threshold)
            predictions[name] = points
            counts = missingness[name]
            counts['missingPredictionRecords'] += key(row) not in items
            counts['detectedFrames'] += bool(pred.get('detected'))
            for j, reference in enumerate(truth):
                if reference is None:
                    continue
                counts['visibleReferenceJoints'] += 1
                counts['usableJoints' if points[j] is not None else 'missingJoints'] += 1
                if j in (4,5):
                    counts['visibleReferenceWrists'] += 1
                    counts['usableWrists' if points[j] is not None else 'missingWrists'] += 1
        nodes[row['videoSHA256']].append({'row':row,'truth':truth,'predictions':predictions,'time':presentation_time(row)})
    pairs, breaks = [], []
    for video, values in sorted(nodes.items()):
        values.sort(key=lambda n:n['row']['frameIndex'])
        for previous, current in zip(values, values[1:]):
            a,b = previous['row']['frameIndex'],current['row']['frameIndex']
            reason = ('nonconsecutive_source_frame' if b!=a+1 else
                      'invalid_presentation_timestamp' if previous['time'] is None or current['time'] is None else
                      'nonpositive_presentation_delta' if current['time']<=previous['time'] else None)
            if reason:
                breaks.append({'videoSHA256':video,'previousFrame':a,'frameIndex':b,'reason':reason})
                continue
            dt = float(current['time']-previous['time'])
            for j in range(12):
                if previous['truth'][j] is None or current['truth'][j] is None:
                    continue
                reference_sources = [n['row']['points'][j].get('source','unspecified') for n in (previous,current)]
                item = {'videoSHA256':video,'previousFrame':a,'frameIndex':b,'jointIndex':j,
                        'joint':JOINTS[j],'deltaPresentationSeconds':dt,
                        'referenceSources':reference_sources,'interpolatedReference':'interpolated' in reference_sources,'models':{}}
                for name in model_predictions:
                    if any(n['predictions'][name][j] is None for n in (previous,current)):
                        item['models'][name] = None
                        continue
                    residuals = [[n['predictions'][name][j][d]-n['truth'][j][d] for d in (0,1)] for n in (previous,current)]
                    delta = math.dist(*residuals)
                    item['models'][name] = {'deltaPixels':delta,'pixelsPerPresentationSecond':delta/dt}
                pairs.append(item)
    common = [p for p in pairs if all(value is not None for value in p['models'].values())]
    support_groups = {'allModels':list(model_predictions)}
    if 'generic' in model_predictions and 'candidate' in model_predictions:
        support_groups['yoloPair'] = ['generic','candidate']
    common_support = {}
    for group,names in support_groups.items():
        subset = [p for p in pairs if all(p['models'][name] is not None for name in names)]
        common_support[group] = {'models':names,'jointPairs':len(subset),
                                 'wristPairs':sum(p['jointIndex'] in (4,5) for p in subset),
                                 'scores':{name:{'allJoints':pair_summary(subset,name),
                                                 'wrists':pair_summary([p for p in subset if p['jointIndex'] in (4,5)],name),
                                                 'byJoint':{joint:pair_summary([p for p in subset if p['joint']==joint],name) for joint in JOINTS}}
                                           for name in names}}
    return {'schema':'body-temporal-quality-v1','keypointConfidenceThreshold':threshold,
            'frames':len(rows),'eligibleJointPairs':len(pairs),'commonObservedJointPairs':len(common),
            'invalidTimingFrames':sum(n['time'] is None for group in nodes.values() for n in group),
            'missingness':missingness,'breaks':breaks,
            'perModelAvailableSupport':{name:{'allJoints':pair_summary(pairs,name),
                                             'wrists':pair_summary([p for p in pairs if p['jointIndex'] in (4,5)],name)} for name in model_predictions},
            'commonSupport':common_support,'jointPairs':pairs,
            'notes':['Only contiguous original indices with valid exact rational PTS and positive presentation time differences are compared.',
                     'Residual change is change in prediction-minus-reference error: correct true motion cancels.',
                     'Compare models on common support; a model with many missing joints can look artificially smooth on its remaining pairs.',
                     'Interpolated/model-derived references can bias residual changes. These are diagnostics, not standalone model selection criteria.',
                     'Pixels per presentation second do not measure physical bat/body speed or verified sensor time.',
                     'No learned thresholds, interpolation, smoothing, person reassignment or inference is performed.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path(__file__).resolve().parent/'session-holdout-20261111')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    paths={'generic':args.run/'generic-val-predictions.json','candidate':args.run/'candidate-val-predictions.json',
           'macVision':args.run/'mac-vision-cpu/predictions.json'}
    truth=args.run/'val-rows.json'
    result=analyze(json.loads(truth.read_text()),{name:json.loads(path.read_text()) for name,path in paths.items()})
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    result['inputs']={'truthSHA256':sha(truth),'sourceSHA256':sha(Path(__file__)),
                      'predictionsSHA256':{name:sha(path) for name,path in paths.items()}}
    output=args.output or args.run/'body-temporal-comparison.json'
    if output.exists():
        raise ValueError('Refusing to overwrite a completed temporal comparison')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(output),'eligibleJointPairs':result['eligibleJointPairs'],
                      'commonObservedJointPairs':result['commonObservedJointPairs'],'missingness':result['missingness']}))


if __name__=='__main__':
    main()
