"""Audit standard evaluation predictions and assemble their immutable render input."""
import json
from pathlib import Path

from render_comparisons import BODY_ORDER, BAT_ORDER, dump, sha256


ROOT = Path(__file__).resolve().parent
SPECS = (
    ('bodyBaseline', 'body', 'generic', 'Generic YOLO11n'),
    ('bodyCandidate', 'body', 'candidate', 'New candidate'),
    ('batBaseline', 'bat', 'current', 'Current BAT model'),
    ('batCandidate', 'bat', 'candidate', 'New candidate'),
)


def read(path):
    return json.loads(Path(path).read_text())


def indexed(rows):
    answer = {(row['videoSHA256'], row['frameIndex']): row for row in rows}
    if len(answer) != len(rows):
        raise ValueError('Duplicate frame identities')
    return answer


def specification(path):
    return {'path': str(path.resolve()), 'sha256': sha256(path)}


def main():
    target = ROOT/'render-input.json'
    if target.exists():
        raise FileExistsError('Render input is already frozen')
    pixel_audit = read(ROOT/'full-pixel-alignment-audit.json')
    if not pixel_audit['everyPixelMatchesIndependentVideoExtraction'] or pixel_audit['framesCompared'] != 601:
        raise ValueError('Full pixel alignment audit is required')
    video_rows_path = ROOT/'materialized/inference-rows.json'
    if sha256(video_rows_path) != pixel_audit['visualRowsSHA256']:
        raise ValueError('Video input rows changed since pixel audit')
    video_rows = indexed(read(video_rows_path))
    config = read(ROOT/'materialized/materialized-selection.json')
    config['models'] = {}
    audit = {'fullPixelAudit': specification(ROOT/'full-pixel-alignment-audit.json'),
             'materializedRows': specification(video_rows_path), 'models': {}}
    for key, task, name, label in SPECS:
        run = ROOT.parent/'training'/task
        prediction_path = run/f'{name}-predictions.json'
        manifest_path = run/f'{name}-prediction-manifest.json'
        manifest = read(manifest_path)
        weights = Path(manifest['checkpoint'])
        if sha256(weights) != manifest['checkpointSHA256']:
            raise ValueError(f'Checkpoint changed: {key}')
        if sha256(prediction_path) != manifest['predictionsSHA256']:
            raise ValueError(f'Prediction file changed: {key}')
        model_rows_path = Path(manifest['rowsPath'])
        if sha256(model_rows_path) != manifest['rowsSHA256']:
            raise ValueError(f'Model input rows changed: {key}')
        reference_path = ROOT.parent/'dataset/splits'/f'{task}-evaluation-rows.json'
        if sha256(reference_path) != pixel_audit['evaluationRowsSHA256'][task]:
            raise ValueError('Reference inputs changed since full pixel audit')
        references, model_rows = indexed(read(reference_path)), indexed(read(model_rows_path))
        predictions = indexed(read(prediction_path))
        if not video_rows.keys() == references.keys() == model_rows.keys() == predictions.keys():
            raise ValueError(f'Frame identity mismatch: {key}')
        if manifest['frames'] != len(video_rows):
            raise ValueError(f'Prediction manifest frame count mismatch: {key}')
        visible = 0
        for identity, prediction in predictions.items():
            row, reference = model_rows[identity], references[identity]
            if row['imageSHA256'] != reference['imageSHA256']:
                raise ValueError(f'Inference pixels differ from audited reference image: {key}, {identity}')
            if Path(prediction['imagePath']).resolve() != Path(row['imagePath']).resolve():
                raise ValueError(f'Prediction image differs from manifest row: {key}, {identity}')
            if prediction['detected']:
                count = 12 if task == 'body' else 2
                if len(prediction['prediction']) != count:
                    raise ValueError(f'Wrong keypoint count: {key}')
                if len(prediction['keypointConfidences' if task == 'body' else 'endpointConfidences']) != count:
                    raise ValueError(f'Wrong confidence count: {key}')
                selected = max(prediction['candidates'], key=lambda item: item['score'])
                if prediction['score'] != selected['score'] or prediction['prediction'] != selected['prediction']:
                    raise ValueError(f'Prediction does not use top-confidence model candidate: {key}')
                visible += 1
        config['models'][key] = {'label': label, 'weights': specification(weights),
                                 'predictions': specification(prediction_path),
                                 'predictionManifest': specification(manifest_path),
                                 'keypointOrder': list(BODY_ORDER if task == 'body' else BAT_ORDER),
                                 'keypointThreshold': .3 if task == 'body' else .25,
                                 'detectionThreshold': .25}
        audit['models'][key] = {'frames': len(predictions), 'rawDetectedFrames': visible,
                                'allInputPixelsMatchAuditedEvaluation': True,
                                'allFrameIdentitiesMatchFrozenVideoSelection': True,
                                'topConfidenceSelectionVerified': True,
                                'predictionManifest': specification(manifest_path)}
    dump(target, config)
    audit['renderInput'] = specification(target)
    dump(ROOT/'prediction-alignment-audit.json', audit)
    print(json.dumps({'renderInput': str(target), 'models': len(config['models']), 'framesPerModel': len(video_rows)}), flush=True)


if __name__ == '__main__':
    main()
