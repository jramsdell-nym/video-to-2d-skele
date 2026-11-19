"""Read-only, generation-pinned labeler export. Never writes to GCS.

Stages: fetch current completed documents; audit eligibility; decode exact frames.
No split is selected here. Raw account/submission metadata remains in private/.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent
GCLOUD = os.environ.get('SWING_GCLOUD', 'gcloud')
PREFIX = os.environ.get('SWING_LABELER_PREFIX', 'gs://YOUR_LABELER_BUCKET/app/v1/').rstrip('/') + '/'
JOINTS = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
          'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
          'right_knee', 'left_ankle', 'right_ankle']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value, private=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    if private:
        temp.chmod(0o600)
    temp.replace(path)


def command(*args):
    result = subprocess.run([GCLOUD, 'storage', *args], text=True, capture_output=True)
    if result.returncode:
        # gcloud errors contain no token values for the supported storage calls.
        raise RuntimeError(result.stderr.strip())
    return result.stdout


def fetch_one(uri, target, generation=None):
    """Describe once, then fetch that immutable generation. Preserve provenance."""
    target = Path(target)
    metadata_path = target.with_suffix(target.suffix + '.metadata.json')
    if target.exists() and metadata_path.exists():
        meta = read(metadata_path)
        if meta['uri'] != uri or (generation and meta['generation'] != str(generation)):
            raise ValueError('Cached object identity mismatch')
        if sha(target) != meta['sha256']:
            raise ValueError('Cached object bytes changed')
        return meta
    described = json.loads(command('objects', 'describe', uri + ('#' + str(generation) if generation else ''), '--format=json'))
    gen = str(described['generation'])
    if generation and gen != str(generation):
        raise ValueError('Source generation mismatch')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.partial')
    command('cp', uri + '#' + gen, str(partial))
    partial.chmod(0o600)
    partial.replace(target)
    meta = {'uri': uri, 'generation': gen, 'sha256': sha(target),
            'bytes': target.stat().st_size, 'retrievedAt': datetime.now(timezone.utc).isoformat(),
            'objectMetadata': {k: described[k] for k in ('name', 'size', 'crc32c', 'md5_hash', 'update_time', 'creation_time') if k in described}}
    save(metadata_path, meta, private=True)
    return meta


def fetch():
    private = ROOT / 'private'
    private.mkdir(parents=True, exist_ok=True)
    private.chmod(0o700)
    objects = [fetch_one(PREFIX + name + '.json', private / (name + '.json')) for name in ('queue', 'catalog', 'config')]
    queue = read(private / 'queue.json')
    tasks = [t for t in queue['tasks'].values() if t.get('status') == 'completed']
    if len(tasks) > 1000:
        raise ValueError('Unexpectedly large completed task inventory; review scope first')
    for task in sorted(tasks, key=lambda t: t['id']):
        if not re.fullmatch(r'[a-zA-Z0-9-]{8,80}', task['id']):
            raise ValueError('Invalid assignment ID')
        objects.append(fetch_one(PREFIX + 'assignments/' + task['id'] + '.json', private / 'assignments' / (task['id'] + '.json'), task.get('submissionGeneration')))
    save(ROOT / 'cloud-objects.json', objects)
    print(json.dumps({'completedDocuments': len(tasks), 'frozenObjects': len(objects)}))


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def point_valid(p):
    return isinstance(p, dict) and type(p.get('visibility')) is int and p['visibility'] in (0, 1, 2) and (
        p['visibility'] == 0 or all(finite(p.get(axis)) and 0 <= p[axis] <= 1 for axis in ('x', 'y')))


def identity(r):
    return r['videoSHA256'], r['frameIndex']


def insert_unique(destination, record, kind):
    """Collapse exact targets while keeping every contributing review record."""
    key = identity(record)
    old = destination.get(key)
    if old is None:
        destination[key] = record
        return
    geometry = ['frame', 'width', 'height', 'points', 'box', 'session', 'trial', 'camera', 'source']
    if any(old.get(field) != record.get(field) for field in geometry):
        raise ValueError('Conflicting duplicate ' + kind + ' labels; manual adjudication required')
    old.setdefault('duplicateReviews', []).append({k: record.get(k) for k in
        ('assignmentId', 'annotatorId', 'reviewMethod', 'reviewedAt', 'updatedAt')})


def verified_snapshot(path):
    """Reject edited/corrupt raw snapshots before producing training records."""
    path = Path(path)
    meta = read(path.with_suffix(path.suffix + '.metadata.json'))
    if sha(path) != meta.get('sha256'):
        raise ValueError('Raw snapshot hash mismatch: ' + path.name)
    return read(path), meta


def frame_time(frame, index):
    if not isinstance(frame, dict) or type(frame.get('index')) is not int or frame['index'] != index:
        raise ValueError('Invalid source frame identity')
    if type(frame.get('pts')) is not int or not isinstance(frame.get('timeBase'), str):
        raise ValueError('Invalid presentation timestamp type')
    time_base = Fraction(frame['timeBase'])
    if time_base <= 0 or not finite(frame.get('time')):
        raise ValueError('Invalid presentation time base')
    exact = frame['pts'] * time_base
    if abs(float(exact) - frame['time']) > 2e-6:
        raise ValueError('Inconsistent presentation timestamp')
    return exact


def source_identity(row):
    uri = row.get('uri')
    if row.get('bucket') and row.get('objectName'):
        from_parts = 'gs://' + row['bucket'] + '/' + row['objectName']
        if uri and uri != from_parts:
            raise ValueError('Catalog URI and bucket/objectName disagree')
        uri = from_parts
    return uri, str(row['generation']) if row.get('generation') is not None else None


def validate_assignment(task, doc, meta):
    if doc.get('version') != 1 or doc.get('id') != task['id'] or doc.get('status') != 'completed':
        raise ValueError('Queue/document identity, version or status changed')
    if not task.get('owner') or doc.get('owner') != task['owner']:
        raise ValueError('Queue/document owners disagree')
    if not task.get('finishMutation') or doc.get('finishMutation') != task['finishMutation']:
        raise ValueError('Queue/document completion revisions disagree')
    if task.get('submissionGeneration') and str(task['submissionGeneration']) != meta['generation']:
        raise ValueError('Queue/document submission generations disagree')
    clip = doc.get('clip', {})
    if not re.fullmatch('[0-9a-f]{64}', clip.get('sha256', '')):
        raise ValueError('Assignment has no valid original video hash')
    if any(type(clip.get(k)) is not int or clip[k] <= 0 for k in ('width', 'height')):
        raise ValueError('Assignment has invalid original dimensions')
    frames = clip.get('frames')
    if not isinstance(frames, list) or not frames:
        raise ValueError('Assignment has no original frame timing table')
    times = [frame_time(f, i) for i, f in enumerate(frames)]
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError('Assignment has ambiguous presentation timestamps')
    pose = doc.get('poseSubmission')
    if pose:
        if pose.get('version') != 1 or pose.get('mutationId') != doc['finishMutation']:
            raise ValueError('Pose submission version/revision disagrees with assignment')
        records = pose.get('records')
        if not isinstance(records, list) or not records or len(records) > len(frames):
            raise ValueError('Invalid pose submission record count')
        if any(r.get('status') not in ('draft', 'reviewed') for r in records):
            raise ValueError('Unexpected pose review state')
        indices = [r.get('frameIndex') for r in records]
        if len(indices) != len(set(indices)):
            raise ValueError('Duplicate frame records within one submission')
        annotator = pose.get('annotator', {}).get('id')
        if not annotator or any(r.get('owner') != annotator for r in records):
            raise ValueError('Pose record owner differs from submission annotator')
        if any(r.get('id') != f"{annotator}:{clip['sha256']}:{r.get('frameIndex')}" for r in records):
            raise ValueError('Pose record ID differs from source identity')
        counts = {'reviewed': sum(r['status'] == 'reviewed' for r in records),
                  'drafts': sum(r['status'] != 'reviewed' for r in records),
                  'smoothedApproved': sum(r.get('reviewMethod') == 'smoothed-batch' for r in records)}
        if pose.get('counts') != counts:
            raise ValueError('Pose submission counts disagree with records')
        reviewed, rejected = counts['reviewed'], 0
    else:
        labels = doc.get('labels', {})
        if not isinstance(labels, dict):
            raise ValueError('Invalid legacy label map')
        if any(str(r.get('frameIndex')) != key for key, r in labels.items()):
            raise ValueError('Legacy label key differs from frame index')
        reviewed = sum(r.get('status') in ('reviewed', 'absent') for r in labels.values())
        rejected = sum(r.get('status') == 'skipped' for r in labels.values())
    if task.get('stats', {}).get('frames') != reviewed or task.get('stats', {}).get('rejectedFrames', 0) != rejected:
        raise ValueError('Queue frame counts disagree with submitted records')
    return reviewed


def legacy_inventory(doc):
    labels = list(doc.get('labels', {}).values())
    clip = doc['clip']
    for r in labels:
        index = r['frameIndex']
        frame_time(r['source'], index)
        if index < 0 or index >= len(clip['frames']) or r['source'] != clip['frames'][index]:
            raise ValueError('Legacy frame timestamp differs from source clip')
        if r.get('videoSHA256') != clip['sha256'] or (r.get('imageWidth'), r.get('imageHeight')) != (clip['width'], clip['height']):
            raise ValueError('Legacy label source identity differs from clip')
        for p in r.get('points', {}).values():
            if p.get('state') == 'visible':
                if not all(finite(p.get(axis)) and 0 <= p[axis] <= 1 for axis in ('x', 'y')):
                    raise ValueError('Legacy visible endpoint has invalid coordinates')
                if abs(p['x'] * clip['width'] - p['pixelX']) > 1e-6 or abs(p['y'] * clip['height'] - p['pixelY']) > 1e-6:
                    raise ValueError('Legacy normalized and pixel coordinates disagree')
    reviewed = [r for r in labels if r.get('status') in ('reviewed', 'absent')]
    providers = sorted({r.get('sourceProvenance', {}).get('source', 'unknown') for r in reviewed})
    return {'legacyFrames': len(reviewed), 'legacySourceProviders': providers,
            'legacyVisiblePairs': sum(r.get('status') == 'reviewed' and all(r.get('points', {}).get(k, {}).get('state') == 'visible' for k in ('tip', 'knob')) for r in reviewed),
            'legacyEndpointStates': dict(Counter(p.get('state', 'unknown') for r in reviewed for p in r.get('points', {}).values())),
            'session': reviewed[0].get('session') if reviewed else None,
            'trial': reviewed[0].get('trial') if reviewed else None,
            'camera': reviewed[0].get('camera') if reviewed else None,
            'legacySourceURIs': sorted({r.get('sourceProvenance', {}).get('uri', 'unknown') for r in reviewed})}


def common_record(r, document, catalog_row):
    if r.get('schema') != 1:
        raise ValueError('Unsupported pose record schema')
    if not re.fullmatch('[0-9a-f]{64}', r.get('videoSHA256', '')):
        raise ValueError('Invalid source video hash')
    index, frame = r.get('frameIndex'), r.get('frame', {})
    if type(index) is not int or index < 0:
        raise ValueError('Invalid source frame identity')
    frame_time(frame, index)
    if any(type(r.get(k)) is not int or r[k] <= 0 for k in ('width', 'height')):
        raise ValueError('Invalid source dimensions')
    clip = document.get('clip', {})
    if clip.get('sha256') and clip['sha256'] != r['videoSHA256']:
        raise ValueError('Submission and clip hashes differ')
    if clip.get('frames'):
        if index >= len(clip['frames']) or clip['frames'][index] != frame:
            raise ValueError('Submission and clip timestamps differ')
    if (r['width'], r['height']) != (clip.get('width'), clip.get('height')):
        raise ValueError('Submission and clip dimensions differ')
    source = r.get('source', {})
    if not isinstance(source, dict) or not isinstance(source.get('uri'), str) or not source['uri'].startswith('gs://') or not re.fullmatch('[1-9][0-9]*', str(source.get('generation', ''))):
        raise ValueError('Source URI/generation missing')
    original_row = document.get('row', {})
    original_uri, original_gen = source_identity(original_row)
    if not original_uri or not original_gen or (source['uri'], str(source['generation'])) != (original_uri, original_gen):
        raise ValueError('Record source differs from original assignment source')
    if source.get('source') != original_row.get('source'):
        raise ValueError('Record source provider differs from original assignment')
    for field in ('session', 'trial', 'camera'):
        if original_row.get(field) and r.get(field) != original_row[field]:
            raise ValueError('Record ' + field + ' differs from original assignment')
    keep = ['schema', 'videoSHA256', 'frameIndex', 'frame', 'width', 'height', 'session', 'trial',
            'camera', 'source', 'status', 'updatedAt', 'reviewedAt', 'reviewMethod']
    out = {k: r[k] for k in keep if k in r}
    out['assignmentId'] = document['id']
    out['canonicalGroup'] = r.get('session', '') + '/' + r.get('trial', '') if r.get('session') and r.get('trial') and r['trial'] != 'unknown' else 'video:' + r['videoSHA256']
    out['trialKeys'] = list(original_row.get('trialKeys') or (['trial:' + r['trial']] if r.get('trial') and r['trial'] != 'unknown' else ['video:' + r['videoSHA256']]))
    out['athleteId'] = catalog_row.get('playerId') or catalog_row.get('player_id')
    out['handedness'] = catalog_row.get('handedness') or catalog_row.get('side')
    out['handednessSourceField'] = 'handedness' if catalog_row.get('handedness') else ('side' if catalog_row.get('side') else None)
    out['annotatorId'] = hashlib.sha256(str(document.get('owner', '')).encode()).hexdigest()[:16]
    out['originalPredictionModel'] = (r.get('prediction') or {}).get('model')
    out['originalBatPredictionModel'] = (r.get('batPrediction') or {}).get('model')
    return out


def audit():
    private = ROOT / 'private'
    queue, _ = verified_snapshot(private / 'queue.json')
    catalog, _ = verified_snapshot(private / 'catalog.json')
    if queue.get('version') != 1 or not isinstance(queue.get('tasks'), dict):
        raise ValueError('Unsupported queue schema')
    if any(k != task.get('id') for k, task in queue['tasks'].items()):
        raise ValueError('Queue key differs from assignment ID')
    rows = {r['id']: r for r in catalog['videos']}
    if len(rows) != len(catalog['videos']):
        raise ValueError('Duplicate catalog identities')
    body, bat, excluded, assignments = {}, {}, [], []
    raw_reviewed = 0
    for task in sorted(queue['tasks'].values(), key=lambda t: t['id']):
        if task.get('status') != 'completed':
            continue
        doc, meta = verified_snapshot(private / 'assignments' / (task['id'] + '.json'))
        observed_count = validate_assignment(task, doc, meta)
        row = rows.get(task['id'], doc.get('row', {}))
        summary = {'assignmentId': task['id'], 'queueFrames': task.get('stats', {}).get('frames'),
                   'session': row.get('session'), 'trial': row.get('trial'), 'camera': row.get('camera'),
                   'sourceVideoSHA256': doc['clip']['sha256'], 'clipFrames': len(doc['clip']['frames']),
                   'submittedCount': observed_count, 'assignmentGeneration': meta['generation'],
                   'athleteId': row.get('playerId') or row.get('player_id'),
                   'handedness': row.get('handedness') or row.get('side'),
                   'bodyFrames': 0, 'batFrames': 0, 'legacyFrames': 0}
        summary['poseReviewedFrames'] = sum(r.get('status') == 'reviewed' for r in (doc.get('poseSubmission') or {}).get('records', []))
        summary['poseDraftFrames'] = sum(r.get('status') == 'draft' for r in (doc.get('poseSubmission') or {}).get('records', []))
        assignments.append(summary)
        pose = doc.get('poseSubmission')
        if not pose:
            # Imported legacy documents can legitimately have an empty row and
            # be absent from the modern Strike Mobile catalog. Inspect labels
            # directly for accounting; never turn this into body supervision.
            summary.update(legacy_inventory(doc))
            reason = 'legacy-source-not-eligible' if summary['legacySourceProviders'] != ['Strike Mobile'] else 'legacy-only-submission-needs-separate-schema'
            excluded.append({'assignmentId': task['id'], 'reason': reason, 'legacyFrames': summary['legacyFrames']})
            continue
        if doc.get('row', {}).get('id') != task['id']:
            raise ValueError('Original assignment catalog ID disagrees')
        summary['catalogSourceChangedSinceAssignment'] = source_identity(row) != source_identity(doc['row'])
        review = queue.get('reviews', {}).get('video:' + row.get('id', ''), {})
        group = queue.get('reviews', {}).get('group:' + row.get('groupId', ''), {})
        reason = None
        if row.get('source') != 'Strike Mobile': reason = 'unsupported-source'
        elif row.get('archived') or row.get('availability') == 'missing': reason = 'archived-or-missing'
        elif review.get('status') == 'reject' or group.get('view') == 'reject': reason = 'rejected-video-or-camera-group'
        if reason:
            excluded.append({'assignmentId': task['id'], 'reason': reason})
            continue
        for r in pose['records']:
            tag = {'assignmentId': task['id'], 'frameIndex': r.get('frameIndex')}
            if r.get('status') != 'reviewed' or not r.get('reviewedAt'):
                excluded.append({**tag, 'reason': 'not-reviewed'})
                continue
            raw_reviewed += 1
            try:
                common = common_record(r, doc, row)
            except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
                excluded.append({**tag, 'reason': str(exc)})
                continue
            raw_points = r.get('points') or []
            # Current cloud submissions retain the original COCO 17 slots.
            points = raw_points[5:17] if isinstance(raw_points, list) and len(raw_points) == 17 else []
            box = r.get('box') or []
            box_valid = isinstance(box, list) and len(box) == 4 and all(finite(x) and 0 <= x <= 1 for x in box) and box[2] - box[0] >= .005 and box[3] - box[1] >= .005
            if box_valid:
                box_valid = all(isinstance(p, dict) and (not p.get('visibility') or (point_valid(p) and box[0] <= p['x'] <= box[2] and box[1] <= p['y'] <= box[3])) for p in points)
            if len(points) == 12 and all(point_valid(p) for p in points) and any(p['visibility'] for p in points) and box_valid:
                record = {**common, 'points': points, 'box': box}
                insert_unique(body, record, 'body')
                summary['bodyFrames'] += 1
            else:
                excluded.append({**tag, 'reason': 'invalid-or-empty-body-supervision'})
            tip, knob = r.get('batTip'), r.get('batKnob')
            if all(point_valid(p) and p['visibility'] > 0 for p in (tip, knob)) and knob.get('reviewedAt'):
                targets = {}
                for name, p in [('tip', tip), ('knob', knob)]:
                    targets[name] = {**p, 'state': 'visible' if p['visibility'] == 2 else 'occluded',
                                     'pixelX': p['x'] * r['width'], 'pixelY': p['y'] * r['height']}
                record = {**common, 'points': targets, 'imageWidth': r['width'], 'imageHeight': r['height']}
                insert_unique(bat, record, 'bat')
                summary['batFrames'] += 1
            else:
                excluded.append({**tag, 'reason': 'missing-or-unaccepted-bat-endpoints'})
    body_rows, bat_rows = [body[k] for k in sorted(body)], [bat[k] for k in sorted(bat)]
    save(ROOT / 'body-rows-unresolved.json', body_rows)
    save(ROOT / 'bat-rows-unresolved.json', bat_rows)
    save(ROOT / 'exclusions.json', excluded)
    save(ROOT / 'assignment-inventory.json', assignments)
    report = {'snapshotAt': datetime.now(timezone.utc).isoformat(), 'priorBodyFrames': 360, 'priorBatFrames': 327,
              'currentBodyFrames': len(body_rows), 'currentBatFrames': len(bat_rows),
              'bodyLargerThanPriorSnapshot': len(body_rows) > 360, 'batLargerThanPriorSnapshot': len(bat_rows) > 327,
              'bodyVideos': len({r['videoSHA256'] for r in body_rows}),
              'batVideos': len({r['videoSHA256'] for r in bat_rows}),
              'bodyCanonicalSwings': len({r['canonicalGroup'] for r in body_rows}),
              'batCanonicalSwings': len({r['canonicalGroup'] for r in bat_rows}),
              'bodySessions': len({r.get('session') for r in body_rows}),
              'cameraFrameCounts': dict(Counter(r.get('camera', 'unknown') for r in body_rows)),
              'resolutionFrameCounts': dict(Counter(str(r['width']) + 'x' + str(r['height']) for r in body_rows)),
              'knownAthletes': len({r['athleteId'] for r in body_rows if r.get('athleteId')}),
              'framesMissingAthleteId': sum(not r.get('athleteId') for r in body_rows),
              'rawReviewedPoseFrames': raw_reviewed, 'completedAssignments': len(assignments),
              'submittedPoseReviewedFrames': sum(a['poseReviewedFrames'] for a in assignments),
              'submittedPoseDraftFrames': sum(a['poseDraftFrames'] for a in assignments),
              'duplicateBodyFramesCollapsed': sum(len(r.get('duplicateReviews', [])) for r in body_rows),
              'duplicateBatFramesCollapsed': sum(len(r.get('duplicateReviews', [])) for r in bat_rows),
              'queueClaimedFrames': sum(a.get('queueFrames') or 0 for a in assignments),
              'legacyOnlyFrames': sum(a['legacyFrames'] for a in assignments),
              'reviewMethods': dict(Counter(r.get('reviewMethod', 'unknown') for r in body_rows)),
              'bodyPointSources': dict(Counter(p.get('source', 'unknown') for r in body_rows for p in r['points'])),
              'bodyPointVisibility': dict(Counter(p['visibility'] for r in body_rows for p in r['points'])),
              'batPointSources': dict(Counter(p.get('source', 'unknown') for r in bat_rows for p in r['points'].values())),
              'exclusionReasons': dict(Counter(x['reason'] for x in excluded)),
              'split': 'Not assigned; root must freeze train/evaluation membership before training.',
              'limitations': ['Reviewed/interpolated/model-assisted references are provisional, not independent gold labels.',
                              'Legacy-only submissions are counted separately; the audited 24-frame entry is TrackMan bat-only and excluded by the current source policy.']}
    save(ROOT / 'inventory.json', report)
    print(json.dumps(report, indent=2))


def images():
    import av
    from PIL import Image
    body = read(ROOT / 'body-rows-unresolved.json')
    bat = read(ROOT / 'bat-rows-unresolved.json')
    needed = {identity(r): r for r in body + bat}
    videos = {}
    for r in needed.values():
        videos.setdefault(r['videoSHA256'], []).append(r)
    files = {}
    for video_hash, records in sorted(videos.items()):
        sources = {(r['source']['uri'], str(r['source']['generation'])) for r in records}
        if len(sources) != 1:
            raise ValueError('One video hash has multiple source identities; review first')
        uri, gen = next(iter(sources))
        video = ROOT / 'sources' / (video_hash + '.mp4')
        fetch_one(uri, video, gen)
        if sha(video) != video_hash:
            raise ValueError('Source MP4 hash does not match annotations')
        by_index = {r['frameIndex']: r for r in records}
        seen = set()
        with av.open(video) as container:
            if len(container.streams.video) != 1:
                raise ValueError('Multiple video streams require explicit primary-track review')
            stream = container.streams.video[0]
            if float(stream.metadata.get('rotate', 0)) % 360:
                raise ValueError('Rotated source requires review before decoding')
            previous_time = None
            for index, frame in enumerate(container.decode(stream)):
                if frame.pts is None or frame.time_base is None:
                    raise ValueError('Decoded frame has no presentation timestamp')
                current_time = frame.pts * frame.time_base
                if previous_time is not None and current_time <= previous_time:
                    raise ValueError('Decoded frames are not in presentation order')
                previous_time = current_time
                if index not in by_index:
                    continue
                r = by_index[index]
                if (frame.width, frame.height) != (r['width'], r['height']):
                    raise ValueError('Decoded frame dimensions differ')
                if abs(float(frame.pts * frame.time_base) - r['frame']['time']) > 2e-6:
                    raise ValueError('Decoded frame PTS differs')
                if frame.rotation:
                    raise ValueError('Rotated source requires review before decoding')
                path = ROOT / 'images' / (video_hash + '_' + str(index) + '.png')
                path.parent.mkdir(exist_ok=True)
                frame.to_image().save(path)
                with Image.open(path) as im:
                    im.verify()
                files[(video_hash, index)] = {'image': str(path.relative_to(ROOT)), 'imagePath': str(path), 'imageSHA256': sha(path)}
                seen.add(index)
        if seen != set(by_index):
            raise ValueError('Source did not contain every requested frame')
    for name, rows in [('body', body), ('bat', bat)]:
        for r in rows:
            r.update(files[identity(r)])
        save(ROOT / (name + '-rows.json'), rows)
    save(ROOT / 'image-manifest.json', [{'videoSHA256': k[0], 'frameIndex': k[1], **v} for k, v in sorted(files.items())])
    print(json.dumps({'verifiedImages': len(files), 'bodyFrames': len(body), 'batFrames': len(bat)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fetch', 'audit', 'images'])
    args = parser.parse_args()
    {'fetch': fetch, 'audit': audit, 'images': images}[args.stage]()
