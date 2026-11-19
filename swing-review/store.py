"""Transactional local drafts, leases, provenance, and explicit review passes."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
import uuid

JOINTS = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
          'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
          'right_knee', 'left_ankle', 'right_ankle', 'batTip', 'batKnob']
PASSES = ['placement', 'replay', 'joints', 'final']


class Conflict(ValueError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def point(record, joint):
    return record.get(joint) if joint in ('batTip', 'batKnob') else record['points'][JOINTS.index(joint) + 5]


def set_point(record, joint, value):
    if joint in ('batTip', 'batKnob'):
        record[joint] = value
    else:
        record['points'][JOINTS.index(joint) + 5] = value


def located(p):
    return isinstance(p, dict) and all(type(p.get(k)) in (int, float) and math.isfinite(p[k]) and 0 <= p[k] <= 1 for k in ('x', 'y'))


def validate_point(p):
    if not isinstance(p, dict) or type(p.get('visibility')) is not int or p['visibility'] not in (0, 1, 2):
        raise ValueError('Visibility must be 0 (unlabelled), 1 (hidden estimate), or 2 (visible)')
    if p['visibility'] and not located(p):
        raise ValueError('Located points require finite normalized x/y within [0, 1]')
    if any(p.get(k) is not None and (type(p[k]) not in (int, float) or not math.isfinite(p[k]) or not 0 <= p[k] <= 1) for k in ('x', 'y')):
        raise ValueError('Coordinates must be null or finite normalized values')


def complete(record):
    return all(located(point(record, joint)) and point(record, joint).get('visibility') in (1, 2) for joint in JOINTS)


def validate_box(record):
    box = record.get('box')
    if not isinstance(box, list) or len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in box):
        raise ValueError('A normalized person box is required before approval')
    if box[2] - box[0] < .005 or box[3] - box[1] < .005:
        raise ValueError('Person box must have nonzero area')
    for joint in JOINTS[:12]:
        p = point(record, joint)
        if p['visibility'] and not (box[0] <= p['x'] <= box[2] and box[1] <= p['y'] <= box[3]):
            raise ValueError('Person box must contain all labelled body joints')


def clear_review(record):
    record['status'] = 'draft'
    for field in ('reviewedAt', 'reviewMethod'):
        record.pop(field, None)
    if record.get('batKnob'):
        record['batKnob'].pop('reviewedAt', None)
    record['updatedAt'] = now()


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
              CREATE TABLE IF NOT EXISTS trials (
                id TEXT PRIMARY KEY, asset TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'draft',
                revision INTEGER NOT NULL DEFAULT 0, settings TEXT NOT NULL,
                lease_owner TEXT, lease_token TEXT, lease_until REAL, created TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS frames (
                trial TEXT NOT NULL REFERENCES trials(id), idx INTEGER NOT NULL,
                baseline TEXT NOT NULL, record TEXT NOT NULL, meta TEXT NOT NULL,
                PRIMARY KEY(trial,idx));
              CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, trial TEXT NOT NULL, revision INTEGER NOT NULL,
                actor TEXT NOT NULL, kind TEXT NOT NULL, created TEXT NOT NULL,
                before TEXT NOT NULL, after TEXT NOT NULL, detail TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS requests (
                trial TEXT NOT NULL, request_id TEXT NOT NULL, digest TEXT NOT NULL,
                response TEXT NOT NULL, PRIMARY KEY(trial,request_id));
              CREATE TABLE IF NOT EXISTS agent_reports (
                id TEXT PRIMARY KEY, trial TEXT NOT NULL REFERENCES trials(id),
                actor TEXT NOT NULL, created TEXT NOT NULL, content TEXT NOT NULL);
            ''')
        self.path.chmod(0o600)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def add(self, asset):
        asset = deepcopy(asset)
        records = asset.pop('records')
        clip, tid = asset['clip'], asset['id']
        owners = {r['owner'] for r in records}
        if len(owners) > 1:
            raise ValueError('Mixed annotation owners require an explicit migration')
        # Historic assignments can retain a legacy record owner after account migration.
        # Preserve the record identity; never rewrite it to the current assignment owner.
        record_owner = next(iter(owners), asset['owner'])
        asset['recordOwner'] = record_owner
        indices = [r['frameIndex'] for r in records]
        if len(indices) != len(set(indices)) or any(type(i) is not int or not 0 <= i < len(clip['frames']) for i in indices):
            raise ValueError('Invalid or duplicate source frames')
        by_index = {r['frameIndex']: r for r in records}
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM trials WHERE id=?', (tid,)).fetchone():
                return False  # Re-import never replaces local drafts or corrections.
            db.execute('INSERT INTO trials(id,asset,settings,created) VALUES(?,?,?,?)',
                       (tid, encode(asset), encode({'swingStart': 0, 'swingEnd': len(clip['frames']) - 1}), now()))
            for i, frame in enumerate(clip['frames']):
                r = by_index.get(i) or {
                    'id': f'{record_owner}:{clip["sha256"]}:{i}', 'schema': 1, 'owner': record_owner,
                    'videoSHA256': clip['sha256'], 'frameIndex': i, 'frame': frame,
                    'width': clip['width'], 'height': clip['height'],
                    'session': asset['row'].get('session'), 'trial': asset['row'].get('trial'),
                    'camera': asset['row'].get('camera'),
                    'source': {'source': asset['row']['source'], 'uri': asset['row']['uri'], 'generation': asset['row']['generation']},
                    'points': [{'x': None, 'y': None, 'visibility': 0, 'confidence': None} for _ in range(17)],
                    'box': None, 'status': 'draft'}
                if (r['videoSHA256'] != clip['sha256'] or r['frame'] != frame or
                    r['width'] != clip['width'] or r['height'] != clip['height'] or len(r['points']) != 17 or
                    r['owner'] != record_owner or r['id'] != f'{record_owner}:{clip["sha256"]}:{i}'):
                    raise ValueError('Annotation source identity mismatch')
                for joint in JOINTS:
                    if point(r, joint) is not None:
                        validate_point(point(r, joint))
                meta = {'locked': r.get('status') == 'reviewed', 'humanEdited': False, 'passes': {},
                        'uncertain': [], 'note': '', 'acknowledged': [], 'changed': False, 'version': 0}
                db.execute('INSERT INTO frames VALUES(?,?,?,?,?)', (tid, i, encode(r), encode(r), encode(meta)))
        return True

    def _frames(self, db, tid):
        return [{'index': r['idx'], 'record': json.loads(r['record']),
                 'baseline': json.loads(r['baseline']), 'meta': json.loads(r['meta'])}
                for r in db.execute('SELECT * FROM frames WHERE trial=? ORDER BY idx', (tid,))]

    def _trial(self, db, tid):
        t = db.execute('SELECT * FROM trials WHERE id=?', (tid,)).fetchone()
        if not t:
            raise ValueError('Unknown local trial')
        return dict(t)

    def issues(self, frames):
        issues = {f['index']: [] for f in frames}
        for f in frames:
            i, r = f['index'], f['record']
            if not complete(r):
                issues[i].append('missing-points')
            issues[i] += ['uncertain:' + j for j in f['meta']['uncertain']]
            if i == 0:
                continue
            prev = frames[i - 1]['record']
            for joint in JOINTS:
                a, b = point(prev, joint), point(r, joint)
                if located(a) and located(b) and a.get('visibility') and b.get('visibility'):
                    distance = math.hypot(a['x'] - b['x'], a['y'] - b['y'])
                    if distance > (0.18 if joint.startswith('bat') else 0.08):
                        issues[i].append('jump:' + joint)
            for left, right in zip(JOINTS[:12:2], JOINTS[1:12:2]):
                a, b, c, d = point(prev, left), point(prev, right), point(r, left), point(r, right)
                if all(located(p) and p.get('visibility') for p in (a, b, c, d)):
                    dist = lambda x, y: math.hypot(x['x'] - y['x'], x['y'] - y['y'])
                    if dist(a, c) + dist(b, d) > dist(a, d) + dist(b, c) + 0.06:
                        issues[i].append('possible-swap:' + left.replace('left_', ''))
        return issues

    def summary(self, frames):
        flags = self.issues(frames)
        remaining = {p: 0 for p in PASSES}
        for f in frames:
            passes = f['meta']['passes']
            for p in PASSES:
                remaining[p] += (len(JOINTS) - len(passes.get('joints', {}))) if p == 'joints' else int(not passes.get(p))
        return {'frames': len(frames), 'changed': sum(f['meta']['changed'] for f in frames),
                'locked': sum(f['meta']['locked'] for f in frames), 'remaining': remaining,
                'issues': sum(len(set(flags[f['index']]) - set(f['meta']['acknowledged'])) for f in frames)}

    def list(self):
        with self.connection() as db:
            result = []
            for t in db.execute('SELECT * FROM trials ORDER BY created'):
                asset = json.loads(t['asset'])
                result.append({'id': t['id'], 'state': t['state'], 'revision': t['revision'],
                               'row': asset['row'], 'lease': {'owner': t['lease_owner'], 'until': t['lease_until']},
                               'summary': self.summary(self._frames(db, t['id'])),
                               'agentReports': self.report_summary(db, t['id'])})
            return result

    def reports(self, db, tid):
        return [{**json.loads(r['content']), 'runId': r['id'], 'actor': r['actor'], 'created': r['created']}
                for r in db.execute('SELECT * FROM agent_reports WHERE trial=? ORDER BY created DESC', (tid,))]

    def report_summary(self, db, tid):
        reports = self.reports(db, tid)
        return {'runs': len(reports), 'pending': sum(c.get('decision') == 'pending' for r in reports for c in r.get('changes', []))}

    def get(self, tid, private=False):
        with self.connection() as db:
            t = self._trial(db, tid)
            asset, frames = json.loads(t['asset']), self._frames(db, tid)
            if not private:
                asset.pop('video', None)
            flags = self.issues(frames)
            for f in frames:
                f['issues'] = flags[f['index']]
            return {'id': tid, 'state': t['state'], 'revision': t['revision'], 'asset': asset,
                    'settings': json.loads(t['settings']), 'frames': frames, 'summary': self.summary(frames),
                    'agentReports': self.reports(db, tid),
                    'lease': {'owner': t['lease_owner'], 'until': t['lease_until']}}

    def claim(self, tid, body):
        owner = str(body.get('owner', '')).strip()
        if not owner or len(owner) > 100:
            raise ValueError('A short agent identity is required')
        seconds = body.get('seconds', 300)
        if type(seconds) is not int or not 30 <= seconds <= 900:
            raise ValueError('Lease duration must be 30–900 seconds')
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            if tid == 'next':
                row = db.execute("SELECT id FROM trials WHERE state='draft' AND (lease_until IS NULL OR lease_until<=?) ORDER BY created LIMIT 1", (time.time(),)).fetchone()
                if not row:
                    raise Conflict('No unclaimed draft trial is available')
                tid = row['id']
            t = self._trial(db, tid)
            if t['state'] != 'draft':
                raise Conflict('Only draft trials can be claimed')
            if (t['lease_until'] or 0) > time.time() and body.get('leaseToken') != t['lease_token']:
                raise Conflict('Trial already claimed; wait for release or lease expiry')
            token = t['lease_token'] if body.get('leaseToken') and body['leaseToken'] == t['lease_token'] else uuid.uuid4().hex
            until = time.time() + seconds
            db.execute('UPDATE trials SET lease_owner=?,lease_token=?,lease_until=? WHERE id=?', (owner, token, until, tid))
            return {'id': tid, 'owner': owner, 'leaseToken': token, 'until': until, 'revision': t['revision']}

    def mutate(self, tid, kind, body):
        if kind == 'upload':
            raise Conflict('Uploads are disabled. User approval and a future conditional publisher are required.')
        request_id = body.get('requestId')
        if not isinstance(request_id, str) or not 8 <= len(request_id) <= 100:
            raise ValueError('A unique requestId (8–100 characters) is required')
        signature = hashlib.sha256(encode({'kind': kind, 'body': body}).encode()).hexdigest()
        actor = body.get('actor', {})
        if actor.get('type') not in ('human', 'agent') or not isinstance(actor.get('id'), str) or not actor['id']:
            raise ValueError('actor requires type human/agent and id')
        human = actor['type'] == 'human'
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            prior = db.execute('SELECT * FROM requests WHERE trial=? AND request_id=?', (tid, request_id)).fetchone()
            if prior:
                if prior['digest'] != signature:
                    raise Conflict('requestId reused with different content')
                return json.loads(prior['response'])
            t = self._trial(db, tid)
            if type(body.get('expectedRevision')) is not int or body['expectedRevision'] != t['revision']:
                raise Conflict('Draft changed. Reload and reapply your edit to the current revision.')
            if not human and (body.get('leaseToken') != t['lease_token'] or actor['id'] != t['lease_owner'] or (t['lease_until'] or 0) <= time.time()):
                raise Conflict('A current per-trial agent lease is required')
            if not human and kind not in ('batch', 'interpolate', 'submit', 'release', 'agent-report'):
                raise ValueError('Only the human reviewer can perform this action')
            if not human and t['state'] != 'draft' and kind != 'release':
                raise Conflict('Agent writes are closed after submission for human review')
            frames = self._frames(db, tid)
            before = deepcopy(frames)
            settings = json.loads(t['settings'])
            old_settings, old_state = deepcopy(settings), t['state']
            touched = set()

            def frame_at(index):
                if type(index) is not int or not 0 <= index < len(frames):
                    raise ValueError('Frame index must be zero-based and in range')
                return frames[index]

            def apply_changes(changes, interpolation=None, accept_locked=False):
                if not isinstance(changes, list) or not 1 <= len(changes) <= 2000:
                    raise ValueError('Send 1–2000 explicit point changes')
                seen = set()
                for edit in changes:
                    f, joint = frame_at(edit['frameIndex']), edit['joint']
                    if joint not in JOINTS:
                        raise ValueError('Unknown anatomical joint')
                    if (f['index'], joint) in seen:
                        raise ValueError('Duplicate point in the same batch')
                    seen.add((f['index'], joint))
                    if f['meta']['locked'] and not (human and accept_locked):
                        raise Conflict('Frame is a locked human reference; unlock it explicitly before editing')
                    if not human and f['meta']['humanEdited']:
                        raise Conflict('Human corrections are protected from agent replacement')
                    patch = edit.get('point', {})
                    if set(patch) - {'x', 'y', 'visibility'}:
                        raise ValueError('Point patches accept only x, y and visibility')
                    value = {**(point(f['record'], joint) or {}), **patch}
                    validate_point(value)
                    value['source'] = 'interpolated' if interpolation else 'manual'
                    value.pop('reviewedAt', None)
                    value.pop('interpolation', None)
                    if interpolation:
                        value['interpolation'] = interpolation
                    set_point(f['record'], joint, value)
                    if not joint.startswith('bat') and located(value) and value['visibility']:
                        points = [point(f['record'], j) for j in JOINTS[:12]]
                        points = [p for p in points if located(p) and p.get('visibility')]
                        xs, ys = [p['x'] for p in points], [p['y'] for p in points]
                        f['record']['box'] = [max(0, min(xs) - .01), max(0, min(ys) - .01), min(1, max(xs) + .01), min(1, max(ys) + .01)]
                    clear_review(f['record'])
                    f['meta'].update(passes={}, acknowledged=[], humanEdited=human or f['meta']['humanEdited'])
                    if 'uncertain' in edit:
                        uncertain = set(f['meta']['uncertain'])
                        if edit['uncertain']:
                            uncertain.add(joint)
                        else:
                            uncertain.discard(joint)
                        f['meta']['uncertain'] = sorted(uncertain)
                    touched.add(f['index'])

            if kind == 'batch':
                apply_changes(body.get('changes'))
                t['state'] = 'draft'
            elif kind == 'agent-report':
                report = deepcopy(body.get('report', {}))
                if report.get('trialId') != tid or not isinstance(report.get('passes'), list):
                    raise ValueError('Report must identify this trial and contain actual agent pass evidence')
                changes = report.setdefault('changes', [])
                if not isinstance(changes, list) or len(changes) > 2000:
                    raise ValueError('Invalid proposal count')
                if changes and report.get('sourceRevision') != t['revision']:
                    raise Conflict('Reference draft changed during agent review; reconcile proposals first')
                if self.report_summary(db, tid)['pending']:
                    raise Conflict('Resolve the previous agent proposals before staging another set')
                seen = set()
                for proposal in changes:
                    f, joint = frame_at(proposal.get('frameIndex')), proposal.get('joint')
                    if joint not in JOINTS or (f['index'], joint) in seen:
                        raise ValueError('Invalid or duplicate proposed joint')
                    seen.add((f['index'], joint))
                    validate_point(proposal.get('point'))
                    if set(proposal['point']) - {'x', 'y', 'visibility'}:
                        raise ValueError('Proposals accept x, y and visibility only')
                    proposal.update(decision='pending', previous=deepcopy(point(f['record'], joint)))
                for check in report['passes']:
                    if not isinstance(check, dict) or not isinstance(check.get('name'), str) or not isinstance(check.get('frames'), list):
                        raise ValueError('Each agent pass requires a name and inspected frame indices')
                    for i in check['frames']:
                        frame_at(i)
                db.execute('INSERT INTO agent_reports VALUES(?,?,?,?,?)',
                           (request_id, tid, actor['id'], now(), encode(report)))
                t['state'] = 'review'
            elif kind in ('accept-proposal', 'reject-proposal'):
                row = db.execute('SELECT * FROM agent_reports WHERE id=? AND trial=?', (body.get('runId'), tid)).fetchone()
                if not row:
                    raise ValueError('Unknown agent report')
                report = json.loads(row['content'])
                frame_at(body.get('frameIndex'))
                chosen = [c for c in report['changes'] if c['decision'] == 'pending' and c['frameIndex'] == body['frameIndex'] and
                          (body.get('allFrame') is True or c['joint'] == body.get('joint'))]
                if not chosen:
                    raise ValueError('No pending proposals selected')
                if kind == 'accept-proposal':
                    for c in chosen:
                        current = point(frames[c['frameIndex']]['record'], c['joint'])
                        geometry = lambda p: {k: (p or {}).get(k) for k in ('x', 'y', 'visibility')}
                        if geometry(current) != geometry(c['previous']):
                            raise Conflict('This point has changed since the proposal. Keep the current correction or review it again.')
                    apply_changes(chosen, accept_locked=True)
                for c in chosen:
                    c.update(decision='accepted' if kind == 'accept-proposal' else 'rejected', decidedAt=now(), decidedBy=actor['id'])
                db.execute('UPDATE agent_reports SET content=? WHERE id=?', (encode(report), row['id']))
                body = {**body, 'proposalKeys': [[c['frameIndex'], c['joint']] for c in chosen]}
                t['state'] = 'review'
            elif kind == 'interpolate':
                start, end, joint = body.get('start'), body.get('end'), body.get('joint')
                a, b = frame_at(start), frame_at(end)
                if joint not in JOINTS or not 1 < end - start <= 10 or end >= settings['swingStart']:
                    raise ValueError('Interpolate only a setup gap of 2–10 frames before swing start')
                pa, pb = point(a['record'], joint), point(b['record'], joint)
                if not a['meta']['locked'] or not b['meta']['locked'] or not all(located(p) and p.get('visibility') for p in (pa, pb)):
                    raise ValueError('Both endpoints must be located, locked human references')
                changes = []
                for i in range(start + 1, end):
                    f = frames[i]
                    if f['meta']['locked'] or f['meta']['humanEdited']:
                        continue
                    fraction = (i - start) / (end - start)
                    changes.append({'frameIndex': i, 'joint': joint, 'point': {
                        'x': pa['x'] + (pb['x'] - pa['x']) * fraction,
                        'y': pa['y'] + (pb['y'] - pa['y']) * fraction,
                        'visibility': min(pa['visibility'], pb['visibility'])}})
                apply_changes(changes, {'method': 'reviewed-position-linear', 'start': start, 'end': end})
                t['state'] = 'draft'
            elif kind == 'lock':
                f = frame_at(body.get('frameIndex'))
                f['meta']['locked'] = bool(body.get('locked'))
                touched.add(f['index'])
            elif kind == 'note':
                f = frame_at(body.get('frameIndex'))
                f['meta']['note'] = str(body.get('note', ''))[:2000]
                touched.add(f['index'])
            elif kind == 'flag':
                f = frame_at(body.get('frameIndex'))
                if body.get('joint') not in JOINTS:
                    raise ValueError('Unknown anatomical joint')
                uncertain = set(f['meta']['uncertain'])
                if body.get('uncertain'):
                    uncertain.add(body['joint'])
                else:
                    uncertain.discard(body['joint'])
                f['meta'].update(uncertain=sorted(uncertain), passes={})
                clear_review(f['record'])
                touched.add(f['index'])
                t['state'] = 'draft'
            elif kind == 'acknowledge':
                f = frame_at(body.get('frameIndex'))
                f['meta']['acknowledged'] = [issue for issue in self.issues(frames)[f['index']] if issue.startswith(('jump:', 'possible-swap:'))]
                touched.add(f['index'])
            elif kind == 'review':
                f = frame_at(body.get('frameIndex'))
                if any(c['frameIndex'] == f['index'] and c.get('decision') == 'pending'
                       for r in self.reports(db, tid) for c in r.get('changes', [])):
                    raise ValueError('Accept or keep the agent suggestions on this frame before marking a pass')
                p = body.get('pass')
                if p not in PASSES:
                    raise ValueError('Unknown review pass')
                if not complete(f['record']) or f['meta']['uncertain']:
                    raise ValueError('Resolve missing or uncertain points before marking a pass')
                flags = set(self.issues(frames)[f['index']]) - set(f['meta']['acknowledged'])
                if flags:
                    raise ValueError('Inspect and acknowledge the motion flags first')
                passes = f['meta']['passes']
                if p != 'placement' and not passes.get('placement'):
                    raise ValueError('Complete placement inspection first')
                if p in ('joints', 'final') and not passes.get('replay'):
                    raise ValueError('Complete the synchronized replay pass first')
                if p == 'final' and len(passes.get('joints', {})) != len(JOINTS):
                    raise ValueError('Inspect every joint before the final pass')
                evidence = {'actor': actor['id'], 'at': now(), 'revision': t['revision']}
                if p == 'joints':
                    selected = JOINTS if body.get('allJointsInspected') is True else [body.get('joint')]
                    if any(j not in JOINTS for j in selected):
                        raise ValueError('Select a joint to inspect')
                    for j in selected:
                        passes.setdefault('joints', {})[j] = evidence
                else:
                    passes[p] = evidence
                touched.add(f['index'])
                t['state'] = 'review'
            elif kind == 'settings':
                start, end = body.get('swingStart'), body.get('swingEnd')
                frame_at(start); frame_at(end)
                if start > end:
                    raise ValueError('Swing start must precede end')
                settings.update(swingStart=start, swingEnd=end)
            elif kind == 'undo':
                event = db.execute("SELECT * FROM events WHERE trial=? AND kind IN ('batch','interpolate','accept-proposal') AND id NOT IN (SELECT CAST(json_extract(detail,'$.undoEvent') AS INTEGER) FROM events WHERE trial=? AND kind='undo') ORDER BY id DESC LIMIT 1", (tid, tid)).fetchone()
                if not event:
                    raise ValueError('No point edit is available to undo')
                restored = json.loads(event['before'])
                applied = {f['index']: f for f in json.loads(event['after'])}
                for f in restored:
                    # Neighbor review invalidation is logged with the edit but did not
                    # change neighbor coordinates. It must not require unlocking them.
                    if f['record'] == applied[f['index']]['record']:
                        continue
                    current = frames[f['index']]
                    if current['meta']['locked'] and event['kind'] != 'accept-proposal':
                        raise Conflict('Unlock the edited reference frame before undoing')
                    current['record'] = f['record']
                    current['meta'].update(passes={}, acknowledged=[], humanEdited=True,
                                           uncertain=f['meta']['uncertain'])
                    clear_review(current['record'])
                    touched.add(f['index'])
                if event['kind'] == 'accept-proposal':
                    detail = json.loads(event['detail'])
                    row = db.execute('SELECT * FROM agent_reports WHERE id=? AND trial=?', (detail['runId'], tid)).fetchone()
                    report = json.loads(row['content'])
                    for c in report['changes']:
                        if [c['frameIndex'], c['joint']] in detail['proposalKeys']:
                            c['decision'] = 'pending'
                            c.pop('decidedAt', None); c.pop('decidedBy', None)
                    db.execute('UPDATE agent_reports SET content=? WHERE id=?', (encode(report), row['id']))
                body = {**body, 'undoEvent': event['id']}
                t['state'] = 'draft'
            elif kind == 'submit':
                t['state'] = 'review'
            elif kind == 'reopen':
                t['state'] = 'draft'
            elif kind == 'approve':
                if self.report_summary(db, tid)['pending']:
                    raise ValueError('Accept or reject the pending agent proposals before approval')
                summary = self.summary(frames)
                if any(summary['remaining'].values()) or summary['issues']:
                    raise ValueError('Approval requires all four passes and no unresolved issues')
                if body.get('confirmation') != 'Approve local draft':
                    raise ValueError('Explicit local approval is required')
                for f in frames:
                    r = f['record']
                    validate_box(r)
                    r.update(status='reviewed', reviewedAt=now(), reviewMethod='frame')
                    r['batKnob']['reviewedAt'] = now()
                    touched.add(f['index'])
                t['state'] = 'approved'
            elif kind == 'release':
                pass
            else:
                raise ValueError('Unknown operation')

            if old_state == 'approved' and kind != 'approve' and t['state'] == 'approved':
                t['state'] = 'review'  # Any new revision requires a fresh approval action.

            # A coordinate edit invalidates temporal inspection on its immediate neighbors too.
            if kind in ('batch', 'interpolate', 'undo', 'accept-proposal'):
                for i in list(touched):
                    for n in (i - 1, i + 1):
                        if 0 <= n < len(frames):
                            for p in ('replay', 'final'):
                                frames[n]['meta']['passes'].pop(p, None)
                            frames[n]['meta']['acknowledged'] = []
                            touched.add(n)
            revision = t['revision'] + 1
            for i in touched:
                f = frames[i]
                geometry = lambda p: {k: (p or {}).get(k) for k in ('x', 'y', 'visibility')}
                f['meta']['changed'] = any(geometry(point(f['record'], j)) != geometry(point(f['baseline'], j)) for j in JOINTS)
                f['meta']['version'] = revision
                db.execute('UPDATE frames SET record=?,meta=? WHERE trial=? AND idx=?', (encode(f['record']), encode(f['meta']), tid, i))
            release = human or kind in ('release', 'submit', 'agent-report')
            db.execute('UPDATE trials SET state=?,revision=?,settings=?,lease_owner=?,lease_token=?,lease_until=? WHERE id=?',
                       (t['state'], revision, encode(settings), None if release else t['lease_owner'],
                        None if release else t['lease_token'], None if release else t['lease_until'], tid))
            detail = {'requestId': request_id, 'note': str(body.get('reason', ''))[:1000],
                      'stateBefore': old_state, 'stateAfter': t['state'],
                      'settingsBefore': old_settings, 'settingsAfter': settings}
            if 'undoEvent' in body:
                detail['undoEvent'] = body['undoEvent']
            if kind in ('accept-proposal', 'reject-proposal'):
                detail.update(runId=body['runId'], proposalKeys=body['proposalKeys'])
            db.execute('INSERT INTO events(trial,revision,actor,kind,created,before,after,detail) VALUES(?,?,?,?,?,?,?,?)',
                       (tid, revision, encode(actor), kind, now(), encode([before[i] for i in sorted(touched)]),
                        encode([frames[i] for i in sorted(touched)]), encode(detail)))
            response = {'id': tid, 'revision': revision, 'state': t['state'], 'changedFrames': sorted(touched)}
            db.execute('INSERT INTO requests VALUES(?,?,?,?)', (tid, request_id, signature, encode(response)))
            return response

    def history(self, tid):
        with self.connection() as db:
            return [{**dict(r), 'actor': json.loads(r['actor']), 'detail': json.loads(r['detail'])}
                    for r in db.execute('SELECT id,revision,actor,kind,created,detail FROM events WHERE trial=? ORDER BY id DESC LIMIT 60', (tid,))]

    def export(self, tid):
        trial = self.get(tid)
        if trial['state'] != 'approved':
            raise Conflict('Only an explicitly approved local draft can be exported')
        records = [f['record'] for f in trial['frames']]
        return {'format': 'swing-review-approved-v1', 'assignmentId': tid,
                'expectedSourceGeneration': trial['asset']['sourceGeneration'],
                'sourceDigest': trial['asset']['sourceDigest'], 'localRevision': trial['revision'],
                'exportedAt': now(), 'uploadAuthorized': False,
                'poseSubmission': {'version': 1, 'records': records,
                                   'counts': {'reviewed': len(records), 'drafts': 0, 'smoothedApproved': 0}},
                'reviewEvidence': [{'frameIndex': f['index'], **f['meta']} for f in trial['frames']],
                'agentReports': trial['agentReports']}
