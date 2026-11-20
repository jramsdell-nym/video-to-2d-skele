"""Read-only adapter to the existing GCS labeler contract. No cloud write API."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT.parent / 'experiments/vision-improvement/cycles/2026-11-18/dataset'
PREFIX = os.environ.get('SWING_LABELER_PREFIX', 'gs://YOUR_LABELER_BUCKET/app/v1/').rstrip('/') + '/'
VIDEO_PREFIX = os.environ.get('SWING_VIDEO_PREFIX', 'gs://YOUR_VIDEO_BUCKET/').rstrip('/') + '/'
GCLOUD = os.environ.get('SWING_GCLOUD', 'gcloud')
MAX_VIDEO_BYTES = 128 * 1024 * 1024
MAX_LOCAL_FRAMES = 3600
LOCAL_VIDEO_EXTENSIONS = {'.mov', '.mp4', '.m4v'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for part in iter(lambda: file.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def verified_json(path):
    path = Path(path)
    meta = json.loads(path.with_suffix(path.suffix + '.metadata.json').read_text())
    if digest(path) != meta['sha256']:
        raise ValueError('Snapshot checksum mismatch: ' + path.name)
    return json.loads(path.read_text()), meta


def decode_frame_index(video, *, local=False):
    import av

    frames = []
    dimensions = None
    with av.open(str(video), format='mov' if local else None) as container:
        if not container.streams.video:
            raise ValueError('The source contains no video track')
        for i, frame in enumerate(container.decode(video=0)):
            if local and i >= MAX_LOCAL_FRAMES:
                raise ValueError(f'Local videos must contain at most {MAX_LOCAL_FRAMES:,} frames')
            if frame.pts is None or frame.time_base is None:
                raise ValueError('The source contains a frame without a presentation timestamp')
            size = (frame.width, frame.height)
            stamp = float(frame.pts * frame.time_base)
            if local and frames and (size != dimensions or stamp <= frames[-1]['time']):
                raise ValueError('Local video requires fixed dimensions and increasing frame timestamps')
            dimensions = size
            frames.append({'pts': frame.pts, 'timeBase': str(frame.time_base), 'time': stamp, 'index': i})
    if not frames:
        raise ValueError('No decodable source frames')
    width, height = dimensions
    return width, height, frames


def file_version(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class LocalVideoLibrary:
    def __init__(self, directory, data):
        self.directory = Path(directory).resolve()
        self.cache = Path(data) / 'local-sources'

    def entries(self):
        if not self.directory.is_dir():
            return
        for path in sorted(self.directory.rglob('*'), key=lambda path: str(path).casefold()):
            if path.suffix.lower() not in LOCAL_VIDEO_EXTENSIONS or path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(self.directory)
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            version = f'{relative.as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}'
            trial_id = 'local-' + hashlib.sha256(version.encode('utf-8', 'surrogateescape')).hexdigest()[:40]
            yield {'id': trial_id, 'trial': path.stem, 'source': 'Local video',
                   'session': relative.parent.as_posix() if len(relative.parts) > 1 else 'Local videos',
                   'cached': False, 'available': True}, path, stat

    def catalog(self):
        return [row for row, _, _ in self.entries()]

    def load(self, trial_id):
        from av import FFmpegError

        for row, path, stat in self.entries():
            if row['id'] == trial_id:
                break
        else:
            raise ValueError('Local video is missing or changed. Reopen the video picker.')
        if not 0 < stat.st_size <= MAX_VIDEO_BYTES:
            raise ValueError('Local videos must be between 1 byte and 128 MiB')
        self.cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        with TemporaryDirectory(dir=self.cache) as staging:
            copy = Path(staging) / ('source' + path.suffix.lower())
            try:
                with path.open('rb') as source, copy.open('xb') as destination:
                    if file_version(os.fstat(source.fileno())) != file_version(stat):
                        raise ValueError('Local video changed while importing. Reopen the video picker.')
                    shutil.copyfileobj(source, destination)
                    after = os.fstat(source.fileno())
            except FileNotFoundError as error:
                raise ValueError('Local video was moved before import. Reopen the video picker.') from error
            if file_version(after) != file_version(stat):
                raise ValueError('Local video changed while importing. Reopen the video picker.')
            try:
                width, height, frames = decode_frame_index(copy, local=True)
            except FFmpegError as error:
                raise ValueError('Could not decode the local video. Select a playable MP4, MOV or M4V file.') from error
            sha = digest(copy)
            video = self.cache / (sha + copy.suffix)
            copy.chmod(0o600)
            copy.replace(video)
        uri = f'local-video:{sha}'
        return {'id': trial_id, 'clip': {'id': sha[:20], 'sha256': sha, 'width': width,
                                         'height': height, 'frames': frames},
                'row': {**row, 'uri': uri, 'generation': None}, 'video': str(video.resolve()),
                'records': [], 'sourceGeneration': None, 'sourceDigest': sha, 'sourceUri': uri,
                'owner': 'local-review'}


class ReadOnlyBackend:
    def __init__(self, data, snapshot=SNAPSHOT):
        self.data, self.snapshot = Path(data).resolve(), Path(snapshot).resolve()
        self.cloud = self.data / 'cloud'
        self.cloud.mkdir(parents=True, exist_ok=True)
        self.cloud.chmod(0o700)

    def command(self, *args):
        # Only these callers build commands. Never accept arbitrary CLI arguments from HTTP.
        result = subprocess.run([GCLOUD, 'storage', *args, '--quiet'],
                                capture_output=True, text=True, timeout=120)
        if result.returncode:
            message = result.stderr.lower()
            if 'reauth' in message or 'login' in message or 'credentials' in message:
                raise ValueError('GCS login needs renewal. Run gcloud auth login in your terminal, then retry.')
            raise ValueError('Read-only GCS request failed. Check configured gcloud access; no upload was attempted.')
        return result.stdout

    def fetch(self, uri, target, generation=None):
        if not uri.startswith((PREFIX, VIDEO_PREFIX)) or '..' in uri.split('/'):
            raise ValueError('Unsupported source URI')
        if generation is not None and not re.fullmatch(r'[1-9][0-9]*', str(generation)):
            raise ValueError('Invalid object generation')
        metadata = json.loads(self.command('objects', 'describe', uri + ('#' + str(generation) if generation else ''), '--format=json'))
        gen = str(metadata['generation'])
        if generation and gen != str(generation):
            raise ValueError('Source generation changed')
        if int(metadata.get('size', 0)) > MAX_VIDEO_BYTES:
            raise ValueError('Source exceeds the 128 MiB local import limit')
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + '.partial')
        self.command('cp', uri + '#' + gen, str(partial))
        partial.chmod(0o600)
        partial.replace(target)
        meta = {'uri': uri, 'generation': gen, 'sha256': digest(target),
                'retrievedAt': datetime.now(timezone.utc).isoformat()}
        mp = target.with_suffix(target.suffix + '.metadata.json')
        mp.write_text(json.dumps(meta)); mp.chmod(0o600)
        return meta

    def refresh(self):
        # Local snapshots only; never reserve or advance the shared assignment queue.
        directory = self.cloud / 'snapshots' / uuid.uuid4().hex
        for name in ('catalog', 'queue'):
            self.fetch(PREFIX + name + '.json', directory / (name + '.json'))
        pointer = self.cloud / (directory.name + '.tmp')
        pointer.write_text(json.dumps({'directory': str(directory.resolve())}))
        pointer.chmod(0o600)
        pointer.replace(self.cloud / 'current.json')
        return self.status()

    def source(self, name):
        pointer = self.cloud / 'current.json'
        if pointer.exists():
            return Path(json.loads(pointer.read_text())['directory']) / (name + '.json')
        live = self.cloud / (name + '.json')
        return live if live.exists() else self.snapshot / 'private' / (name + '.json')

    def is_live(self):
        return self.source('catalog').is_relative_to(self.cloud)

    def status(self):
        try:
            _, meta = verified_json(self.source('catalog'))
        except FileNotFoundError:
            return {'mode': 'local only', 'catalogGeneration': None, 'retrievedAt': None,
                    'liveSnapshot': False, 'uploadEnabled': False}
        return {'mode': 'read-only GCS', 'catalogGeneration': meta['generation'],
                'retrievedAt': meta['retrievedAt'], 'liveSnapshot': self.is_live(),
                'uploadEnabled': False}

    def catalog(self):
        catalog, _ = verified_json(self.source('catalog'))
        queue, _ = verified_json(self.source('queue'))
        results = []
        for row in catalog['videos']:
            if row.get('source') != 'Strike Mobile' or row.get('archived') or row.get('availability') == 'missing':
                continue
            task = queue.get('tasks', {}).get(row['id'], {})
            reviews = queue.get('reviews', {})
            review = reviews.get('video:' + row['id'], {})
            group = reviews.get('group:' + row.get('groupId', ''), {})
            rejected = review.get('status') == 'reject' or group.get('view') == 'reject' or task.get('status') == 'rejected'
            keys = row.get('trialKeys') or (['trial:' + row['trial']] if row.get('trial') and row['trial'] != 'unknown' else ['video:' + row['id']])
            reserved = any(queue.get('trialOwners', {}).get(k) or queue.get('completedTrials', {}).get(k) for k in keys)
            available = not task and not rejected and not reserved and (group.get('view') not in ('back', 'side') or review.get('status') == 'keep')
            cached = self.snapshot / 'private/assignments' / (row['id'] + '.json')
            results.append({**row, 'sharedStatus': task.get('status', 'unassigned'),
                            'cached': cached.exists(), 'rejected': rejected, 'available': available})
        return results

    def load(self, trial_id):
        if not re.fullmatch(r'[a-zA-Z0-9-]{8,80}', trial_id):
            raise ValueError('Invalid trial ID')
        rows = {row['id']: row for row in self.catalog()}
        if trial_id not in rows:
            raise ValueError('Trial not in the eligible source catalog')
        row = rows[trial_id]
        if row['sharedStatus'] == 'active':
            raise ValueError('This trial is active in the shared labeler. Choose another trial.')
        if row['rejected'] or row['sharedStatus'] == 'rejected':
            raise ValueError('This trial has a shared rejection decision')
        if row['sharedStatus'] == 'unassigned' and not row['available']:
            raise ValueError('This trial is reserved, already completed in another view, or excluded by camera review')
        assignment = self.snapshot / 'private/assignments' / (trial_id + '.json')
        doc, metadata = None, None
        # Prefer a fresh, pinned assignment when catalog has been explicitly refreshed.
        if row['sharedStatus'] == 'completed' and self.is_live():
            assignment = self.cloud / 'assignments' / (trial_id + '.json')
            self.fetch(PREFIX + 'assignments/' + trial_id + '.json', assignment)
        if assignment.exists():
            doc, metadata = verified_json(assignment)
            if doc.get('id') != trial_id:
                raise ValueError('Assignment identity does not match the catalog')
        sha = (doc or {}).get('clip', {}).get('sha256')
        video = self.snapshot / 'sources' / (str(sha) + '.mp4')
        if not video.exists():
            video = self.cloud / 'videos' / (trial_id + '.mp4')
            self.fetch(row['uri'], video, row['generation'])
        actual_sha = digest(video)
        if sha and actual_sha != sha:
            raise ValueError('Video hash does not match the annotation source')
        # Verify every decoded frame, including exact PTS. No FPS-based seeking.
        width, height, frames = decode_frame_index(video)
        clip = (doc or {}).get('clip') or {'id': actual_sha[:20], 'sha256': actual_sha,
                                         'width': width, 'height': height, 'frames': frames}
        if clip['width'] != width or clip['height'] != height or len(clip['frames']) != len(frames):
            raise ValueError('Decoded video does not match clip dimensions or frame count')
        for expected, actual in zip(clip['frames'], frames):
            if any(expected[k] != actual[k] for k in ('pts', 'timeBase', 'index')) or abs(expected['time'] - actual['time']) > 1e-6:
                raise ValueError('Decoded timestamps do not match the saved clip')
        # Never retain assignment tokens, mutation receipts or account profiles in the local API.
        return {'id': trial_id, 'clip': clip, 'row': row, 'video': str(video.resolve()),
                'records': (doc or {}).get('poseSubmission', {}).get('records', []),
                'sourceGeneration': metadata['generation'] if metadata else None,
                'sourceDigest': metadata['sha256'] if metadata else None,
                'sourceUri': metadata['uri'] if metadata else row['uri'],
                'owner': (doc or {}).get('owner', 'local-review')}
