import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import ReadOnlyBackend, digest


def write_snapshot(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    path.with_suffix(path.suffix + '.metadata.json').write_text(json.dumps({
        'sha256': digest(path), 'generation': '123', 'retrievedAt': '2026-11-18T12:00:00Z'}))


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.snapshot = self.root / 'snapshot'
        self.backend = ReadOnlyBackend(self.root / 'data', self.snapshot)
        rows = [{'id': f'trial00{i}', 'trial': f'trial{i}', 'source': 'Strike Mobile', 'groupId': f'group{i}',
                 'availability': 'listed', 'uri': 'gs://YOUR_VIDEO_BUCKET/test.mp4', 'generation': '10'} for i in range(5)]
        write_snapshot(self.snapshot / 'private/catalog.json', {'version': 1, 'videos': rows})
        write_snapshot(self.snapshot / 'private/queue.json', {'tasks': {'trial004': {'status': 'active'}},
              'trialOwners': {'trial:trial3': 'someone'}, 'completedTrials': {},
              'reviews': {'video:trial001': {'status': 'reject'}, 'group:group2': {'view': 'back'}}})

    def tearDown(self):
        self.tmp.cleanup()

    def test_shared_exclusions_and_reservations(self):
        rows = {r['id']: r for r in self.backend.catalog()}
        self.assertTrue(rows['trial000']['available'])
        self.assertTrue(rows['trial001']['rejected'])
        self.assertFalse(rows['trial002']['available'])
        self.assertFalse(rows['trial003']['available'])
        with self.assertRaisesRegex(ValueError, 'active'):
            self.backend.load('trial004')

    def test_corrupt_source_snapshot_rejected(self):
        path = self.snapshot / 'private/catalog.json'
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.backend.catalog()

    def test_refresh_uses_atomic_snapshot_pointer(self):
        def fetch(uri, path):
            name = Path(uri).name
            value = json.loads((self.snapshot / 'private' / name).read_text())
            write_snapshot(Path(path), value)
        with patch.object(self.backend, 'fetch', side_effect=fetch):
            self.assertTrue(self.backend.refresh()['liveSnapshot'])
        prior = self.backend.source('catalog')
        def failed(uri, path):
            if uri.endswith('queue.json'):
                raise ValueError('network failure')
            fetch(uri, path)
        with patch.object(self.backend, 'fetch', side_effect=failed):
            with self.assertRaises(ValueError):
                self.backend.refresh()
        self.assertEqual(self.backend.source('catalog'), prior)
        self.assertEqual(len(self.backend.catalog()), 5)

    def test_download_is_generation_pinned_and_read_only(self):
        calls = []
        def command(*args):
            calls.append(args)
            if args[0] == 'objects':
                return json.dumps({'generation': '456', 'size': 2})
            Path(args[2]).write_text('{}')
            return ''
        with patch.object(self.backend, 'command', side_effect=command):
            self.backend.fetch('gs://YOUR_LABELER_BUCKET/app/v1/catalog.json', self.root / 'download.json')
        self.assertEqual([c[0] for c in calls], ['objects', 'cp'])
        self.assertTrue(calls[1][1].endswith('#456'))
        self.assertFalse(calls[1][2].startswith('gs://'))


if __name__ == '__main__':
    unittest.main()
