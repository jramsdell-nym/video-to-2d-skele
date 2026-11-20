import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import LocalVideoLibrary, ReadOnlyBackend, digest
from store import Store


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

    def test_status_without_cloud_snapshot(self):
        empty = ReadOnlyBackend(self.root / 'fresh', self.root / 'empty')
        self.assertIsNone(empty.status()['retrievedAt'])
        self.assertFalse(empty.status()['uploadEnabled'])


class LocalVideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.videos = self.root / 'clips'
        self.videos.mkdir()
        self.library = LocalVideoLibrary(self.videos, self.root / 'data')

    def test_native_frames_and_idempotent_draft(self):
        import av
        from PIL import Image

        group = self.videos / 'Scout swings'
        group.mkdir()
        video = group / 'recording.MOV'
        with av.open(str(video), mode='w') as output:
            stream = output.add_stream('mpeg4', rate=24)
            stream.width, stream.height = 96, 64
            stream.pix_fmt = 'yuv420p'
            for pts in (0, 1, 4):
                frame = av.VideoFrame.from_image(Image.new('RGB', (96, 64)))
                frame.pts = pts
                output.mux(stream.encode(frame))
            output.mux(stream.encode())
        video.with_suffix('.json').write_text('{"private":"unused"}')
        (group / 'alias.mov').symlink_to(video)
        [row] = self.library.catalog()
        self.assertEqual(row['session'], 'Scout swings')
        self.assertNotIn(str(self.videos), json.dumps(row))
        asset = self.library.load(row['id'])
        self.assertEqual(asset['records'], [])
        self.assertEqual(asset['clip']['sha256'], digest(video))
        frames = asset['clip']['frames']
        self.assertEqual([f['index'] for f in frames], [0, 1, 2])
        self.assertAlmostEqual(frames[-1]['time'], 4 / 24)

        store = Store(self.root / 'data' / 'drafts.sqlite3')
        self.assertTrue(store.add(asset))
        self.assertFalse(store.add(self.library.load(row['id'])))
        video.unlink()
        self.assertEqual(self.library.catalog(), [])
        self.assertEqual(digest(asset['video']), asset['clip']['sha256'])
        self.assertEqual(store.get(row['id'])['state'], 'draft')
        self.assertEqual(store.get(row['id'])['frames'][0]['record']['status'], 'draft')

    def test_corrupt_media_and_unknown_ids_fail_without_leaving_a_copy(self):
        source = self.videos / 'broken.mp4'
        source.write_bytes(b'invalid media')
        [row] = self.library.catalog()
        with self.assertRaisesRegex(ValueError, 'Could not decode'):
            self.library.load(row['id'])
        self.assertEqual(list(self.library.cache.iterdir()), [])
        with self.assertRaisesRegex(ValueError, 'missing or changed'):
            self.library.load('local-nonexistent')


if __name__ == '__main__':
    unittest.main()
