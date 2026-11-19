"""Small invariant tests: source identity and unsafe-label exclusion."""
import copy
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import snapshot


def fixture():
    point = {'x': .4, 'y': .5, 'visibility': 2, 'source': 'manual'}
    record = {'schema': 1, 'id': 'private-owner:'+'a'*64+':0', 'owner': 'private-owner', 'videoSHA256': 'a'*64, 'frameIndex': 0,
              'frame': {'pts': 0, 'timeBase': '1/60', 'time': 0, 'index': 0},
              'width': 1920, 'height': 1080, 'session': 'session', 'trial': 'trial',
              'source': {'uri': 'gs://example/source.mp4', 'generation': '1', 'source': 'Strike Mobile'},
              'status': 'reviewed', 'reviewedAt': '2026-11-18',
              'points': [copy.deepcopy(point) for _ in range(17)],
              'box': [.2, .2, .8, .8], 'batTip': copy.deepcopy(point),
              'batKnob': {**point, 'reviewedAt': '2026-11-18'}}
    row = {'id': 'assignment-1', 'source': 'Strike Mobile', 'session': 'session', 'trial': 'trial',
           'uri': 'gs://example/source.mp4', 'generation': '1', 'bucket': 'example', 'objectName': 'source.mp4'}
    doc = {'version': 1, 'id': 'assignment-1', 'status': 'completed', 'owner': 'private-owner', 'finishMutation': 'finish-1',
           'row': row, 'clip': {'sha256': 'a'*64, 'width': 1920, 'height': 1080, 'frames': [record['frame']]},
           'poseSubmission': {'version': 1, 'records': [record], 'mutationId': 'finish-1', 'annotator': {'id': 'private-owner'},
                              'counts': {'reviewed': 1, 'drafts': 0, 'smoothedApproved': 0}}}
    queue = {'version': 1, 'tasks': {'assignment-1': {'id': 'assignment-1', 'status': 'completed', 'owner': 'private-owner',
                                                    'finishMutation': 'finish-1', 'submissionGeneration': '2', 'stats': {'frames': 1}}}, 'reviews': {}}
    return record, row, doc, queue


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_root, snapshot.ROOT = snapshot.ROOT, Path(self.temp.name)

    def tearDown(self):
        snapshot.ROOT = self.old_root
        self.temp.cleanup()

    def run_audit(self, modify=lambda r, row, doc, queue: None):
        record, row, doc, queue = fixture()
        modify(record, row, doc, queue)
        for path, value in [(snapshot.ROOT/'private/queue.json', queue),
                            (snapshot.ROOT/'private/catalog.json', {'videos': [row]}),
                            (snapshot.ROOT/'private/assignments/assignment-1.json', doc)]:
            snapshot.save(path, value)
            snapshot.save(path.with_suffix('.json.metadata.json'), {'generation': '2', 'sha256': snapshot.sha(path)})
        with contextlib.redirect_stdout(io.StringIO()): snapshot.audit()
        return snapshot.read(snapshot.ROOT/'inventory.json')

    def test_valid_record_has_twelve_body_and_ordered_endpoints(self):
        report = self.run_audit()
        self.assertEqual((report['currentBodyFrames'], report['currentBatFrames']), (1, 1))
        body = snapshot.read(snapshot.ROOT/'body-rows-unresolved.json')[0]
        self.assertEqual(len(body['points']), 12)
        self.assertNotIn('owner', body)
        self.assertFalse((snapshot.ROOT/'body-rows.json').exists())

    def test_missing_knob_acceptance_excludes_bat_only(self):
        report = self.run_audit(lambda r, *_: r['batKnob'].pop('reviewedAt'))
        self.assertEqual((report['currentBodyFrames'], report['currentBatFrames']), (1, 0))

    def test_timestamp_mismatch_excludes_record(self):
        with self.assertRaisesRegex(ValueError, 'Inconsistent presentation timestamp'):
            self.run_audit(lambda r, *_: r['frame'].update(time=.1))

    def test_rejected_camera_group_excludes_assignment(self):
        def modify(r, row, doc, queue):
            row['groupId'] = 'bad-camera'
            queue['reviews']['group:bad-camera'] = {'view': 'reject'}
        report = self.run_audit(modify)
        self.assertEqual(report['currentBodyFrames'], 0)

    def test_conflicting_duplicate_fails(self):
        def modify(r, row, doc, queue):
            second = copy.deepcopy(r)
            second['points'][5]['x'] = .6
            doc['poseSubmission']['records'].append(second)
        with self.assertRaisesRegex(ValueError, 'Invalid pose submission record count'):
            self.run_audit(modify)

    def test_legacy_is_counted_separately(self):
        def modify(r, row, doc, queue):
            doc.pop('poseSubmission')
            doc['row'] = {}
            doc['clip']['frames'] = [{'index': i, 'pts': i*5, 'timeBase': '1/150', 'time': i/30} for i in range(24)]
            doc['labels'] = {str(i): {'status': 'reviewed', 'frameIndex': i, 'source': frame,
                'videoSHA256': 'a'*64, 'imageWidth': 1920, 'imageHeight': 1080,
                'sourceProvenance': {'source': 'TrackMan', 'uri': 'gs://example/legacy.mov'},
                'points': {'tip': {'state': 'visible', 'x': .4, 'y': .5, 'pixelX': 768., 'pixelY': 540.},
                           'knob': {'state': 'occluded', 'x': None, 'y': None}}}
                for i, frame in enumerate(doc['clip']['frames'])}
            queue['tasks']['assignment-1']['stats']['frames'] = 24
        report = self.run_audit(modify)
        self.assertEqual(report['legacyOnlyFrames'], 24)
        self.assertEqual(report['currentBodyFrames'], 0)
        self.assertIn('legacy-source-not-eligible', report['exclusionReasons'])

    def test_queue_count_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, 'Queue frame counts disagree'):
            self.run_audit(lambda r, row, doc, q: q['tasks']['assignment-1']['stats'].update(frames=20))

    def test_pose_count_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, 'Pose submission counts disagree'):
            self.run_audit(lambda r, row, doc, q: doc['poseSubmission']['counts'].update(reviewed=20))

    def test_owner_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, 'owners disagree'):
            self.run_audit(lambda r, row, doc, q: doc.update(owner='another-owner'))

    def test_revision_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, 'completion revisions disagree'):
            self.run_audit(lambda r, row, doc, q: doc.update(finishMutation='later-finish'))

    def test_generation_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, 'submission generations disagree'):
            self.run_audit(lambda r, row, doc, q: q['tasks']['assignment-1'].update(submissionGeneration='3'))

    def test_record_source_mismatch_is_excluded(self):
        report = self.run_audit(lambda r, *_: r['source'].update(uri='gs://example/different.mp4'))
        self.assertEqual(report['currentBodyFrames'], 0)
        self.assertIn('Record source differs from original assignment source', report['exclusionReasons'])

    def test_dimension_mismatch_is_excluded(self):
        report = self.run_audit(lambda r, *_: r.update(width=1280))
        self.assertEqual(report['currentBodyFrames'], 0)

    def test_480_frame_cam1_non_sixty_fps(self):
        def modify(r, row, doc, queue):
            row['camera'] = r['camera'] = 'cam1'
            records = []
            for i in range(480):
                frame = copy.deepcopy(r)
                frame['frameIndex'] = i
                frame['id'] = 'private-owner:'+'a'*64+':'+str(i)
                frame['frame'] = {'index': i, 'pts': 1001*i, 'timeBase': '1/24000', 'time': 1001*i/24000}
                records.append(frame)
            doc['clip']['frames'] = [x['frame'] for x in records]
            doc['poseSubmission']['records'] = records
            doc['poseSubmission']['counts']['reviewed'] = 480
            queue['tasks']['assignment-1']['stats']['frames'] = 480
        report = self.run_audit(modify)
        self.assertEqual((report['currentBodyFrames'], report['currentBatFrames']), (480, 480))
        self.assertEqual(snapshot.read(snapshot.ROOT/'body-rows-unresolved.json')[-1]['camera'], 'cam1')

    def test_null_predictions_are_accepted(self):
        report = self.run_audit(lambda r, *_: r.update(prediction=None, batPrediction=None))
        self.assertEqual(report['currentBodyFrames'], 1)

    def test_identical_targets_across_assignments_deduplicate(self):
        original = {'videoSHA256': 'a'*64, 'frameIndex': 0, 'points': [[.4, .5]], 'assignmentId': 'first'}
        duplicate = {**original, 'assignmentId': 'second'}
        destination = {}
        snapshot.insert_unique(destination, original, 'body')
        snapshot.insert_unique(destination, duplicate, 'body')
        self.assertEqual(len(destination), 1)
        self.assertEqual(original['duplicateReviews'][0]['assignmentId'], 'second')

    def test_conflicting_targets_across_assignments_fail(self):
        original = {'videoSHA256': 'a'*64, 'frameIndex': 0, 'points': [[.4, .5]], 'assignmentId': 'first'}
        duplicate = {**original, 'points': [[.6, .5]], 'assignmentId': 'second'}
        destination = {}
        snapshot.insert_unique(destination, original, 'body')
        with self.assertRaisesRegex(ValueError, 'Conflicting duplicate body'):
            snapshot.insert_unique(destination, duplicate, 'body')


if __name__ == '__main__': unittest.main()
