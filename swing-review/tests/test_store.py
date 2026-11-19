import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from store import Store, Conflict, JOINTS, point


def asset(count=3, reviewed=False):
    sha = 'a' * 64
    frames = [{'pts': i * 512, 'timeBase': '1/15360', 'time': i / 30, 'index': i} for i in range(count)]
    records = []
    for i in range(count):
        points = [{'x': .3 + (j % 2) * .2, 'y': .3 + (j // 2) * .02, 'visibility': 2} for j in range(17)]
        records.append({'id': f'owner:{sha}:{i}', 'owner': 'owner', 'schema': 1, 'videoSHA256': sha,
                        'frameIndex': i, 'frame': frames[i], 'width': 1920, 'height': 1080,
                        'points': points, 'box': [.2, .2, .6, .8], 'status': 'reviewed' if reviewed else 'draft',
                        'batTip': {'x': .6, 'y': .2, 'visibility': 2}, 'batKnob': {'x': .5, 'y': .4, 'visibility': 2}})
    return {'id': 'trial001', 'clip': {'sha256': sha, 'width': 1920, 'height': 1080, 'frames': frames},
            'records': records, 'owner': 'owner', 'video': '/tmp/unused.mp4',
            'row': {'source': 'Strike Mobile', 'uri': 'gs://test', 'generation': '123'},
            'sourceGeneration': '123', 'sourceDigest': 'b' * 64}


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / 'test.sqlite3')
        self.store.add(asset())

    def tearDown(self):
        self.tmp.cleanup()

    def body(self, **extra):
        return {'actor': {'type': 'human', 'id': 'test-user'}, 'requestId': uuid.uuid4().hex,
                'expectedRevision': self.store.get('trial001')['revision'], **extra}

    def mutate(self, kind, **extra):
        return self.store.mutate('trial001', kind, self.body(**extra))

    def change(self, idx=0, x=.52):
        return {'frameIndex': idx, 'joint': 'left_elbow', 'point': {'x': x, 'y': .4, 'visibility': 2}}

    def test_atomic_batch_and_bounds(self):
        before = self.store.get('trial001')
        for bad in [float('nan'), -1, 1.2, True]:
            with self.assertRaises(ValueError):
                self.mutate('batch', changes=[self.change(), self.change(1, bad)])
            self.assertEqual(before, self.store.get('trial001'))

    def test_revision_idempotency_and_baseline(self):
        original = self.store.get('trial001')['frames'][0]['baseline']
        body = self.body(changes=[self.change()])
        first = self.store.mutate('trial001', 'batch', body)
        self.assertEqual(first, self.store.mutate('trial001', 'batch', body))
        modified = copy.deepcopy(body); modified['changes'][0]['point']['x'] = .6
        with self.assertRaises(Conflict):
            self.store.mutate('trial001', 'batch', modified)
        with self.assertRaises(Conflict):
            self.store.mutate('trial001', 'batch', {**body, 'requestId': uuid.uuid4().hex})
        self.assertEqual(original, self.store.get('trial001')['frames'][0]['baseline'])
        self.assertEqual(len(self.store.history('trial001')), 1)

    def test_only_one_concurrent_claim(self):
        def claim(owner):
            try:
                return self.store.claim('trial001', {'owner': owner})
            except Conflict:
                return None
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(claim, ['agent-a', 'agent-b']))
        self.assertEqual(sum(r is not None for r in results), 1)

    def test_agent_lease_and_human_protection(self):
        lease = self.store.claim('trial001', {'owner': 'agent-a'})
        agent_body = self.body(actor={'type': 'agent', 'id': 'agent-a'}, leaseToken=lease['leaseToken'], changes=[self.change()])
        self.store.mutate('trial001', 'batch', agent_body)
        self.mutate('batch', changes=[self.change(x=.53)])
        with self.assertRaises(Conflict):
            self.store.mutate('trial001', 'batch', self.body(actor={'type': 'agent', 'id': 'agent-a'}, leaseToken=lease['leaseToken'], changes=[self.change()]))
        lease = self.store.claim('trial001', {'owner': 'agent-a'})
        with self.assertRaises(Conflict):
            self.store.mutate('trial001', 'batch', self.body(actor={'type': 'agent', 'id': 'agent-a'}, leaseToken=lease['leaseToken'], changes=[self.change()]))

    def test_locks_and_reimport(self):
        self.mutate('lock', frameIndex=0, locked=True)
        before = self.store.get('trial001')
        with self.assertRaises(Conflict):
            self.mutate('batch', changes=[self.change()])
        self.assertFalse(self.store.add(asset()))
        self.assertEqual(before, self.store.get('trial001'))

    def test_approval_passes_and_raw_compatibility(self):
        with self.assertRaises(ValueError):
            self.mutate('approve', confirmation='Approve local draft')
        with self.assertRaises(ValueError):
            self.mutate('review', frameIndex=0, **{'pass': 'replay'})
        for i in range(3):
            self.mutate('review', frameIndex=i, **{'pass': 'placement'})
            self.mutate('review', frameIndex=i, **{'pass': 'replay'})
            for joint in JOINTS:
                self.mutate('review', frameIndex=i, joint=joint, **{'pass': 'joints'})
            self.mutate('review', frameIndex=i, **{'pass': 'final'})
        self.mutate('approve', confirmation='Approve local draft')
        bundle = self.store.export('trial001')
        self.assertFalse(bundle['uploadAuthorized'])
        self.assertEqual(bundle['expectedSourceGeneration'], '123')
        self.assertEqual(bundle['poseSubmission']['counts']['reviewed'], 3)
        self.assertTrue(all(len(r['points']) == 17 and r['batKnob']['reviewedAt'] for r in bundle['poseSubmission']['records']))
        self.assertEqual(self.store.get('trial001')['summary']['changed'], 0)
        with self.assertRaises(Conflict):
            self.mutate('upload')
        self.mutate('batch', changes=[self.change(x=.51)])
        self.assertEqual(self.store.get('trial001')['state'], 'draft')
        self.assertNotIn('replay', self.store.get('trial001')['frames'][1]['meta']['passes'])
        with self.assertRaises(Conflict):
            self.store.export('trial001')

    def test_undo_and_uncertainty(self):
        original = copy.deepcopy(self.store.get('trial001')['frames'][0]['record'])
        self.mutate('batch', changes=[{**self.change(), 'uncertain': True}])
        with self.assertRaises(ValueError):
            self.mutate('review', frameIndex=0, **{'pass': 'placement'})
        self.mutate('undo')
        self.assertEqual(point(original, 'left_elbow'), point(self.store.get('trial001')['frames'][0]['record'], 'left_elbow'))
        self.assertEqual([x['kind'] for x in self.store.history('trial001')], ['undo', 'batch'])
        self.assertEqual(self.store.get('trial001')['frames'][0]['meta']['uncertain'], [])

    def test_undo_with_locked_neighbors(self):
        self.mutate('lock', frameIndex=1, locked=True)
        self.mutate('batch', changes=[self.change()])
        self.mutate('undo')
        self.assertTrue(self.store.get('trial001')['frames'][1]['meta']['locked'])

    def test_legacy_record_owner_is_preserved(self):
        a=asset(); a['id']='legacy01'; a['owner']='migrated-account'
        self.store.add(a)
        self.assertEqual(self.store.get('legacy01')['frames'][0]['record']['owner'], 'owner')

    def test_uncertainty_on_locked_reference_invalidates_pass(self):
        self.mutate('review', frameIndex=0, **{'pass':'placement'})
        self.mutate('lock', frameIndex=0, locked=True)
        before=self.store.get('trial001')['frames'][0]['record']['points']
        self.mutate('flag', frameIndex=0, joint='left_elbow', uncertain=True)
        f=self.store.get('trial001')['frames'][0]
        self.assertEqual(f['record']['points'],before)
        self.assertFalse(f['meta']['passes'])
        self.assertTrue(f['meta']['locked'])

    def test_setup_interpolation_preserves_endpoints(self):
        self.store = Store(Path(self.tmp.name) / 'interpolation.sqlite3')
        self.store.add(asset(12))
        self.mutate('settings', swingStart=11, swingEnd=11)
        self.mutate('lock', frameIndex=0, locked=True)
        self.mutate('lock', frameIndex=10, locked=True)
        self.mutate('batch', changes=[self.change(4)])
        before = self.store.get('trial001')
        self.mutate('interpolate', start=0, end=10, joint='left_elbow')
        after = self.store.get('trial001')
        for i in (0,4,10):
            self.assertEqual(before['frames'][i]['record'], after['frames'][i]['record'])
        self.assertEqual(point(after['frames'][5]['record'], 'left_elbow')['source'], 'interpolated')
        self.assertFalse(after['frames'][5]['meta']['passes'])
        with self.assertRaises(ValueError):
            self.mutate('interpolate', start=1, end=11, joint='left_elbow')

    def test_reject_reduced_export_and_cross_trial_identity(self):
        for key,value in [('points', [{}]*12), ('videoSHA256', 'wrong')]:
            a=asset();a['id']='badtrial';a['records'][0][key]=value
            with self.assertRaises(ValueError):
                self.store.add(a)
            self.assertEqual(len(self.store.list()), 1)

    def test_persistence_after_reopening(self):
        self.mutate('batch', changes=[self.change()])
        reopened=Store(self.store.path)
        self.assertEqual(self.store.get('trial001'),reopened.get('trial001'))

    def stage_report(self, changes=None):
        self.mutate('lock', frameIndex=0, locked=True)
        lease=self.store.claim('trial001', {'owner':'agent-a'})
        report={'trialId':'trial001', 'sourceRevision':self.store.get('trial001')['revision'],
                'passes':[{'name':'Placement', 'frames':[0,1,2]}],
                'changes':changes if changes is not None else [self.change()]}
        before=self.store.get('trial001')
        body=self.body(actor={'type':'agent','id':'agent-a'},leaseToken=lease['leaseToken'],report=report)
        self.store.mutate('trial001','agent-report',body)
        after=self.store.get('trial001')
        self.assertEqual(before['frames'],after['frames'])
        self.assertEqual(after['state'],'review')
        self.assertIsNone(after['lease']['owner'])
        return after['agentReports'][0]['runId']

    def test_report_accept_undo_and_reject_preserve_reference(self):
        run=self.stage_report()
        before=self.store.get('trial001')['frames'][0]
        with self.assertRaisesRegex(ValueError,'suggestions'):
            self.mutate('review',frameIndex=0,**{'pass':'placement'})
        self.mutate('accept-proposal',runId=run,frameIndex=0,joint='left_elbow')
        after=self.store.get('trial001')['frames'][0]
        self.assertTrue(after['meta']['locked'])
        self.assertTrue(after['meta']['humanEdited'])
        self.assertEqual(after['baseline'],before['baseline'])
        self.assertEqual(point(after['record'],'left_elbow')['x'],.52)
        self.assertEqual(self.store.list()[0]['agentReports']['pending'],0)
        self.mutate('undo')
        self.assertEqual(point(self.store.get('trial001')['frames'][0]['record'],'left_elbow'),point(before['record'],'left_elbow'))
        self.assertEqual(self.store.list()[0]['agentReports']['pending'],1)
        self.mutate('reject-proposal',runId=run,frameIndex=0,allFrame=True)
        self.assertEqual(self.store.list()[0]['agentReports']['pending'],0)

    def test_proposal_cannot_overwrite_newer_human_edit(self):
        run=self.stage_report()
        self.mutate('lock',frameIndex=0,locked=False)
        self.mutate('batch',changes=[self.change(x=.7)])
        before=self.store.get('trial001')
        with self.assertRaises(Conflict):
            self.mutate('accept-proposal',runId=run,frameIndex=0,allFrame=True)
        self.assertEqual(before,self.store.get('trial001'))

    def test_agent_reports_are_atomic_and_do_not_approve(self):
        lease=self.store.claim('trial001', {'owner':'agent-a'})
        before=self.store.get('trial001')
        for changes,rev,error in [([self.change()],9,Conflict),([self.change(),self.change()],0,ValueError)]:
            report={'trialId':'trial001','sourceRevision':rev,'passes':[],'changes':changes}
            with self.assertRaises(error):
                self.store.mutate('trial001','agent-report',self.body(actor={'type':'agent','id':'agent-a'},leaseToken=lease['leaseToken'],report=report))
            self.assertEqual(before,self.store.get('trial001'))
        with self.assertRaises(ValueError):
            self.store.mutate('trial001','accept-proposal',self.body(actor={'type':'agent','id':'agent-a'},leaseToken=lease['leaseToken']))
        self.store.mutate('trial001','agent-report',self.body(actor={'type':'agent','id':'agent-a'},leaseToken=lease['leaseToken'],report={'trialId':'trial001','passes':[],'changes':[]}))
        self.assertEqual(self.store.get('trial001')['summary']['remaining']['placement'],3)


if __name__ == '__main__':
    unittest.main()
