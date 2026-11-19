import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile


SPEC=importlib.util.spec_from_file_location('package_training_run',Path(__file__).resolve().parents[1]/'tools/package_training_run.py')
PACKAGE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)


def summary():
    return {'baselineRelease':'2026-11-18',
            'training':{kind:{'epochs':20,'imageSize':640 if kind=='body' else 768,
                'learningRate':.00005,'seed':42,'freeze':10,'trainFrames':1560,
                'trainVideos':10,'valFrames':241,'valVideos':2} for kind in ('body','bat')},
            'evaluation':{'frames':601,'videos':5,'sessions':3,
                **{name:{'baseline':{'correct':70,'total':100},'candidate':{'correct':68,'total':100}}
                   for name in ('bodyJoints','wrists','batPairs')}}}


class PackageTests(unittest.TestCase):
    def test_tensor_records_actual_rounded_resize_scales(self):
        import numpy as np
        tensor,gains=PACKAGE.prepare_tensor(np.zeros((359,641,3),dtype=np.uint8),640)
        self.assertEqual(tensor.shape,(1,3,640,640))
        self.assertEqual(gains,(640/641,358/359))
        self.assertNotEqual(gains[0],gains[1])

    def test_summary_allows_regression_and_rejects_private_extra_fields(self):
        valid=summary()
        self.assertEqual(PACKAGE.validate_summary(valid),valid)
        valid['playerEmail']='private@example.com'
        with self.assertRaises(ValueError): PACKAGE.validate_summary(valid)

    def test_summary_rejects_nonfinite_or_mismatched_denominator(self):
        bad=summary()
        bad['training']['body']['learningRate']=float('nan')
        with self.assertRaises(ValueError): PACKAGE.validate_summary(bad)
        bad=summary()
        bad['evaluation']['bodyJoints']['candidate']['total']=101
        with self.assertRaises(ValueError): PACKAGE.validate_summary(bad)

    def test_metadata_rejects_paths_accounts_and_cloud_uris(self):
        for secret in ('/Users/someone/private.mp4','/home/user/labels.json',
                       's3://private/input','gs://private/labels','name@example.com',
                       'C:\\Users\\someone\\input.mp4'):
            with self.subTest(secret=secret),self.assertRaises(ValueError):
                PACKAGE.reject_private_strings({'nested':[secret]})
        PACKAGE.reject_private_strings({'shape':[1,3,640,640],'task':'pose'})
        PACKAGE.reject_private_strings({'docs':'https://docs.ultralytics.com'})

    def test_zip_rejects_parent_traversal_and_symlinks(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            for index,member in enumerate(('../leak.txt','/absolute.txt','symlink')):
                path=root/f'{index}.zip'
                with zipfile.ZipFile(path,'w') as archive:
                    info=zipfile.ZipInfo(member)
                    if member=='symlink': info.external_attr=0o120777<<16
                    archive.writestr(info,'bad')
                with self.assertRaises(ValueError): PACKAGE.safe_extract(path,root/'out')

    def test_checksums_detect_changed_file_in_extracted_package(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            (root/'data.txt').write_text('expected')
            (root/'verify_checksums.py').write_text((PACKAGE.TEMPLATES/'verify_checksums.py').read_text())
            PACKAGE.write_checksums(root)
            result=subprocess.run([sys.executable,str(root/'verify_checksums.py')],capture_output=True)
            self.assertEqual(result.returncode,0)
            (root/'data.txt').write_text('changed')
            result=subprocess.run([sys.executable,str(root/'verify_checksums.py')],capture_output=True)
            self.assertNotEqual(result.returncode,0)

    def test_readme_preserves_negative_delta_and_selection_caveat(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            PACKAGE.write_readme(root,{'release':'2026-11-19','summary':summary()}, {})
            text=(root/'README.md').read_text()
            self.assertIn('-2.00 pp',text)
            self.assertIn('highest-confidence-person selection',text)
            self.assertIn('exact completed training settings',text)
            self.assertNotIn('A new bucket snapshot was not available',text)
            self.assertIn('five face slots',text)
            self.assertIn('50% probability of horizontal reflection',text)
            self.assertIn('BAT tip and knob keep their physical identities',text)
            self.assertIn('does not automatically mirror',text)
            self.assertNotIn('$results_table',text)

    def test_round_two_readme_compares_with_round_one(self):
        with tempfile.TemporaryDirectory() as name:
            data = summary()
            data['baselineRelease'] = '2026-11-19'
            PACKAGE.validate_summary(data)
            root = Path(name)
            PACKAGE.write_readme(root, {'release':'2026-11-19-r2', 'summary':data}, {})
            self.assertIn('| Metric | 2026-11-19 | 2026-11-19-r2 |', (root/'README.md').read_text())
            data['baselineRelease'] = '../../outside'
            with self.assertRaisesRegex(ValueError, 'Invalid baseline release'):
                PACKAGE.validate_summary(data)


if __name__=='__main__':
    unittest.main()
