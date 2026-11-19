from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest

from build_gallery import build, delta, fraction
from render_comparisons import dump, sha256


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items = []

    def handle_starttag(self, tag, attrs):
        self.items.append((tag, dict(attrs)))


class GalleryChecks(unittest.TestCase):
    def fixture(self, directory):
        exports = directory / 'exports'
        exports.mkdir()
        clips, results = [], []
        for index in range(5):
            video, poster = exports/f'clip {index}.mp4',exports/f'poster {index}.jpg'
            video.write_bytes(b'test asset, not a real video')
            poster.write_bytes(b'test asset, not a real image')
            slug, source_sha = f'swing-{index}',str(index)*64
            clips.append({'slug':slug,'title':'Synthetic <markup> & test',
                          'source':{'sha256':source_sha},'output':str(video),'outputSHA256':sha256(video),
                          'validation':{'allFramesDecoded':True,'allTimestampsVerified':True,
                                        'frameCount':120,'durationSeconds':8,
                                        'posters':[{'frameIndex':59,'path':str(poster),'sha256':sha256(poster)}]}})
            results.append({'slug':slug,'videoSHA256':source_sha,'sourceFrames':120,
                            'body':{'generic':.5,'candidate':.6},'bat':{'current':.7,'candidate':.65}})
        manifest, result = exports/'export-manifest.json',directory/'results.json'
        dump(manifest,{'clips':clips})
        split={k:{s:{'frames':n,'videos':v} for s,n,v in [('train',100,2),('val',120,1),('evaluation',600,5)]} for k in ('body','bat')}
        dump(result,{'perVideo':results,'dataset':{'priorBodyFrames':120,'currentBodyFrames':820,
                                                 'priorBatFrames':120,'currentBatFrames':820},
                     'split':split,'body':{'generic':{'bodyPCK':.5,'wristsPCK':.4},'previous':{'bodyPCK':.58,'wristsPCK':.35},'candidate':{'bodyPCK':.6,'wristsPCK':.3}},
                     'bat':{'current':{'pairedEndpointsPCK':.7},'candidate':{'pairedEndpointsPCK':.65}},
                     'limitations':['Synthetic fixture only.']})
        return manifest,result

    def test_exactly_five_manual_playback_videos_and_escaped_text(self):
        with tempfile.TemporaryDirectory(prefix='gallery-check-',dir=Path(__file__).parent) as temp:
            directory=Path(temp).resolve()
            manifest,result=self.fixture(directory)
            output=directory/'index.html'
            receipt=build(manifest,result,output)
            page=output.read_text()
            parser=Tags();parser.feed(page)
            videos=[attrs for tag,attrs in parser.items if tag=='video']
            self.assertEqual(len(videos),5)
            self.assertTrue(all('controls' in attrs and 'autoplay' not in attrs for attrs in videos))
            self.assertFalse(any(tag=='script' for tag,_ in parser.items))
            self.assertIn('Synthetic &lt;markup&gt; &amp; test',page)
            self.assertIn('exports/clip%200.mp4',page)
            self.assertIn('+10.0 percentage points',page)
            self.assertIn('-10.0 percentage points',page)
            self.assertIn('Prior candidate · November 11',page)
            self.assertEqual(receipt['clipCount'],5)

    def test_rejects_changed_media_and_mismatched_results(self):
        with tempfile.TemporaryDirectory(prefix='gallery-check-',dir=Path(__file__).parent) as temp:
            directory=Path(temp).resolve()
            manifest,result=self.fixture(directory)
            data=json.loads(result.read_text())
            data['perVideo'][0]['videoSHA256']='mismatch'
            dump(result,data)
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                build(manifest,result,directory/'index.html')
            data['perVideo'][0]['videoSHA256']='0'*64
            dump(result,data)
            (directory/'exports/clip 0.mp4').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed after validation'):
                build(manifest,result,directory/'index.html')

    def test_metric_validation_does_not_invent_missing_scores(self):
        self.assertIsNone(fraction(None))
        self.assertIn('unavailable',delta(None,.5))
        with self.assertRaises(ValueError):
            fraction(float('nan'))

    def test_handedness_summary_keeps_unknown_and_small_sample_caveat(self):
        with tempfile.TemporaryDirectory(prefix='gallery-check-',dir=Path(__file__).parent) as temp:
            directory=Path(temp).resolve()
            manifest,result=self.fixture(directory)
            dataset=directory/'dataset'
            (dataset/'postevaluation-body-review').mkdir(parents=True)
            (dataset/'HANDEDNESS_AUDIT.txt').write_text('Synthetic audit fixture')
            (dataset/'horizontal-flip-check.png').write_bytes(b'Synthetic image fixture')
            (dataset/'postevaluation-body-review/REVIEW.txt').write_text('Synthetic review fixture')
            slices={side:{'frames':frames,'videos':videos,'jointPCK0.05BodyScale':.5,
                          'wrists':{'jointPCK0.05BodyScale':.4},'bothEndpointsPCK0.1BatLength':.6}
                    for side,frames,videos in [('L',120,1),('R',241,2),('unknown',240,2)]}
            report={'schema':'catalog-handedness-slices-v1','tasks':{
                task:{'models':{model:{'slices':slices} for model in models}}
                for task,models in [('body',('previous','candidate')),('bat',('current','candidate'))]}}
            training=directory/'training';training.mkdir()
            dump(training/'handedness-scores.json',report)
            output=directory/'index.html';build(manifest,result,output)
            page=output.read_text()
            self.assertIn('Body: prior → new',page)
            self.assertIn('Unknown</th><td>2 / 240',page)
            self.assertIn('These slices do not establish broad performance',page)
            self.assertIn('all 17 lack a verified saved camera-view label',page)


if __name__=='__main__':
    unittest.main()
