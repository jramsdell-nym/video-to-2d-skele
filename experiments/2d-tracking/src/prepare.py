"""Copy a fixed, small cached cohort; never modify upstream assets."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/Users/jramsdell/Documents/bitbucket/fan engagement/experiments/vision-kinatrax')
dataset=json.loads((OLD/'runs/pilot-v3/dataset.json').read_text())['trials']
specs=[('elian','markers_2026-10-30_12-19-45','L','Clean left-handed BP'),
       ('trey','markers_2026-10-30_11-47-18','R','Right-handed full swing'),
       ('occlusion','markers_2026-10-30_11-51-24','R','Background people and hood/head occlusion')]
clips=[]
for short,trial,side,case in specs:
    t=next(t for t in dataset if t['id']==trial); folder=OLD/t['folder']
    clips.append(dict(id=short,trial=trial,session=t['session'],athlete=t['athleteName'],side=side,sideSource='stored trial metadata',case=case,source=folder/('cam5.mp4' if short=='elian' else 'camera.mp4'),baseline=folder/'vision.json'))
clips.append(dict(id='seed',trial='markers_2026-11-06_11-18-35',session='session_2026-11-06_12-51',athlete='November 6 seed (identity not verified)',side='L',sideSource='visual batting stance only',case='Exact requested November 6 seed',source=OLD/'data/reviewed/seed/cam5.mp4',baseline=OLD/'data/reviewed/seed/vision.json'))
for c in clips:
    dest=ROOT/'data'/c['id'];dest.mkdir(exist_ok=True)
    for key,name in [('source','source.mp4'),('baseline','vision-original.json')]:
        p=c[key];c[key+'OriginalPath']=str(p);c[key+'SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
        shutil.copy2(p,dest/name);c[key]=str((dest/name).relative_to(ROOT))
    c['camera']='cam5';c['orientation']='Original, unmirrored';c['angleReview']='Visually verified full-body open-side view from cached thumbnails; inspect motion in gallery'
(ROOT/'manifest.json').write_text(json.dumps({'schema':1,'created':'2026-11-08','cohortSelection':'Purposive diagnostic sample, not held-out benchmark','clips':clips},indent=2)+'\n')
build=ROOT/'src/vision';build.mkdir(exist_ok=True)
for f in ['PoseProcessor.swift','PoseModels.swift','Orientation.swift']:
    shutil.copy2(OLD/'data/vision-runtime'/f,build/f)
(ROOT/'provenance/apple-source.json').write_text(json.dumps({f:hashlib.sha256((build/f).read_bytes()).hexdigest() for f in ['PoseProcessor.swift','PoseModels.swift','Orientation.swift']},indent=2))
