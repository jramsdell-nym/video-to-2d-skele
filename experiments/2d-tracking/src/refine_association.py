"""Post-inspection association revision: remove rack bats accepted by the loose gate.
Preserves every raw detection and the original association. Diagnostic tuning, not evaluation.
"""
import shutil
from common import *
for c in load(ROOT/'manifest.json')['clips']:
    folder=ROOT/'outputs'/c['id'];p=folder/'yolo-bat.json';backup=folder/'yolo-bat-loose-gate.json'
    if not backup.exists():shutil.copy2(p,backup)
    raw=load(backup);poses=load(folder/'yolo-pose.json')['frames'];last=None
    for f,pose in zip(raw['frames'],poses):
        hands=[xy(p) for n,p in pose['points'].items() if n.endswith('wrist') and visible(p)]
        choices=[]
        for detection in f['rawCandidates']:
            b=np.array(detection['bbox']);score=detection['confidence']
            if score<.25 or not hands:continue
            distance=min(np.linalg.norm((np.maximum(b[:2]-h,0)+np.minimum(b[2:]-h,0))*[960,540]) for h in hands)
            if distance>50:continue
            association=score-.15*distance/50+(.1*iou(last,b) if last is not None else 0)
            choices.append((association,{**detection,'nearestWristDistanceAt960px':float(distance)}))
        choices.sort(key=lambda x:-x[0]);f['detection']=choices[0][1] if choices else None
        f['status']='observed_bbox' if choices else 'missing'
        if choices:last=np.array(f['detection']['bbox'])
    raw['method']='YOLO11m COCO class 34; confidence >=0.25; bbox within 50px of wrist at 960x540; temporal IoU association; boxes only'
    raw['revision']='After visual inspection, replaced loose normalized-distance gate with pixel-distance gate to reject rack bats. Same four clips; not held-out.'
    dump(p,raw)
    print(c['id'],sum(f['detection'] is not None for f in raw['frames']))
