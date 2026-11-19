import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
COCO=['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
MAJOR=COCO[5:]
BONES=[('left_shoulder','right_shoulder'),('left_shoulder','left_hip'),('right_shoulder','right_hip'),('left_hip','right_hip'),('left_shoulder','left_elbow'),('left_elbow','left_wrist'),('right_shoulder','right_elbow'),('right_elbow','right_wrist'),('left_hip','left_knee'),('left_knee','left_ankle'),('right_hip','right_knee'),('right_knee','right_ankle')]
def load(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2)+'\n')
def visible(p):return bool(p and p['confidence']>=.3 and 0<=p['x']<=1 and 0<=p['y']<=1)
def xy(p):return np.array([p['x'],p['y']])
def bbox(points):
    ps=[xy(p) for p in points.values() if visible(p)]
    if len(ps)<5:return None
    ps=np.array(ps);return np.r_[ps.min(0),ps.max(0)]
def iou(a,b):
    inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(inter/(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter+1e-9))
class HitterSelector:
    """Largest visible person initialization, then gated bbox/center association. No imputation."""
    def __init__(self):self.last=None
    def choose(self,poses):
        candidates=[]
        for i,points in enumerate(poses):
            b=bbox(points)
            if b is None:continue
            area=np.prod(b[2:]-b[:2]);center=(b[:2]+b[2:])/2
            if area<.012:continue
            if self.last is None:score=area
            else:
                a=self.last;dist=np.linalg.norm(center-(a[:2]+a[2:])/2)
                ratio=area/max(1e-6,np.prod(a[2:]-a[:2]))
                if dist>.22 or not .30<ratio<3.3:continue
                score=2*iou(a,b)-dist+.1*min(ratio,1/ratio)
            candidates.append((float(score),i,b))
        candidates.sort(key=lambda x:-x[0])
        if not candidates:return None,'missing'
        if self.last is not None and len(candidates)>1 and candidates[0][0]-candidates[1][0]<.08:return None,'ambiguous'
        _,i,self.last=candidates[0]
        return i,'observed'
def select_vision(clip):
    allv=load(ROOT/'data'/clip/'vision-all.json');selector=HitterSelector();frames=[]
    for f in allv['frames']:
        i,status=selector.choose(f['candidates'])
        frames.append({'time':f['time'],'points':f['candidates'][i] if i is not None else {},'people':len(f['candidates']),'selectedIndex':i,'status':status,'trackId':'H1' if i is not None else None})
    dump(ROOT/'outputs'/clip/'vision-selected.json',{'method':'Apple Vision + automatic largest-person initialization and gated temporal association','frames':frames,'processingSeconds':allv['processingSeconds']})
    return frames
