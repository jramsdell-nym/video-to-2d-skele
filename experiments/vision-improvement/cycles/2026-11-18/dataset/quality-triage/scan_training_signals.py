"""Rank TRAINING-ONLY label-review leads; numbers are not label-error verdicts."""
from collections import defaultdict
from pathlib import Path
import json
import math
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from snapshot import sha,save

JOINTS=['LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle']

def xy(p,r): return p['x']*r['width'],p['y']*r['height']
def dist(a,b): return math.hypot(a[0]-b[0],a[1]-b[1])

def main():
    body_path,bat_path=ROOT/'splits/body-train-rows.json',ROOT/'splits/bat-train-rows.json'
    body,bat=json.loads(body_path.read_text()),json.loads(bat_path.read_text())
    signals=[];lookup={(r['videoSHA256'],r['frameIndex']):r for r in body}
    def append(kind,value,row,**fields):
        signals.append(dict(kind=kind,value=value,videoSHA256=row['videoSHA256'],frameIndex=row['frameIndex'],trial=row['trial'],**fields))
    for kind,rows in [('body',body),('bat',bat)]:
        groups=defaultdict(list)
        for r in rows: groups[r['videoSHA256']].append(r)
        for h,rs in groups.items():
            rs.sort(key=lambda r:r['frameIndex'])
            if kind=='bat':
                for r in rs:
                    b=lookup[(h,r['frameIndex'])];tip,knob=[xy(r['points'][k],r) for k in ['tip','knob']]
                    length=dist(tip,knob);wrists=[xy(b['points'][j],b) for j in (4,5) if b['points'][j]['visibility']]
                    if length and wrists: append('knob-wrist-distance',min(dist(knob,p) for p in wrists)/length,r,batLengthPixels=length)
            for a,r,z in zip(rs,rs[1:],rs[2:]):
                if (r['frameIndex']-a['frameIndex'],z['frameIndex']-r['frameIndex'])!=(1,1): continue
                f=(r['frame']['time']-a['frame']['time'])/(z['frame']['time']-a['frame']['time'])
                scale=(r['box'][3]-r['box'][1])*r['height'] if kind=='body' else dist(xy(r['points']['tip'],r),xy(r['points']['knob'],r))
                if not scale: continue
                for j in (range(12) if kind=='body' else ['tip','knob']):
                    if any(t['points'][j]['visibility']==0 for t in (a,r,z)): continue
                    p=r['points'][j];ax,ay=xy(a['points'][j],a);zx,zy=xy(z['points'][j],z)
                    value=dist(xy(p,r),(ax+(zx-ax)*f,ay+(zy-ay)*f))/scale
                    append(kind+'-temporal-residual',value,r,joint=JOINTS[j] if kind=='body' else j,visibility=p['visibility'],provenance=p.get('source','unknown'))
    selected=[]
    for kind in sorted({s['kind'] for s in signals}):
        chosen=[]
        for s in sorted((s for s in signals if s['kind']==kind),key=lambda s:s['value'],reverse=True):
            if any(s['videoSHA256']==t['videoSHA256'] and abs(s['frameIndex']-t['frameIndex'])<8 for t in chosen): continue
            chosen.append(s)
            if len(chosen)==12: break
        selected.extend(chosen)
    save(ROOT/'quality-triage/signals.json',{'trainingOnly':True,'notAutomaticErrors':True,'bodyTrainSHA256':sha(body_path),'batTrainSHA256':sha(bat_path),'candidates':selected})
    print(json.dumps({'trainingBodyRows':len(body),'trainingBatRows':len(bat),'reviewLeads':len(selected)}))

if __name__=='__main__': main()
