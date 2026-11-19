"""Automatic hand/motion-constrained LSD segments + short, gated Lucas-Kanade propagation.
Segments are image evidence candidates, NOT labeled physical tip/knob endpoints.
"""
import time
import av,cv2,numpy as np
from common import *
cv2.setNumThreads(2)
lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
def samples(seg,n=25):return np.linspace(seg[0],seg[1],n).astype(np.float32)
def support(mask,seg):
    p=samples(seg).round().astype(int);p[:,0]=p[:,0].clip(0,mask.shape[1]-1);p[:,1]=p[:,1].clip(0,mask.shape[0]-1)
    return float((mask[p[:,1],p[:,0]]>0).mean())
def hand_distance(seg,hands):return min(np.linalg.norm(p-h) for p in seg for h in hands)
for c in load(ROOT/'manifest.json')['clips']:
    clip=c['id'];poses=select_vision(clip);frames=[];times=[];previous=None;lastseg=None;age=0
    with av.open(str(ROOT/c['source'])) as container:
        for i,f in enumerate(container.decode(video=0)):
            im=f.to_ndarray(format='bgr24');h,w=im.shape[:2];im=cv2.resize(im,(960,round(h*960/w)))
            gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);start=time.perf_counter();H,W=gray.shape
            hands=[xy(p)*[W,H] for n,p in poses[i]['points'].items() if n.endswith('wrist') and visible(p)]
            motion=None if previous is None else cv2.dilate((cv2.absdiff(gray,previous)>15).astype(np.uint8),np.ones((5,5),np.uint8))
            edge=cv2.dilate(cv2.Canny(gray,60,160),np.ones((5,5),np.uint8))
            candidates=[];det=lsd.detect(gray)[0]
            if det is not None and hands:
                for item in det:
                    seg=item.reshape(2,2);length=np.linalg.norm(seg[1]-seg[0]);distance=hand_distance(seg,hands)
                    if not 35<length<210 or distance>42:continue
                    ms=0 if motion is None else support(motion,seg)
                    if ms<.30:continue
                    score=.45*min(length/110,1)+.35*ms+.20*(1-distance/42)
                    candidates.append((score,seg,ms,distance))
            candidates.sort(key=lambda x:-x[0]);chosen=None;status='missing';reason='No supported line near a visible wrist';quality=None
            if candidates and candidates[0][0]>.62:
                score,seg,ms,distance=candidates[0]
                # Near-duplicate parallel edges do not count as ambiguous; distinct directions do.
                direction=(seg[1]-seg[0])/np.linalg.norm(seg[1]-seg[0])
                ambiguous=False
                for score2,seg2,*_ in candidates[1:]:
                    d=(seg2[1]-seg2[0])/np.linalg.norm(seg2[1]-seg2[0])
                    if score-score2<.05 and abs(direction@d)<.85:ambiguous=True
                if not ambiguous:
                    chosen=seg;status='observed_line';quality=float(score);reason='LSD edge segment with motion and hand proximity';age=0
                else:reason='Competing line directions'
            if chosen is None and lastseg is not None and age<2 and hands and previous is not None:
                p=samples(lastseg).reshape(-1,1,2)
                q,st,err=cv2.calcOpticalFlowPyrLK(previous,gray,p,None,winSize=(21,21),maxLevel=3)
                back,bt,_=cv2.calcOpticalFlowPyrLK(gray,previous,q,None,winSize=(21,21),maxLevel=3)
                fb=np.linalg.norm(back-p,axis=2).ravel();good=(st.ravel()>0)&(bt.ravel()>0)&(fb<1.5)&(err.ravel()<25)
                if good.sum()>=15:
                    M,inliers=cv2.estimateAffinePartial2D(p.reshape(-1,2)[good],q.reshape(-1,2)[good],method=cv2.RANSAC,ransacReprojThreshold=2)
                    if M is not None:
                        estimate=lastseg@M[:,:2].T+M[:,2];scale=np.linalg.norm(M[:,0])
                        es=support(edge,estimate)
                        if .85<scale<1.15 and hand_distance(estimate,hands)<50 and es>.72:
                            chosen=estimate;status='propagated_line';quality=float(es);reason='LK + forward/backward + affine + current edge gates';age+=1
            if chosen is not None:lastseg=chosen
            else:lastseg=None;age=0
            frames.append({'time':float(f.pts*f.time_base),'status':status,'segment':None if chosen is None else (chosen/[W,H]).tolist(),'quality':quality,'qualityIsProbability':False,'reason':reason,'tip':None,'knob':None,'candidateCount':len(candidates)})
            times.append(time.perf_counter()-start);previous=gray
    dump(ROOT/'outputs'/clip/'line-bat.json',{'method':'Automatic OpenCV LSD + hand/motion gates; LK propagation at most 2 frames; line endpoints are not bat tip/knob','frames':frames,'meanMilliseconds':1000*np.mean(times),'timingExcludes':'Video decode, resize and upstream Vision body pose'})
    print(clip,{s:sum(f['status']==s for f in frames) for s in ['observed_line','propagated_line','missing']},flush=True)
