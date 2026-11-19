"""Entirely local pretrained inference. Source decode PTS and missing observations retained."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'provenance/yolo-config')
os.environ['MPLCONFIGDIR']=str(ROOT/'provenance/mpl-config')
os.environ['YOLO_OFFLINE']='true'
import time,platform,hashlib,subprocess
import av,cv2,torch,numpy as np
from ultralytics import YOLO,settings
from common import *
settings.update({'sync':False})
torch.set_num_threads(4)
cv2.setNumThreads(2)
device=os.environ.get('TRACK_DEVICE','cpu')
def sync():
    if device=='mps':torch.mps.synchronize()
pose=YOLO(str(ROOT/'models/yolo11s-pose.pt'))
bat=YOLO(str(ROOT/'models/yolo11m.pt'))
assert bat.names[34]=='baseball bat'
warm=time.perf_counter()
for model in [pose,bat]:model.predict(np.zeros((540,960,3),np.uint8),imgsz=960,device=device,verbose=False)
warm=time.perf_counter()-warm
runtime={'platform':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'opencv':cv2.__version__,'av':av.__version__,'device':device,'torchThreads':4,'inputLongSide':960,'warmupSeconds':warm,'timing':'Per-frame wall time including preprocess/inference/postprocess, excluding decode and model load; first warmup excluded','models':[]}
for name in ['yolo11s-pose.pt','yolo11m.pt']:
    p=ROOT/'models'/name
    runtime['models'].append({'name':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'url':'https://github.com/ultralytics/assets/releases/download/v8.3.0/'+name,'license':'AGPL-3.0 / commercial Enterprise option','pretraining':'COCO; no lab finetuning'})
dump(ROOT/'provenance/runtime.json',runtime)
for c in load(ROOT/'manifest.json')['clips']:
    clip=c['id'];out=ROOT/'outputs'/clip;out.mkdir(exist_ok=True)
    vf=select_vision(clip)
    selector=HitterSelector();frames=[];batframes=[];timing={'pose':[],'bat':[]};lastbat=None
    with av.open(str(ROOT/c['source'])) as container:
        stream=container.streams.video[0]
        for i,f in enumerate(container.decode(stream)):
            assert f.rotation==0,'Source has rotation; explicitly apply before inference'
            img=f.to_ndarray(format='bgr24');h,w=img.shape[:2];pts=float(f.pts*f.time_base)
            assert abs(pts-vf[i]['time'])<.0001,'PTS mismatch'
            t=time.perf_counter();r=pose.predict(img,imgsz=960,conf=.25,device=device,verbose=False)[0];sync();timing['pose'].append(time.perf_counter()-t)
            candidates=[]
            for person in r.keypoints.data.cpu().numpy():
                candidates.append({name:{'x':float(p[0]/w),'y':float(p[1]/h),'confidence':float(p[2])} for name,p in zip(COCO,person)})
            k,status=selector.choose(candidates);points={} if k is None else candidates[k]
            frames.append({'time':pts,'pts':f.pts,'timeBase':str(f.time_base),'points':points,'candidates':candidates,'people':len(candidates),'status':status,'selectedIndex':k,'trackId':'H1' if k is not None else None})
            t=time.perf_counter();r=bat.predict(img,imgsz=960,conf=.10,classes=[34],device=device,verbose=False)[0];sync();timing['bat'].append(time.perf_counter()-t)
            boxes=r.boxes.data.cpu().numpy();raw=[];eligible=[]
            hands=[xy(p) for n,p in points.items() if n.endswith('wrist') and visible(p)]
            for row in boxes:
                b=row[:4]/[w,h,w,h];conf=float(row[4]);rec={'bbox':b.tolist(),'confidence':conf};raw.append(rec)
                if not hands or conf<.25:continue
                distance=min(np.linalg.norm(np.maximum(b[:2]-hand,0)+np.minimum(b[2:]-hand,0)) for hand in hands)
                if distance>.12:continue
                score=conf-distance+(.1*iou(b,lastbat) if lastbat is not None else 0)
                eligible.append((score,rec))
            eligible.sort(key=lambda x:-x[0]);chosen=eligible[0][1] if eligible else None
            if chosen:lastbat=np.array(chosen['bbox'])
            batframes.append({'time':pts,'rawCandidates':raw,'detection':chosen,'status':'observed_bbox' if chosen else 'missing'})
            if i%30==0:print(clip,i,'pose',len(candidates),'bats',len(raw),flush=True)
    dump(out/'yolo-pose.json',{'method':'YOLO11s-pose COCO + same hitter association','frames':frames})
    dump(out/'yolo-bat.json',{'method':'YOLO11m COCO class 34; confidence >=0.25, wrist-distance gate, temporal bbox association; boxes only','frames':batframes})
    dump(out/'timing.json',{'frameSeconds':timing,'meanMilliseconds':{k:1000*np.mean(v) for k,v in timing.items()}})
    print(clip,'DONE',{k:round(1000*np.mean(v),1) for k,v in timing.items()},flush=True)
subprocess.run([str(ROOT/'.venv/bin/pip'),'freeze'],stdout=open(ROOT/'provenance/requirements-lock.txt','w'),check=True)
