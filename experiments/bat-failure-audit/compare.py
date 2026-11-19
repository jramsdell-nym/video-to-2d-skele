"""Read-only diagnosis: fixed PTS full-frame vs body ROI, same raw model/decoder."""
from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / '.runtime/yolo')
os.environ['MPLCONFIGDIR'] = str(ROOT / '.runtime/matplotlib')
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['YOLO_AUTOINSTALL'] = 'false'
sys.dont_write_bytecode = True
import av, cv2, json, hashlib, importlib.metadata
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE = Path('/Users/jramsdell/Documents/bitbucket/fan engagement/validation/motion6-phone-swings')
MODEL = ROOT.parent / '2d-tracking/labeler/models/bat-20261109T164108Z/model.pt'
SIZE = 768

def preprocess(rgb):
    h,w = rgb.shape[:2]
    ratio = min(SIZE/w,SIZE/h)
    rw,rh = round(w*ratio),round(h*ratio)
    left,top=(SIZE-rw)//2,(SIZE-rh)//2
    inp=np.full((SIZE,SIZE,3),114,np.uint8)
    inp[top:top+rh,left:left+rw] = cv2.resize(rgb,(rw,rh),interpolation=cv2.INTER_LINEAR)
    return inp,dict(ratio=ratio,left=left,top=top,width=w,height=h,resized_width=rw,resized_height=rh)

def decode(raw, trans, crop, full_shape, points):
    """Matches current app selection: score-distance wrist association else score argmax."""
    h,w=full_shape[:2]
    p=raw[0].T.astype(float)
    xy=p[:,5:].reshape(-1,2,3).copy()
    xy[:,:,0]=((xy[:,:,0]-trans['left'])/trans['ratio']+crop[0])/w
    xy[:,:,1]=((xy[:,:,1]-trans['top'])/trans['ratio']+crop[1])/h
    wrists=np.array([[v['x'],v['y']] for k,v in points.items() if k in ('left_wrist','right_wrist') and v['confidence']>=.3])
    distances=np.full(len(p),np.inf)
    if len(wrists):
        diff=xy[:,1,None,:2]-wrists[None,:,:]
        diff[:,:,0] *= w/h
        distances=np.linalg.norm(diff,axis=2).min(axis=1)
    inside=np.all((xy[:,:,:2]>=0)&(xy[:,:,:2]<=1),axis=(1,2))
    mask=(p[:,4]>=.25)&(xy[:,:,2].min(axis=1)>=.25)&inside&(distances<=.12)
    top=int(np.argmax(p[:,4]))
    associated=bool(mask.any())
    chosen=int(np.argmax(np.where(mask,p[:,4]-distances,-np.inf))) if associated else top
    def record(i):
        return dict(index=i,score=float(p[i,4]),tip=xy[i,0].tolist(),knob=xy[i,1].tolist(),
                    wrist_distance_height=float(distances[i]),inside=bool(inside[i]),
                    valid_2d=bool(p[i,4]>=.25 and xy[i,:,2].min()>=.25 and inside[i]))
    result=record(chosen)
    result.update(associated=associated,top=record(top),eligible_candidates=int(mask.sum()))
    # Distinct endpoint hypotheses, for diagnosis only (not application NMS).
    distinct=[]
    for i in np.argsort(-p[:,4]):
        if p[i,4]<.01:break
        if all(np.linalg.norm(xy[i,:,:2]-xy[j,:,:2])>.05 for j in distinct): distinct.append(int(i))
        if len(distinct)==6:break
    result['distinct_hypotheses']=[record(i) for i in distinct]
    return result

def roi_from_body(points,w,h):
    pts=np.array([[v['x']*w,v['y']*h] for v in points.values() if v['confidence']>=.3])
    lo,hi=pts.min(axis=0),pts.max(axis=0)
    bh=max(hi[1]-lo[1],.25*h)
    return [max(0,int(np.floor(lo[0]-.65*bh))),max(0,int(np.floor(lo[1]-.6*bh))),
            min(w,int(np.ceil(hi[0]+.65*bh))),min(h,int(np.ceil(hi[1]+.2*bh)))]

def plot_row(ax,rgb,records,roi,title):
    h,w=rgb.shape[:2]
    ax.imshow(cv2.resize(rgb,(960,540),interpolation=cv2.INTER_AREA),extent=(0,w,h,0))
    colors={'saved':'#ff3b30','full':'#00bfff','roi':'#40ff60'}
    for name,r in records.items():
        if r.get('tip') is None or r.get('knob') is None:continue
        pts=[]
        for key in ['tip','knob']:
            p=r[key]
            pts.append([p['x']*w,p['y']*h] if isinstance(p,dict) else [p[0]*w,p[1]*h])
        p=np.array(pts)
        ax.plot(p[:,0],p[:,1],'-o',c=colors[name],lw=2,ms=4,label=name+' '+str(round(r.get('score',r.get('confidence',0)),3)))
        ax.text(p[0,0],p[0,1],name+' T',c=colors[name],fontsize=8)
    ax.set_xlim(roi[0],roi[2]);ax.set_ylim(roi[3],roi[1]);ax.set_title(title,fontsize=11);ax.axis('off');ax.legend(fontsize=8,loc='lower right')

def main():
    torch.set_num_threads(4)
    model=YOLO(str(MODEL)).model.cpu().float().eval()
    rows=[]
    (ROOT/'raw').mkdir(exist_ok=True)
    (ROOT/'figures').mkdir(exist_ok=True)
    for manifest in sorted(SOURCE.glob('*-frames.json')):
        stem=manifest.name.removesuffix('-frames.json')
        meta=json.loads((SOURCE/(stem+'.json')).read_text())
        targets=json.loads(manifest.read_text())
        with av.open(str(SOURCE/(stem+'.source.mov'))) as c:
            for frame in c.decode(video=0):
                target=next((t for t in targets if abs(t['actualTime']-frame.time)<1e-8),None)
                if target is None: continue
                rgb=frame.to_ndarray(format='rgb24')
                h,w=rgb.shape[:2]
                body=min(meta['frames'],key=lambda x:abs(x['time']-frame.time))
                saved=min(meta['batDetections'],key=lambda x:abs(x['time']-frame.time))
                roi=roi_from_body(body['points'],w,h)
                row=dict(clip=stem,pts=frame.pts,time_base=str(frame.time_base),actual_time=frame.time,
                         requested_time=target['requestedTime'],source_shape=list(rgb.shape),body_time=body['time'],roi=roi,saved=saved)
                for name,crop in [('full',[0,0,w,h]),('roi',roi)]:
                    inp,tr=preprocess(rgb[crop[1]:crop[3],crop[0]:crop[2]])
                    tensor=torch.from_numpy(inp.transpose(2,0,1).copy()).unsqueeze(0).float()/255
                    with torch.inference_mode(): raw=model(tensor)[0].cpu().numpy()
                    row[name]=decode(raw,tr,crop,rgb.shape,body['points'])
                    row[name]['transform']=tr
                    np.savez_compressed(ROOT/'raw'/f'{stem}-{frame.pts}-{name}.npz',raw=raw)
                rows.append(row)
                (ROOT/'partial-results.json').write_text(json.dumps(rows,indent=2)+'\n')
                fig,axes=plt.subplots(1,2,figsize=(16,6))
                records={k:row[k] for k in ['saved','full','roi']}
                plot_row(axes[0],rgb,records,[0,0,w,h],f'{stem[6:14]} PTS {frame.pts} ({frame.time:.4f}s)')
                plot_row(axes[1],rgb,records,roi,'Body ROI; red=saved blue=full green=crop')
                fig.tight_layout();fig.savefig(ROOT/'figures'/f'{stem}-{frame.pts}.jpg',dpi=130);plt.close(fig)
                print(stem[6:14],frame.time,'full',round(row['full']['score'],3),row['full']['valid_2d'],'roi',round(row['roi']['score'],3),row['roi']['valid_2d'],flush=True)
    result=dict(model=str(MODEL),model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                runtime={p:importlib.metadata.version(p) for p in ['torch','ultralytics','av','opencv-python','numpy']},
                note='CPU PyTorch on original source frames, not exact reproduction of Core ML/Core Graphics preprocessing. Same coordinates/thresholds/wrist selection; no temporal or 3D gate. ROI based only on saved visible body joints.',rows=rows)
    (ROOT/'results.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
