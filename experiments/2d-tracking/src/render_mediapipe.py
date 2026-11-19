"""Add MediaPipe assets without overwriting the original body/bat experiment."""
from fractions import Fraction
import av,cv2,numpy as np
from common import *
from mediapipe_common import NAMES,EDGES
cv2.setNumThreads(2)
TITLES={'source':'Original source','original':'Apple Vision | original rule','selected':'Apple Vision | selected hitter','pose':'YOLO11s-pose | selected hitter','full':'MediaPipe Full | selected hitter','heavy':'MediaPipe Heavy | selected hitter'}
ORDER=list(TITLES)
COLORS={'left':(255,225,55),'right':(110,125,255),'mid':(240,240,240)}

def crop_for(frames,joints):
    ps=np.array([[p['x'],p['y']] for f in frames for n,p in f['points'].items() if n in joints and visible(p)])
    lo=np.maximum(0,ps.min(0)-[.045,.06]);hi=np.minimum(1,ps.max(0)+[.045,.06])
    center=(lo+hi)/2;size=max(hi-lo) # Equal normalized width/height preserves 16:9 source aspect.
    size=min(1,max(.25,size));lo=np.clip(center-size/2,0,1-size)
    return [*lo,*(lo+size)]

def panel(source,row,title,rect=None,full=False):
    im=source.copy();h,w=im.shape[:2]
    if row is not None:
        p=row['points'];bones=[(NAMES[a],NAMES[b]) for a,b in EDGES] if full else BONES
        def pt(n):return tuple(np.round(xy(p[n])*[w,h]).astype(int))
        for a,b in bones:
            if visible(p.get(a)) and visible(p.get(b)):
                color=COLORS['left' if a.startswith('left') and b.startswith('left') else 'right' if a.startswith('right') and b.startswith('right') else 'mid']
                cv2.line(im,pt(a),pt(b),color,2,cv2.LINE_AA)
        for n,q in p.items():
            if (full or n in MAJOR) and visible(q):
                color=COLORS['left' if n.startswith('left') else 'right' if n.startswith('right') else 'mid'];cv2.circle(im,pt(n),3,color,-1,cv2.LINE_AA)
                if n.endswith('wrist'):cv2.putText(im,n[0].upper(),tuple(np.array(pt(n))+[5,-5]),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
    if rect:
        x1,y1,x2,y2=np.round(np.array(rect)*[w,h,w,h]).astype(int);im=cv2.resize(im[y1:y2,x1:x2],(w,h))
    cv2.rectangle(im,(0,0),(w,57),(18,24,30),-1)
    cv2.putText(im,title,(12,23),cv2.FONT_HERSHEY_SIMPLEX,.64,(245,245,245),1,cv2.LINE_AA)
    if row is not None:
        n=sum(visible(row['points'].get(j)) for j in MAJOR)
        status=f"{row['time']:.3f}s | {n}/12 shared joints | {row['people']} people | {'H1' if n>=5 else 'MISSING'}"
    else:status='Original orientation | identical source frame in every panel'
    cv2.putText(im,status,(12,47),cv2.FONT_HERSHEY_SIMPLEX,.48,(200,215,224),1,cv2.LINE_AA)
    return im

def writer(path,w,h):
    c=av.open(str(path),'w');s=c.add_stream('libx264',rate=60);s.width=w;s.height=h;s.pix_fmt='yuv420p';s.time_base=Fraction(1,15360);s.codec_context.time_base=s.time_base;s.options={'crf':'19','preset':'fast','bf':'0'};return c,s
def put(pair,im,source):
    c,s=pair;f=av.VideoFrame.from_ndarray(im,format='bgr24');f.pts=source.pts;f.time_base=source.time_base
    for p in s.encode(f):c.mux(p)
def close(pair):
    c,s=pair
    for p in s.encode():c.mux(p)
    c.close()

def main():
    raw=(ROOT/'gallery-data.js').read_text();old=json.loads(raw.removeprefix('window.REVIEW=').removesuffix(';'))
    gallery={'clips':{},'metrics':[],'crops':{},'names':NAMES,'edges':EDGES,'runtime':load(ROOT/'provenance/mediapipe-runtime.json'),**load(ROOT/'provenance/mediapipe-observations.json')}
    summaries=[]
    for clip in old['manifest']['clips']:
        name=clip['id'];out=ROOT/'outputs'/name;data=old['clips'][name].copy();models={}
        for variant in ['full','heavy']:
            models[variant]=load(out/f'mediapipe-{variant}.json');data[variant]=models[variant]['frames']
        gallery['clips'][name]={v:[{k:f[k] for k in ['frameIndex','time','pts','timeBase','points','people','selectedIndex','status']} for f in data[v]] for v in ['full','heavy']}
        crops={'full':[0,0,1,1],'upper':crop_for(data['pose'],['left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist']),'feet':crop_for(data['pose'],['left_knee','right_knee','left_ankle','right_ankle'])};gallery['crops'][name]=crops
        metric={'clip':name,'models':{}}
        for key in ORDER[1:]:
            fs=data[key];n=len(fs)
            metric['models'][key]={'retained':sum(sum(visible(f['points'].get(j)) for j in MAJOR)>=5 for f in fs),'sharedJointCoverage':{str(t):sum(bool((p:=f['points'].get(j)) and 0<=p['x']<=1 and 0<=p['y']<=1 and p['confidence']>=t) for f in fs for j in MAJOR)/(12*n) for t in [.3,.5,.7,.9]},'wristCoverage':sum(visible(f['points'].get(j)) for f in fs for j in ['left_wrist','right_wrist'])/(2*n),'maxPeople':max(f['people'] for f in fs)}
            if key in models:metric['models'][key]['timing']={k:v for k,v in models[key]['timing'].items() if not k.endswith('FrameSeconds')}
        gallery['metrics'].append(metric)
        writers={'compare':writer(out/'mediapipe-comparison.mp4',1920,720),'detail':writer(out/'mediapipe-detail.mp4',1920,1080),'full':writer(out/'mediapipe-full-overlay.mp4',960,540),'heavy':writer(out/'mediapipe-heavy-overlay.mp4',960,540)}
        audit=[]
        with av.open(str(ROOT/clip['source'])) as container:
            for i,f in enumerate(container.decode(video=0)):
                src=f.reformat(width=960,height=540,format='bgr24').to_ndarray()
                panels=[cv2.resize(panel(src,None if k=='source' else data[k][i],TITLES[k]),(640,360)) for k in ORDER]
                compare=np.vstack([np.hstack(panels[:3]),np.hstack(panels[3:])]);put(writers['compare'],compare,f)
                detail=[panel(src,data[k][i],TITLES[k],crops['upper']) for k in ['selected','pose','full','heavy']]
                put(writers['detail'],np.vstack([np.hstack(detail[:2]),np.hstack(detail[2:])]),f)
                for variant in ['full','heavy']:put(writers[variant],panel(src,data[variant][i],TITLES[variant]+' | 33 landmarks',full=True),f)
                if i in [0,30,60,80,100,119]:
                    audit.append(np.hstack([cv2.resize(im,(400,225)) for im in detail]))
                    cv2.imwrite(str(out/f'mediapipe-compare-{i:03d}.jpg'),compare)
                    if i==80:summaries.append(np.hstack([cv2.resize(im,(400,225)) for im in detail]))
        for w in writers.values():close(w)
        cv2.imwrite(str(out/'mediapipe-contact-sheet.jpg'),np.vstack(audit))
        print(name,json.dumps(metric),flush=True)
    cv2.imwrite(str(ROOT/'outputs/mediapipe-four-clips.jpg'),np.vstack(summaries))
    dump(ROOT/'outputs/mediapipe-metrics.json',gallery['metrics'])
    (ROOT/'mediapipe-data.js').write_text('window.MEDIAPIPE='+json.dumps(gallery,separators=(',',':'))+';')
if __name__=='__main__':main()
