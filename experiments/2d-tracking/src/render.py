"""Render source-PTS H.264 comparison videos, audit stills, and compact gallery data."""
import hashlib
from fractions import Fraction
import av,cv2,numpy as np
from common import *
cv2.setNumThreads(2)
COLORS={'left':(255,225,55),'right':(110,125,255),'mid':(240,240,240)}
def label(im,text,y=25,color=(245,245,245),scale=.6):
    cv2.putText(im,text,(10,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
def header(im,title,status):
    cv2.rectangle(im,(0,0),(im.shape[1],56),(18,24,30),-1);label(im,title,22);label(im,status,45,scale=.45)
def body(im,points):
    h,w=im.shape[:2]
    def pt(n):return tuple((xy(points[n])*[w,h]).round().astype(int))
    for a,b in BONES:
        if visible(points.get(a)) and visible(points.get(b)):
            color=COLORS['left' if a.startswith('left') and b.startswith('left') else 'right' if a.startswith('right') and b.startswith('right') else 'mid']
            cv2.line(im,pt(a),pt(b),color,2,cv2.LINE_AA)
    for name,p in points.items():
        if name in MAJOR and visible(p):
            color=COLORS['left' if name.startswith('left') else 'right'];cv2.circle(im,pt(name),3,color,-1,cv2.LINE_AA)
            if name.endswith('wrist'):cv2.putText(im,name[0].upper(),tuple(np.array(pt(name))+[4,-5]),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
def batbox(im,frame):
    if frame['detection']:
        h,w=im.shape[:2];d=frame['detection'];b=np.round(np.array(d['bbox'])*[w,h,w,h]).astype(int)
        cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(75,235,255),2)
        cv2.putText(im,f"bbox {d['confidence']:.2f}",tuple(b[:2]+[0,-5]),cv2.FONT_HERSHEY_SIMPLEX,.45,(75,235,255),1,cv2.LINE_AA)
def batline(im,frame):
    if frame['segment']:
        h,w=im.shape[:2];a,b=np.round(np.array(frame['segment'])*[w,h]).astype(int)
        color=(235,105,235) if frame['status']=='observed_line' else (50,165,255)
        if frame['status']=='observed_line':cv2.line(im,tuple(a),tuple(b),color,3,cv2.LINE_AA)
        else:
            for start in np.arange(0,1,.2):cv2.line(im,tuple((a*(1-start)+b*start).astype(int)),tuple((a*(1-min(1,start+.1))+b*min(1,start+.1)).astype(int)),color,3)
def writer(path,w,h):
    c=av.open(str(path),'w');s=c.add_stream('libx264',rate=60)
    s.width=w;s.height=h;s.pix_fmt='yuv420p';s.time_base=Fraction(1,15360);s.codec_context.time_base=s.time_base
    s.options={'crf':'19','preset':'fast','bf':'0'}
    return c,s
def put(c,s,im,source):
    f=av.VideoFrame.from_ndarray(im,format='bgr24');f.pts=source.pts;f.time_base=source.time_base
    for packet in s.encode(f):c.mux(packet)
def close(c,s):
    for packet in s.encode():c.mux(packet)
    c.close()
def coverage(frames):return sum(sum(visible(p) for p in f['points'].values())>=5 for f in frames)
gallery={'manifest':load(ROOT/'manifest.json'),'clips':{},'metrics':[],'sessions':load(ROOT/'provenance/session-coverage.json')}
for clip in gallery['manifest']['clips']:
    name=clip['id'];out=ROOT/'outputs'/name
    orig=load(ROOT/clip['baseline']);sel=load(out/'vision-selected.json');pose=load(out/'yolo-pose.json');bat=load(out/'yolo-bat.json');line=load(out/'line-bat.json');timing=load(out/'timing.json')
    data={'original':orig['frames'],'selected':sel['frames'],'pose':[{k:v for k,v in f.items() if k!='candidates'} for f in pose['frames']],'bat':bat['frames'],'line':line['frames']}
    gallery['clips'][name]=data
    metrics={'id':name,'frames':len(data['pose']),'bodyFrames':{k:coverage(data[k]) for k in ['original','selected','pose']},'majorJointCoverage':{k:sum(visible(f['points'].get(j)) for f in data[k] for j in MAJOR)/(12*len(data[k])) for k in ['original','selected','pose']},'bboxFrames':sum(f['detection'] is not None for f in data['bat']),'lineFrames':{s:sum(f['status']==s for f in data['line']) for s in ['observed_line','propagated_line','missing']},'milliseconds':{**timing['meanMilliseconds'],'visionAllWithDecode':sel['processingSeconds']*1000/len(data['pose']),'linePostprocess':line['meanMilliseconds']}}
    gallery['metrics'].append(metrics)
    wc,ws=writer(out/'body-comparison.mp4',1920,360);bc,bs=writer(out/'bat-comparison.mp4',1920,360)
    lc,ls=writer(out/'learned-overlay.mp4',960,540);vc,vs=writer(out/'vision-lines-overlay.mp4',960,540)
    audit=[]
    with av.open(str(ROOT/clip['source'])) as container:
        for i,f in enumerate(container.decode(video=0)):
            src=cv2.resize(f.to_ndarray(format='bgr24'),(960,540));pts=float(f.pts*f.time_base)
            assert abs(pts-data['pose'][i]['time'])<1e-6
            bodypanels=[]
            for title,key in [('Apple Vision | original rule','original'),('Apple Vision | hitter selection','selected'),('YOLO11s-pose | hitter selection','pose')]:
                im=src.copy();p=data[key][i];body(im,p['points']);n=sum(visible(p['points'].get(j)) for j in MAJOR)
                header(im,title,f"source {pts:.3f}s | {n}/12 major joints | {p['people']} people | {'H1' if p['points'] else 'MISSING'}")
                bodypanels.append(cv2.resize(im,(640,360)))
            put(wc,ws,np.hstack(bodypanels),f)
            a=src.copy();header(a,'Source | original orientation',f'{name} | frame {i} | source {pts:.3f}s')
            b=src.copy();body(b,data['pose'][i]['points']);batbox(b,data['bat'][i]);header(b,'YOLO11m | bat BOX only',data['bat'][i]['status']+' | no tip/knob estimates')
            l=src.copy();body(l,data['selected'][i]['points']);batline(l,data['line'][i]);header(l,'OpenCV LSD + LK | LINE candidate',data['line'][i]['status']+' | not physical endpoints')
            # Short trace: observed bbox centers only; no lines bridging gaps.
            for j in range(max(0,i-7),i):
                det=data['bat'][j]['detection']
                if det:
                    bb=np.array(det['bbox']);center=(bb[:2]+bb[2:])/2
                    cv2.circle(b,tuple((center*[960,540]).astype(int)),2,(75,235,255),-1)
            put(lc,ls,b,f);put(vc,vs,l,f)
            batpanel=np.hstack([cv2.resize(z,(640,360)) for z in [a,b,l]]);put(bc,bs,batpanel,f)
            if i in [0,30,60,72,80,90,100,119]:
                cv2.imwrite(str(out/f'audit-{i:03d}.jpg'),batpanel);audit.append(batpanel)
    for c,s in [(wc,ws),(bc,bs),(lc,ls),(vc,vs)]:close(c,s)
    cv2.imwrite(str(out/'audit-sheet.jpg'),np.vstack(audit))
    print(name,metrics,flush=True)
dump(ROOT/'outputs/metrics.json',gallery['metrics'])
(ROOT/'gallery-data.js').write_text('window.REVIEW='+json.dumps(gallery,separators=(',',':'))+';')
print('Rendered all comparison videos and gallery data')
