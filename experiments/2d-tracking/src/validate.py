"""Check real media/trace invariants; does not assert model accuracy."""
import hashlib,math
import av,numpy as np
from common import *
checks=[]
for c in load(ROOT/'manifest.json')['clips']:
    source=ROOT/c['source'];assert hashlib.sha256(source.read_bytes()).hexdigest()==c['sourceSHA256']
    with av.open(str(source)) as container:
        frames=list(container.decode(video=0));times=np.array([float(f.pts*f.time_base) for f in frames]);rotation=[f.rotation for f in frames]
    assert len(times)==120 and np.all(np.diff(times)>0) and not any(rotation)
    out=ROOT/'outputs'/c['id'];tracechecks={}
    for name in ['vision-selected','yolo-pose','yolo-bat','line-bat']:
        rows=load(out/(name+'.json'))['frames'];assert len(rows)==len(times)
        assert np.max(np.abs(np.array([f['time'] for f in rows])-times))<1e-6
        if 'points' in rows[0]:
            for row in rows:
                for p in row['points'].values():assert all(math.isfinite(p[k]) for k in ['x','y','confidence'])
        tracechecks[name]='120 monotonic source-matched timestamps'
    rows=load(out/'line-bat.json')['frames'];age=0
    for row in rows:
        age=age+1 if row['status']=='propagated_line' else 0;assert age<=2
        assert (row['segment'] is None)==(row['status']=='missing')
        assert row['tip'] is None and row['knob'] is None
    videoChecks=[]
    for p in sorted(out.glob('*.mp4')):
        with av.open(str(p)) as container:
            fs=list(container.decode(video=0));ot=np.array([float(f.pts*f.time_base) for f in fs])
        assert len(ot)==len(times);error=float(np.max(np.abs(ot-times)));assert error<1e-6
        videoChecks.append({'file':str(p.relative_to(ROOT)),'frames':len(ot),'maxPTSErrorSeconds':error,'size':[fs[0].width,fs[0].height]})
    checks.append({'clip':c['id'],'sourceSHA256Matches':True,'sourceOrientation':'identity / unmirrored','traces':tracechecks,'videos':videoChecks})
dump(ROOT/'provenance/validation.json',{'passed':True,'scope':'Actual media hashes, full output decode, per-frame timestamps, finite coordinates and missing-state contracts. Not accuracy validation.','clips':checks})
print(f"PASS: {len(checks)} original hashes, {sum(len(c['videos']) for c in checks)} full video decodes, {sum(v['frames'] for c in checks for v in c['videos'])} output PTS, 16 source-aligned traces, line-state contracts.")
