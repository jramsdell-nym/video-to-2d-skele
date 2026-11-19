"""Verify complete source-aligned inference and exported media; not pose accuracy."""
import hashlib,math
import av,numpy as np
from common import *
from mediapipe_common import NAMES
checks=[]
for c in load(ROOT/'manifest.json')['clips']:
    path=ROOT/c['source'];assert hashlib.sha256(path.read_bytes()).hexdigest()==c['sourceSHA256']
    with av.open(str(path)) as con:source=[(f.pts,str(f.time_base),float(f.pts*f.time_base),f.rotation) for f in con.decode(video=0)]
    assert len(source)==120 and all(r[3]==0 for r in source)
    out=ROOT/'outputs'/c['id'];record={'clip':c['id'],'sourceHashMatches':True,'rotation':0,'variants':{},'videos':[]}
    for variant in ['full','heavy']:
        obj=load(out/f'mediapipe-{variant}.json');rows=obj['frames'];assert len(rows)==120
        assert obj['config']['runningMode']=='VIDEO' and obj['config']['numPoses']==4
        assert obj['timing']['firstFrameReplayMaxDifference']<1e-6
        assert hashlib.sha256((ROOT/'models'/obj['model']['file']).read_bytes()).hexdigest()==obj['model']['sha256']
        for i,(r,s) in enumerate(zip(rows,source)):
            assert (r['frameIndex'],r['pts'],r['timeBase'])==(i,s[0],s[1]) and abs(r['time']-s[2])<1e-9
            assert r['timestampMs']==round(s[2]*1000)
            if i:assert rows[i-1]['timestampMs']<r['timestampMs']
            k=r['selectedIndex']
            assert (r['points']=={})==(k is None)
            if k is not None:
                assert r['points']==r['candidates'][k] and r['worldLandmarksMeters']==r['worldCandidatesMeters'][k]
                assert set(r['points'])==set(NAMES)
            for p in r['points'].values():
                assert all(math.isfinite(p[k]) for k in ['x','y','z','visibility','presence','confidence'])
                assert 0<=p['visibility']<=1 and p['confidence']==p['visibility']
        record['variants'][variant]={'frames':len(rows),'sourcePTSError':0,'independentFirstFrameResetVerified':True,'selectedCandidateMappingVerified':True}
    for file in ['mediapipe-comparison.mp4','mediapipe-detail.mp4','mediapipe-full-overlay.mp4','mediapipe-heavy-overlay.mp4']:
        with av.open(str(out/file)) as con:
            frames=list(con.decode(video=0));times=[float(f.pts*f.time_base) for f in frames]
        assert len(times)==120
        error=max(abs(a-b[2]) for a,b in zip(times,source));assert error<1e-6
        record['videos'].append({'file':str((out/file).relative_to(ROOT)),'frames':len(times),'maxPTSError':error,'dimensions':[frames[0].width,frames[0].height]})
    checks.append(record)
dump(ROOT/'provenance/mediapipe-validation.json',{'passed':True,'scope':'Source identity, every output frame/PTS, distinct tracker reset, selected-candidate coordinate mapping, finite scores, all 16 new videos fully decoded. No accuracy assertion.','clips':checks})
print('PASS: 960 inference frames, 8 independent resets, 16 full MP4 decodes / 1,920 matching output timestamps.')
