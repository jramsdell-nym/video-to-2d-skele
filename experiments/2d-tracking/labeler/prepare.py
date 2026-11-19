"""Index existing local video caches; no downloads or upstream mutations."""
import hashlib,json
from pathlib import Path
import av
ROOT=Path(__file__).resolve().parent
EXPERIMENT=ROOT.parent
OLD=Path('/Users/jramsdell/Documents/bitbucket/fan engagement/experiments/vision-kinatrax')
def read(p):return json.loads(p.read_text())
def main():
    candidates=[]
    for c in read(EXPERIMENT/'manifest.json')['clips']:
        candidates.append(dict(path=EXPERIMENT/c['source'],title=c['athlete'],trial=c['trial'],session=c['session'],side=c['side'],camera='cam5'))
    for c in read(OLD/'runs/pilot-v3/dataset.json')['trials']:
        p=OLD/c['folder'];p=p/('cam5.mp4' if (p/'cam5.mp4').exists() else 'camera.mp4')
        candidates.append(dict(path=p,title=c['athleteName'],trial=c['id'],session=c['session'],side=({-1: 'L', 1: 'R'}.get(c.get('hitterSide'), 'unknown')),camera='cam5'))
    for c in read(EXPERIMENT/'provenance/angle-sample-downloads.json'):
        session,trial,_=c['name'].split('/')
        candidates.append(dict(path=EXPERIMENT/c['local'],title='Recent lab clip',trial=trial,session=session,side='unknown',camera='cam5',uri='gs://YOUR_VIDEO_BUCKET/'+c['name']))
    clips=[];seen=set()
    for c in candidates:
        p=c.pop('path').resolve()
        if not p.is_file():continue
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        if sha in seen:continue
        seen.add(sha)
        with av.open(str(p)) as container:
            frames=list(container.decode(video=0))
        if any(f.rotation for f in frames):raise ValueError('Handle source rotation explicitly: '+str(p))
        clips.append({**c,'id':sha[:20],'sha256':sha,'path':str(p),'width':frames[0].width,'height':frames[0].height,
            'frames':[{'index':i,'pts':f.pts,'timeBase':str(f.time_base),'time':float(f.pts*f.time_base)} for i,f in enumerate(frames)]})
    (ROOT/'clips.json').write_text(json.dumps({'schema':1,'clips':clips},indent=2)+'\n')
    print(f'Indexed {len(clips)} unique cached clips, {sum(len(c["frames"]) for c in clips)} exact source frames.')
if __name__=='__main__':main()
