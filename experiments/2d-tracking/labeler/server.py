"""Loopback-only bat labeler, exact decoded frames and durable SQLite annotations."""
import argparse,io,json,math,sqlite3,threading
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs
import av
import predictor

ROOT=Path(__file__).resolve().parent
CLIPS={c['id']:c for c in json.loads((ROOT/'clips.json').read_text())['clips']}
CACHE=ROOT/'cache';CACHE.mkdir(exist_ok=True)
FRAME_LOCK=threading.Lock()
DB=ROOT/'annotations.sqlite3'
def now():return datetime.now(timezone.utc).isoformat()
def connection():
    con=sqlite3.connect(DB,timeout=20);con.row_factory=sqlite3.Row;return con
def initialize():
    with connection() as con:
        con.executescript('''CREATE TABLE IF NOT EXISTS labels (clip TEXT, frame INTEGER, record TEXT NOT NULL, PRIMARY KEY(clip,frame));
        CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, updated TEXT, clip TEXT, frame INTEGER, record TEXT);
        CREATE TABLE IF NOT EXISTS clip_state (clip TEXT PRIMARY KEY, status TEXT);
        CREATE TABLE IF NOT EXISTS swing_ranges (clip TEXT PRIMARY KEY, record TEXT NOT NULL);''')
def public(c):return {k:v for k,v in c.items() if k!='path'}
def all_labels():
    with connection() as con:return [json.loads(r['record']) for r in con.execute('SELECT record FROM labels ORDER BY clip,frame')]
def all_swing_ranges():
    with connection() as con:return {r['clip']:json.loads(r['record']) for r in con.execute('SELECT * FROM swing_ranges')}
def validate_swing(body):
    clip=CLIPS.get(body.get('clipId'));swing=body.get('range')
    if clip is None:raise ValueError('Unknown video')
    if swing is None:return None
    if not isinstance(swing,dict):raise ValueError('Invalid swing range')
    start,end,count=(swing.get(k) for k in ['startFrame','endFrame','sampleCount'])
    if any(type(v)!=int for v in [start,end,count]):raise ValueError('Use whole frame numbers')
    if not 0<=start<=end<len(clip['frames']):raise ValueError('Swing start must be before its end and within the video')
    if not 1<=count<=end-start+1:raise ValueError('Choose between 1 and the number of frames in this swing')
    return {'startFrame':start,'endFrame':end,'sampleCount':count,'updatedAt':now()}
def validate(body):
    clip=CLIPS.get(body.get('clipId'));frame=body.get('frameIndex')
    if clip is None or type(frame)!=int or not 0<=frame<len(clip['frames']):raise ValueError('Unknown clip or frame')
    status=body.get('status')
    if status not in ['draft','reviewed','skipped','absent']:raise ValueError('Invalid annotation status')
    points={}
    for name in ['tip','knob']:
        point=body.get('points',{}).get(name,{'state':'unmarked'})
        state=point.get('state')
        if state not in ['visible','occluded','uncertain','unmarked']:raise ValueError('Invalid endpoint state')
        item={'state':state,'x':None,'y':None,'pixelX':None,'pixelY':None}
        if state=='visible':
            x,y=point.get('x'),point.get('y')
            if type(x) not in [int,float] or type(y) not in [int,float] or not math.isfinite(x+y) or not (0<=x<=1 and 0<=y<=1):raise ValueError('Endpoint outside image')
            item.update(x=x,y=y,pixelX=x*clip['width'],pixelY=y*clip['height'])
        points[name]=item
    if status=='reviewed' and any(p['state']=='unmarked' for p in points.values()):raise ValueError('Mark or classify both endpoints first')
    if status in ['skipped','absent']:
        points={n:{'state':'unmarked','x':None,'y':None,'pixelX':None,'pixelY':None} for n in ['tip','knob']}
    assistance=None
    if body.get('assistance'):
        suggested=predictor.cached(body['assistance'].get('modelId',''),clip['id'],frame)
        if not suggested or not suggested.get('detected'):raise ValueError('Unknown model suggestion')
        assistance={'modelId':suggested['modelId'],'suggestedPoints':suggested['points'],'modelScore':suggested['score'],
                    'inTrainingSet':suggested['inTrainingSet'],'suggestedAt':suggested['createdAt']}
    record={'clipId':clip['id'],'videoSHA256':clip['sha256'],'session':clip['session'],'trial':clip['trial'],'camera':clip['camera'],
        'frameIndex':frame,'source':clip['frames'][frame],'imageWidth':clip['width'],'imageHeight':clip['height'],
        'coordinateSystem':'original unmirrored image; normalized top-left x,y; continuous pixel coordinates',
        'status':status,'points':points,'samplingStep':max(1,min(len(clip['frames']),int(body.get('samplingStep',5)))),
        'note':str(body.get('note',''))[:1000],'updatedAt':now(),'origin':'manual_labeler'}
    if assistance:
        record.update(assistance=assistance,origin='human_reviewed_model' if status in ['reviewed','absent'] else 'model_assisted_draft')
    return record
def frame_jpeg(c,index):
    path=CACHE/f'{c["id"]}-{index:04d}.jpg'
    with FRAME_LOCK:
        if not path.exists():
            # Full decode to requested index avoids keyframe-seek rounding and FPS assumptions.
            with av.open(c['path']) as container:
                for i,f in enumerate(container.decode(video=0)):
                    if i==index:
                        assert f.pts==c['frames'][index]['pts'] and str(f.time_base)==c['frames'][index]['timeBase']
                        f.to_image().save(path,format='JPEG',quality=97,subsampling=0);break
            entries=sorted(CACHE.glob('*.jpg'),key=lambda p:p.stat().st_mtime)
            for old in entries[:-160]:old.unlink(missing_ok=True)
        return path.read_bytes()
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send(self,status,data,kind='application/json'):
        if isinstance(data,(dict,list)):data=json.dumps(data).encode()
        if isinstance(data,str):data=data.encode()
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        try:self.end_headers();self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError):pass  # A user may skip while inference finishes.
    def do_GET(self):
        u=urlparse(self.path);parts=u.path.strip('/').split('/')
        try:
            if u.path=='/api/bootstrap':
                with connection() as con:states={r['clip']:r['status'] for r in con.execute('SELECT * FROM clip_state')}
                return self.send(200,{'clips':[public(c) for c in CLIPS.values()],'labels':all_labels(),'clipStates':states,'swingRanges':all_swing_ranges(),'modelStatus':predictor.status()})
            if u.path=='/api/model':return self.send(200,predictor.status())
            if len(parts)==4 and parts[:2]==['api','predict']:
                c=CLIPS[parts[2]];index=int(parts[3])
                if not 0<=index<len(c['frames']):raise ValueError('Invalid frame')
                return self.send(200,predictor.predict(c,index,frame_jpeg))
            if u.path=='/api/export':
                labels=all_labels();reviewed=[r for r in labels if r['status'] in ['reviewed','absent']]
                data={'schema':'bat-labeler-v1','exportedAt':now(),'definitions':{'tip':'center of distal barrel end','knob':'center of handle knob; never substitute the wrist','occluded':'endpoint hidden; no guessed coordinate','uncertain':'cannot localize reliably, including blur','absent':'no visible bat in the image','skipped':'not a negative training label'},'clips':[public(c) for c in CLIPS.values()],'annotations':reviewed,'draftsAndSkips':[r for r in labels if r['status'] not in ['reviewed','absent']]}
                data['swingRanges']=all_swing_ranges()
                return self.send(200,data)
            if len(parts)==3 and parts[0]=='frame':
                c=CLIPS[parts[1]];index=int(parts[2].removesuffix('.jpg'))
                if not 0<=index<len(c['frames']):raise ValueError('Invalid frame')
                return self.send(200,frame_jpeg(c,index),'image/jpeg')
            files={'/':'index.html','/index.html':'index.html','/app.js':'app.js','/sampling.js':'sampling.js','/style.css':'style.css'}
            if u.path in files:
                name=files[u.path];kind={'html':'text/html; charset=utf-8','js':'text/javascript; charset=utf-8','css':'text/css; charset=utf-8'}[name.rsplit('.',1)[1]]
                return self.send(200,(ROOT/name).read_bytes(),kind)
            return self.send(404,{'error':'Not found'})
        except (KeyError,ValueError):return self.send(400,{'error':'Invalid clip/frame request'})
        except Exception as e:return self.send(500,{'error':str(e)})
    def do_POST(self):
        # A custom header plus same-origin check prevents blind cross-site form writes.
        origin=self.headers.get('Origin');allowed={f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}
        if self.headers.get('X-Bat-Labeler')!='1' or (origin and origin not in allowed):return self.send(403,{'error':'Local same-origin requests only'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if length>16384:raise ValueError('Request too large')
            body=json.loads(self.rfile.read(length))
            if self.path=='/api/swing-range':
                swing=validate_swing(body);clip=body['clipId']
                with connection() as con:
                    if swing is None:con.execute('DELETE FROM swing_ranges WHERE clip=?',(clip,))
                    else:con.execute('INSERT OR REPLACE INTO swing_ranges VALUES (?,?)',(clip,json.dumps(swing)))
                    con.execute('INSERT OR REPLACE INTO clip_state VALUES (?,?)',(clip,'in_progress'))
                return self.send(200,{'clipId':clip,'range':swing})
            if self.path=='/api/label':
                record=validate(body)
                with connection() as con:
                    data=json.dumps(record)
                    con.execute('INSERT OR REPLACE INTO labels VALUES (?,?,?)',(record['clipId'],record['frameIndex'],data))
                    con.execute('INSERT INTO events(updated,clip,frame,record) VALUES (?,?,?,?)',(now(),record['clipId'],record['frameIndex'],data))
                return self.send(200,record)
            if self.path=='/api/clip-state':
                clip=body.get('clipId');state=body.get('status')
                if clip not in CLIPS or state not in ['in_progress','skipped','done']:raise ValueError('Invalid clip status')
                with connection() as con:con.execute('INSERT OR REPLACE INTO clip_state VALUES (?,?)',(clip,state))
                return self.send(200,{'ok':True})
            return self.send(404,{'error':'Not found'})
        except (ValueError,TypeError,KeyError) as e:return self.send(400,{'error':str(e)})
        except Exception as e:return self.send(500,{'error':str(e)})
def main():
    global DB
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8779);p.add_argument('--db',type=Path,default=DB);args=p.parse_args();DB=args.db
    initialize();server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Bat labeler: http://127.0.0.1:{args.port}/ | {len(CLIPS)} clips | saves: {DB}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()
