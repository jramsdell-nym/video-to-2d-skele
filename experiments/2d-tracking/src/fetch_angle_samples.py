"""Download one small cam5 example from each of the three other recent sessions."""
import json,hashlib
from pathlib import Path
from urllib.parse import quote
import requests
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
ROOT=Path(__file__).resolve().parents[1]
creds=Credentials.from_authorized_user_file(str(Path.home()/'.config/gcloud/application_default_credentials.json'));creds.refresh(Request())
headers={'Authorization':'Bearer '+creds.token};ledger=[];total=0
for s in json.loads((ROOT/'provenance/session-coverage.json').read_text())['sessions'][:-1]:
    v=next(v for v in s['videoExamples'] if v['name'].endswith('_cam5.mp4'))
    size=int(v['size']);assert size<12*1024*1024 and total+size<36*1024*1024
    folder=ROOT/'data/angle-samples'/s['session'];folder.mkdir(parents=True,exist_ok=True)
    url='https://storage.googleapis.com/storage/v1/b/YOUR_VIDEO_BUCKET/o/'+quote(v['name'],safe='')
    r=requests.get(url,headers=headers,params={'alt':'media','generation':v['generation']},timeout=60);r.raise_for_status();assert len(r.content)==size
    p=folder/'cam5.mp4';p.write_bytes(r.content);total+=size
    ledger.append({**v,'local':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(r.content).hexdigest(),'method':'First cached metadata example ending cam5; no quality selection'})
(ROOT/'provenance/angle-sample-downloads.json').write_text(json.dumps(ledger,indent=2))
print('Downloaded',len(ledger),'read-only samples;',total,'bytes')
