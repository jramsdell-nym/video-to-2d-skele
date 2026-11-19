"""Bounded, GET-only object-name inventory. No media download or cloud mutation."""
import json
import os
from collections import Counter
from pathlib import Path
import requests
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

ROOT = Path(__file__).resolve().parents[1]
creds = Credentials.from_authorized_user_file(str(Path.home()/'.config/gcloud/application_default_credentials.json'))
creds.refresh(Request())
identity = requests.get('https://www.googleapis.com/oauth2/v3/tokeninfo', params={'access_token':creds.token}, timeout=20).json().get('email')
assert identity == os.environ['EXPECTED_GCLOUD_ACCOUNT'], 'Unexpected ADC identity'
headers = {'Authorization':'Bearer '+creds.token}
base = 'https://storage.googleapis.com/storage/v1/b/YOUR_VIDEO_BUCKET/o'
ledger=[]
def listing(params, cap):
    items=[]; prefixes=[]; token=None
    for page in range(cap):
        p={**params, 'maxResults':1000, 'fields':'items(name,size,generation),prefixes,nextPageToken'}
        if token:p['pageToken']=token
        r=requests.get(base,headers=headers,params=p,timeout=45);r.raise_for_status();j=r.json()
        ledger.append({'prefix':params.get('prefix'), 'page':page+1,'objects':len(j.get('items',[]))})
        items+=j.get('items',[]);prefixes+=j.get('prefixes',[]);token=j.get('nextPageToken')
        if not token:break
    return items,prefixes,not bool(token)
_,prefixes,complete=listing({'prefix':'session_2026-', 'delimiter':'/', 'startOffset':'session_2026-10-01', 'endOffset':'session_2026-11-09'},6)
(ROOT/'provenance/session-prefixes.json').write_text(json.dumps({'prefixes':prefixes,'complete':complete,'requests':ledger},indent=2))
assert complete, 'Root inventory cap reached'
recent=sorted(prefixes)[-4:];sessions=[]
for prefix in recent:
    items,_,complete=listing({'prefix':prefix},10)
    videos=[i for i in items if i['name'].endswith('.mp4')]
    trials={i['name'].split('/')[1] for i in items if '/markers_' in i['name']}
    sessions.append({'session':prefix.rstrip('/'),'complete':complete,'listedObjects':len(items),
        'markerTrials':len(trials),'captureDates':dict(Counter(t[8:18] for t in trials)),
        'videos':len(videos),'cameras':dict(Counter(v['name'].rsplit('_',1)[-1] for v in videos)),
        'videoExamples':([v for v in videos if v['name'].endswith('_cam4.mp4')][:4]+[v for v in videos if v['name'].endswith('_cam5.mp4')][:4]), 'angleVerified':False})
result={'asOf':'2026-11-08','bucket':'YOUR_VIDEO_BUCKET','method':'Read-only GCS JSON API listing; last four lexicographically dated session folders; 10000-object cap per session', 'rootComplete':complete,'sessions':sessions,'requests':ledger}
(ROOT/'provenance/session-coverage.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'sessions':sessions},indent=2))
