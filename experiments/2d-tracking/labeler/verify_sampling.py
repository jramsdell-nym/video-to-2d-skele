"""Verify swing persistence through HTTP using a disposable copy of the label DB."""
import json
import sqlite3
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path
from http.server import ThreadingHTTPServer

import server


def main():
    real_db = server.DB
    with tempfile.TemporaryDirectory(prefix='bat-sampling-qa-') as folder:
        server.DB = Path(folder) / 'qa.sqlite3'
        with sqlite3.connect(f'file:{real_db}?mode=ro', uri=True) as src, sqlite3.connect(server.DB) as dst:
            src.backup(dst)
        server.initialize()
        http = ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        threading.Thread(target=http.serve_forever, daemon=True).start()
        base = f'http://127.0.0.1:{http.server_port}'

        def api(path, body=None):
            request = urllib.request.Request(base + path,
                data=json.dumps(body).encode() if body is not None else None,
                headers={'Content-Type': 'application/json', 'X-Bat-Labeler': '1'})
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.load(response)

        try:
            before = api('/api/bootstrap')['labels']
            ids = list(server.CLIPS)
            clip, other = ids[:2]
            body = {'clipId': clip, 'range': {'startFrame': 7, 'endFrame': 18, 'sampleCount': 5}}
            saved = api('/api/swing-range', body)['range']
            assert api('/api/bootstrap')['swingRanges'][clip] == saved
            assert api('/api/export')['swingRanges'][clip] == saved
            server.initialize()  # Repeated startup migration must preserve data.
            assert api('/api/bootstrap')['swingRanges'][clip] == saved
            api('/api/swing-range', {'clipId':other,'range':{'startFrame':0,'endFrame':0,'sampleCount':1}})
            body['range']['sampleCount'] = 12
            api('/api/swing-range', body)
            assert api('/api/bootstrap')['swingRanges'][other]['sampleCount'] == 1
            for invalid in [
                {'startFrame':18,'endFrame':7,'sampleCount':2},
                {'startFrame':-1,'endFrame':18,'sampleCount':2},
                {'startFrame':0,'endFrame':999,'sampleCount':2},
                {'startFrame':7,'endFrame':18,'sampleCount':13},
                {'startFrame':7,'endFrame':18,'sampleCount':0},
                {'startFrame':7,'endFrame':18,'sampleCount':2.5},
                {'startFrame':True,'endFrame':18,'sampleCount':2},
            ]:
                try:
                    api('/api/swing-range', {'clipId':clip,'range':invalid})
                    raise AssertionError('Invalid range accepted')
                except urllib.error.HTTPError as error:
                    assert error.code == 400
            api('/api/swing-range', {'clipId':clip,'range':None})
            after = api('/api/bootstrap')
            assert clip not in after['swingRanges']
            assert other in after['swingRanges']
            assert before == after['labels'], 'Range edits must never modify annotations'
            print(f'PASS: range persistence, export, per-video isolation, validation, clear, migration; all {len(before)} labels unchanged')
        finally:
            http.shutdown()
            http.server_close()
            server.DB = real_db


if __name__ == '__main__':
    main()
