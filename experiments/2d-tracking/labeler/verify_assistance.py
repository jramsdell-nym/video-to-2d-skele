"""Exercise the real prediction/review API against a disposable annotation DB."""
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
    with tempfile.TemporaryDirectory(prefix='bat-assistance-qa-') as folder:
        server.DB = Path(folder) / 'qa.sqlite3'
        with sqlite3.connect(f'file:{real_db}?mode=ro', uri=True) as src, sqlite3.connect(server.DB) as dst:
            src.backup(dst)
        server.initialize()
        http = ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{http.server_port}'

        def api(path, body=None):
            req = urllib.request.Request(base + path,
                    data=json.dumps(body).encode() if body is not None else None,
                    headers={'Content-Type': 'application/json', 'X-Bat-Labeler': '1'})
            with urllib.request.urlopen(req, timeout=90) as response:
                return json.load(response)

        checks = []
        try:
            before = api('/api/bootstrap')['labels']
            model = api('/api/model')
            assert model['ready'], 'Train the model before running this check'
            prediction = api('/api/predict/821a299e99cb0a60b164/0')
            assert prediction['detected'], 'Expected a real prediction on the QA clip'
            assert before == api('/api/bootstrap')['labels']
            assert api('/api/predict/821a299e99cb0a60b164/0') == prediction
            checks.append('Prediction and cache reads never write annotation records')

            body = {'clipId': prediction['clipId'], 'frameIndex': prediction['frameIndex'],
                    'status': 'draft', 'points': prediction['points'],
                    'assistance': {'modelId': prediction['modelId']}, 'samplingStep': 5,
                    'note': 'Disposable API verification, never saved to user database'}
            body['points']['tip']['x'] += 0.001
            draft = api('/api/label', body)
            assert draft['origin'] == 'model_assisted_draft'
            assert draft['points']['tip']['x'] != draft['assistance']['suggestedPoints']['tip']['x']
            exported = api('/api/export')
            assert not any(r['clipId'] == body['clipId'] and r['frameIndex'] == 0 for r in exported['annotations'])
            checks.append('Corrected suggestions save as drafts and remain outside reviewed export')

            body['status'] = 'reviewed'
            reviewed = api('/api/label', body)
            assert reviewed['origin'] == 'human_reviewed_model'
            assert reviewed['assistance']['modelId'] == prediction['modelId']
            assert reviewed['source'] == prediction['source']
            assert any(r == reviewed for r in api('/api/export')['annotations'])
            checks.append('Explicit review preserves correction, model provenance and exact frame identity')

            body['assistance'] = {'modelId': '../../invalid'}
            try:
                api('/api/label', body)
                raise AssertionError('Invalid model identifier accepted')
            except urllib.error.HTTPError as error:
                assert error.code == 400
            checks.append('Invalid model provenance rejected')

            after = {(r['clipId'], r['frameIndex']): r for r in api('/api/bootstrap')['labels']}
            for row in before:
                assert after[(row['clipId'], row['frameIndex'])] == row
            checks.append('Every pre-existing annotation preserved')

            original_manifest = server.predictor.manifest
            server.predictor.manifest = lambda: None
            try:
                fallback = api('/api/predict/821a299e99cb0a60b164/5')
                assert fallback['available'] is False
            finally:
                server.predictor.manifest = original_manifest
            checks.append('No-model fallback allows manual labeling')

            result = {'passed': True, 'modelId': model['model']['id'], 'checks': checks,
                      'database': 'Disposable SQLite backup; real user database never written'}
            out = server.ROOT / 'models' / model['model']['id'] / 'api-verification.json'
            out.write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2))
        finally:
            http.shutdown()
            http.server_close()
            server.DB = real_db


if __name__ == '__main__':
    main()
