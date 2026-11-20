"""Loopback-only Swing Review server. Run with the project's existing PyAV venv."""
from __future__ import annotations

import argparse
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import threading
from urllib.parse import urlparse, parse_qs

from backend import LocalVideoLibrary, ReadOnlyBackend, ROOT, digest
from store import Store, Conflict, JOINTS, PASSES


class Application:
    def __init__(self, data, videos=None):
        self.data = Path(data)
        self.data.mkdir(parents=True, exist_ok=True)
        self.data.chmod(0o700)
        self.store = Store(self.data / 'drafts.sqlite3')
        self.backend = ReadOnlyBackend(self.data)
        self.local_videos = LocalVideoLibrary(videos or self.data / 'local-videos', self.data)
        self.locks = defaultdict(threading.Lock)

    def frame(self, tid, index):
        trial = self.store.get(tid, private=True)
        asset = trial['asset']
        if not 0 <= index < len(asset['clip']['frames']):
            raise ValueError('Unknown frame')
        cache = self.data / 'frames' / asset['clip']['sha256']
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / f'{index}.jpg'
        with self.locks[tid]:
            if not target.exists():
                import av
                if digest(asset['video']) != asset['clip']['sha256']:
                    raise ValueError('Source video bytes changed after import')
                # Decode one continuous pass. Images retain original orientation and dimensions.
                with av.open(asset['video']) as container:
                    for i, frame in enumerate(container.decode(video=0)):
                        expected = asset['clip']['frames'][i]
                        if frame.pts != expected['pts'] or str(frame.time_base) != expected['timeBase']:
                            raise ValueError('Source timestamp mismatch')
                        output = cache / f'{i}.jpg'
                        if not output.exists():
                            partial = output.with_suffix('.partial')
                            frame.to_image().save(partial, 'JPEG', quality=92, subsampling=0)
                            partial.replace(output)
        return target.read_bytes()


class Handler(BaseHTTPRequestHandler):
    server_version = 'SwingReview/1'

    def log_message(self, *args):
        pass

    @property
    def app(self):
        return self.server.app

    def send(self, status, value, content_type='application/json', cache=False):
        if isinstance(value, (dict, list)):
            value = json.dumps(value, allow_nan=False).encode()
        elif isinstance(value, str):
            value = value.encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(value)))
        self.send_header('Cache-Control', 'private, max-age=86400' if cache else 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        try:
            self.end_headers()
            self.wfile.write(value)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def local_request(self, mutation=False):
        allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        if self.headers.get('Host') not in allowed:
            raise PermissionError('Loopback host required')
        origin = self.headers.get('Origin')
        if origin and origin not in {'http://' + host for host in allowed}:
            raise PermissionError('Same-origin requests only')
        if mutation and (self.headers.get('X-Swing-Review') != '1' or not self.headers.get('Content-Type', '').startswith('application/json')):
            raise PermissionError('Local JSON client header required')

    def do_GET(self):
        try:
            self.local_request()
            url = urlparse(self.path)
            parts = url.path.strip('/').split('/')
            query = parse_qs(url.query)
            if url.path == '/api/bootstrap':
                return self.send(200, {'trials': self.app.store.list(), 'backend': self.app.backend.status(), 'joints': JOINTS, 'passes': PASSES})
            if url.path == '/api/catalog':
                mode = query.get('mode', ['cached'])[0]
                rows = self.app.local_videos.catalog() if mode == 'local' else self.app.backend.catalog()
                term = query.get('q', [''])[0].lower()
                if term:
                    rows = [r for r in rows if term in json.dumps({k: r.get(k) for k in ('id', 'trial', 'session', 'camera')}).lower()]
                if mode == 'cached':
                    rows = [r for r in rows if r['cached']]
                elif mode == 'available':
                    rows = [r for r in rows if r['available']]
                return self.send(200, {'total': len(rows), 'rows': rows[:100]})
            if len(parts) >= 3 and parts[:2] == ['api', 'trials']:
                tid = parts[2]
                if len(parts) == 3:
                    return self.send(200, self.app.store.get(tid))
                if len(parts) == 4 and parts[3] == 'history':
                    return self.send(200, self.app.store.history(tid))
                if len(parts) == 4 and parts[3] == 'export':
                    return self.send(200, self.app.store.export(tid))
                if len(parts) == 5 and parts[3] == 'frames':
                    return self.send(200, self.app.frame(tid, int(parts[4])), 'image/jpeg', cache=True)
            files = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}
            if url.path in files:
                path = ROOT / 'static' / files[url.path]
                return self.send(200, path.read_bytes(), mimetypes.guess_type(path)[0] or 'text/plain')
            self.send(404, {'error': 'Not found'})
        except Exception as error:
            self.error(error)

    def do_POST(self):
        try:
            self.local_request(mutation=True)
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 2 * 1024 * 1024:
                raise ValueError('Request must contain 1 byte–2 MiB of JSON')
            body = json.loads(self.rfile.read(length))
            parts = urlparse(self.path).path.strip('/').split('/')
            if parts == ['api', 'refresh']:
                return self.send(200, self.app.backend.refresh())
            if parts == ['api', 'import']:
                trial_id = body['id']
                if not isinstance(trial_id, str):
                    raise ValueError('Invalid trial ID')
                source = self.app.local_videos if trial_id.startswith('local-') else self.app.backend
                with self.app.locks['import']:
                    existing = {t['id'] for t in self.app.store.list()}
                    if trial_id not in existing:
                        self.app.store.add(source.load(trial_id))
                return self.send(200, self.app.store.get(trial_id))
            if len(parts) == 4 and parts[:2] == ['api', 'trials']:
                if parts[3] == 'claim':
                    return self.send(200, self.app.store.claim(parts[2], body))
                return self.send(200, self.app.store.mutate(parts[2], parts[3], body))
            self.send(404, {'error': 'Not found'})
        except Exception as error:
            self.error(error)

    def error(self, error):
        if isinstance(error, PermissionError):
            code = 403
        elif isinstance(error, Conflict):
            code = 409
        elif isinstance(error, (ValueError, KeyError, TypeError, IndexError)):
            code = 400
        else:
            code = 500
        message = str(error) if code != 500 else 'Local operation failed; inspect the server terminal.'
        if code == 500:
            import traceback
            traceback.print_exc()
        self.send(code, {'error': message})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8794)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--videos', type=Path, help='Local video folder (defaults to local-videos under --data)')
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.app = Application(args.data, args.videos)
    print(f'Swing Review: http://127.0.0.1:{server.server_port}/ | local drafts: {args.data}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
