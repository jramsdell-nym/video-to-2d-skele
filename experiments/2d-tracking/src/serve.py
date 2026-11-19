#!/usr/bin/env python3
"""Loopback-only preview with byte ranges for native video seeking."""
import argparse
import functools
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import re


class Handler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.byte_range=None
        path=Path(self.translate_path(self.path))
        if path.is_dir():return super().send_head()
        try:f=path.open('rb')
        except OSError:
            self.send_error(404);return None
        size=path.stat().st_size;start=0;end=size-1
        value=self.headers.get('Range')
        if value:
            m=re.fullmatch(r'bytes=(\d*)-(\d*)',value.strip())
            if not m or not any(m.groups()):
                f.close();self.send_error(416);return None
            if m[1]:start=int(m[1]);end=min(int(m[2]) if m[2] else end,end)
            else:start=max(0,size-int(m[2]))
            if start>=size or end<start:
                f.close();self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.end_headers();return None
            self.byte_range=(start,end);self.send_response(206)
            self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        else:self.send_response(200)
        self.send_header('Content-Type',self.guess_type(str(path)))
        self.send_header('Accept-Ranges','bytes');self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(end-start+1));self.end_headers();return f

    def copyfile(self,source,outputfile):
        if self.byte_range is None:return super().copyfile(source,outputfile)
        start,end=self.byte_range;source.seek(start);remaining=end-start+1
        while remaining:
            data=source.read(min(65536,remaining))
            if not data:break
            outputfile.write(data);remaining-=len(data)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8778);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    print(f'Open http://127.0.0.1:{a.port}/index.html',flush=True)
    server=ThreadingHTTPServer(('127.0.0.1',a.port),functools.partial(Handler,directory=str(root)))
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
