"""Structured agent client; intentionally has no approval or upload commands."""
import argparse
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8794')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('queue')
    catalog = commands.add_parser('catalog'); catalog.add_argument('--mode', choices=['cached','available','all','local'], default='cached')
    load = commands.add_parser('load'); load.add_argument('trial')
    claim = commands.add_parser('claim'); claim.add_argument('trial'); claim.add_argument('--owner', required=True); claim.add_argument('--session', type=Path, required=True)
    inspect = commands.add_parser('inspect'); inspect.add_argument('trial'); inspect.add_argument('--start', type=int, default=0); inspect.add_argument('--end', type=int); inspect.add_argument('--joint')
    frame = commands.add_parser('frame'); frame.add_argument('trial'); frame.add_argument('index', type=int); frame.add_argument('--output', type=Path, required=True)
    for action in ('batch', 'interpolate', 'report', 'submit', 'release', 'renew'):
        sub = commands.add_parser(action); sub.add_argument('--session', type=Path, required=True)
        if action in ('batch', 'interpolate'):
            sub.add_argument('--file', type=Path, required=True, help='JSON with expectedRevision and explicit point changes')
        elif action == 'report':
            sub.add_argument('--file', type=Path, required=True, help='Agent result JSON with models, passes and proposed changes')
            sub.add_argument('--revision', type=int, required=True)
        elif action in ('submit', 'release'):
            sub.add_argument('--revision', type=int, required=True)
    args = parser.parse_args()

    def call(path, body=None, binary=False):
        data = json.dumps(body).encode() if body is not None else None
        request = Request(args.url + path, data=data, headers={'X-Swing-Review': '1', 'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=180) as response:
                content = response.read()
            return content if binary else json.loads(content)
        except HTTPError as error:
            raise SystemExit(error.read().decode()) from None

    if args.command == 'queue':
        result = call('/api/bootstrap')['trials']
    elif args.command == 'catalog':
        result = call('/api/catalog?mode=' + args.mode)
    elif args.command == 'load':
        t = call('/api/import', {'id': args.trial})
        result = {k: t[k] for k in ('id', 'revision', 'state', 'summary')}
    elif args.command == 'claim':
        result = call(f'/api/trials/{args.trial}/claim', {'owner': args.owner})
        args.session.write_text(json.dumps(result)); args.session.chmod(0o600)
        result = {k: v for k, v in result.items() if k != 'leaseToken'}
    elif args.command == 'inspect':
        t = call(f'/api/trials/{args.trial}')
        frames = t['frames'][args.start:None if args.end is None else args.end + 1]
        if args.joint:
            from store import JOINTS, point
            if args.joint not in JOINTS:
                parser.error('Unknown joint')
            frames = [{'frameIndex': f['index'], 'frame': f['record']['frame'],
                       'point': point(f['record'], args.joint), 'baseline': point(f['baseline'], args.joint),
                       'review': f['meta'], 'issues': f['issues']} for f in frames]
        result = {'id': t['id'], 'revision': t['revision'], 'state': t['state'], 'frames': frames}
    elif args.command == 'frame':
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(call(f'/api/trials/{args.trial}/frames/{args.index}', binary=True))
        result = {'frameIndex': args.index, 'path': str(args.output.resolve())}
    else:
        session = json.loads(args.session.read_text())
        tid = session['id']
        if args.command == 'renew':
            result = call(f'/api/trials/{tid}/claim', {'owner': session['owner'], 'leaseToken': session['leaseToken']})
            args.session.write_text(json.dumps(result)); args.session.chmod(0o600)
            result = {k: v for k, v in result.items() if k != 'leaseToken'}
        else:
            if args.command == 'report':
                body = {'expectedRevision': args.revision, 'report': json.loads(args.file.read_text())}
            else:
                body = json.loads(args.file.read_text()) if hasattr(args, 'file') else {'expectedRevision': args.revision}
            body.update(actor={'type': 'agent', 'id': session['owner']}, leaseToken=session['leaseToken'])
            body.setdefault('requestId', uuid.uuid4().hex)
            operation = 'agent-report' if args.command == 'report' else args.command
            result = call(f'/api/trials/{tid}/{operation}', body)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
