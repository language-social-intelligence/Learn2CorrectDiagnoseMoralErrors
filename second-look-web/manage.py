"""Start, inspect or stop this portable website and its optional public tunnel."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from check_setup import validate_config

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'run'
PYTHON = ROOT / '.venv/bin/python'
PORT = int(os.environ.get('SECOND_LOOK_PORT','8766'))
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def pid(name):
    try:
        value = int((RUN / (name+'.pid')).read_text())
        args = Path(f'/proc/{value}/cmdline').read_bytes().split(b'\0')
        expected = ROOT / ('server.py' if name == 'server' else 'bin/cloudflared')
        return value if str(expected).encode() in args else None
    except (OSError,ValueError):
        return None


def launch(name,command):
    RUN.mkdir(exist_ok=True, mode=0o700)
    os.chmod(RUN,0o700)
    if pid(name):
        print(f'{name} is already running: PID {pid(name)}')
        return
    config = json.loads((ROOT/'assistant_config.json').read_text())
    env = dict(os.environ, HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
               CUDA_VISIBLE_DEVICES=str(config.get('gpu_index',1)))
    env.pop('PYTHONPATH',None)
    with (RUN / (name+'.log')).open('w') as log:
        os.chmod(log.name,0o600)
        process = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (RUN / (name+'.pid')).write_text(str(process.pid)+'\n')
    print(f'Started {name}: PID {process.pid}. Log: {RUN / (name+".log")}')


def status():
    result = {name:pid(name) for name in ('server','tunnel')}
    if result['server'] is None:
        result['model_status'] = 'stopped'
    else:
        try:
            with HTTP.open(f'http://127.0.0.1:{PORT}/api/info',timeout=3) as response:
                info = json.load(response)
            result.update(model_status=info['status'],model=info['model_label'])
        except Exception:
            result['model_status'] = 'unreachable'
    url = RUN / 'url.txt'
    result['public_url'] = url.read_text().strip() if pid('tunnel') and url.exists() else None
    if result['public_url']:
        config = json.loads((ROOT/'assistant_config.json').read_text())
        result['chat_url'] = result['public_url'].rstrip('/') + config.get('public_path', '/')
    print(json.dumps(result,indent=2))


def stop(name):
    value = pid(name)
    if value is None:
        print(f'{name} is not running.')
        return
    os.kill(value,signal.SIGTERM)
    deadline = time.monotonic()+30
    while pid(name) and time.monotonic() < deadline:
        time.sleep(.25)
    if pid(name):
        raise SystemExit(f'{name} is still shutting down; wait and check status before restarting.')
    print(f'Stopped {name}: PID {value}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['start','status','stop','tunnel'])
    parser.add_argument('--server-only',action='store_true',help='Keep the public tunnel when stopping the server')
    parser.add_argument('--legacy',action='store_true',help='Use the optional old one-model toxicity demo')
    args = parser.parse_args()
    if args.action == 'status':
        status()
    elif args.action == 'stop':
        for name in (('server',) if args.server_only else ('tunnel','server')):
            stop(name)
    elif args.action == 'start':
        if pid('server'):
            print('Server is already running. Use status, or stop --server-only before restarting.')
            return
        if not PYTHON.is_file():
            parser.error('Run bash setup.sh first to create .venv/bin/python.')
        config_path = ROOT / ('config.json' if args.legacy else 'assistant_config.json')
        try:
            validate_config(json.loads(config_path.read_text()))
        except (ValueError,OSError,KeyError) as exc:
            parser.error(str(exc))
        with socket.socket() as connection:
            if connection.connect_ex(('127.0.0.1',PORT)) == 0:
                parser.error(f'Port {PORT} is already occupied; do not start a second server on it.')
        launch('server',[str(PYTHON),'-u',str(ROOT/'server.py'),'--config',str(config_path),'--port',str(PORT)])
        print('Models load in the background. Run status until model_status is ready.')
    else:
        if not pid('server'):
            parser.error('Start this package server before starting its tunnel.')
        binary = ROOT / 'bin/cloudflared'
        if not binary.is_file():
            parser.error('Run bash install_cloudflared.sh first.')
        if not pid('tunnel'):
            (RUN/'url.txt').unlink(missing_ok=True)
        launch('tunnel',[str(binary),'tunnel','--no-autoupdate','--protocol','http2','--url',f'http://127.0.0.1:{PORT}'])
        deadline = time.monotonic()+30
        while time.monotonic() < deadline:
            log = (RUN/'tunnel.log').read_text(errors='replace')
            match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com',log)
            if match and pid('tunnel'):
                (RUN/'url.txt').write_text(match.group()+'\n')
                print('Public website:',match.group())
                return
            if not pid('tunnel'):
                parser.error('Tunnel exited. Inspect run/tunnel.log.')
            time.sleep(1)
        parser.error('No URL yet; inspect run/tunnel.log and run status.')


if __name__ == '__main__':
    main()
