"""Restart failed transfers; only committed, checksum-verified jobs are published."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/workspace/cxxgpt-ops')
deadline=datetime.datetime.fromisoformat(json.loads((ROOT/'run.json').read_text())['backup_start_utc']).timestamp()
existing=subprocess.run(['pgrep','-f','^python /workspace/cxxgpt-ops/sync_corpus.py$'],capture_output=True,text=True).stdout.split()
while existing and time.time()<deadline:
    alive=[]
    for pid in existing:
        path=Path('/proc')/pid/'stat'
        try:
            if path.read_text().split()[2]!='Z':alive.append(pid)
        except FileNotFoundError:pass
    existing=alive
    if existing:time.sleep(10)
while time.time()<deadline:
    corpus=json.loads(Path('/workspace/CxxGPT/data/licensed-10b/manifest.json').read_text())
    state=json.loads((ROOT/'uploaded.json').read_text()) if (ROOT/'uploaded.json').exists() else {'jobs':{}}
    if corpus['status']=='complete' and set(corpus['completed'])<=set(state['jobs']):
        print('All committed corpus jobs transferred',flush=True);break
    with open(ROOT/'sync-corpus.log','a') as log:
        print('Starting/restarting corpus transfer',flush=True)
        p=subprocess.run(['python',str(ROOT/'sync_corpus.py')],stdout=log,stderr=subprocess.STDOUT)
        print('Transfer process exit:',p.returncode,flush=True)
    time.sleep(30)
