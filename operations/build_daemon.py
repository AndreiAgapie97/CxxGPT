"""Keep the free cloud corpus assembly running across transient failures."""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('/workspace/CxxGPT')
existing=subprocess.run(['pgrep','-f','^python scripts/build_licensed_corpus.py$'],capture_output=True,text=True).stdout.split()
while existing:
    alive=[]
    for pid in existing:
        path=Path('/proc')/pid/'stat'
        try:
            if path.read_text().split()[2]!='Z':alive.append(pid)
        except FileNotFoundError:pass
    existing=alive
    if existing:time.sleep(10)
for attempt in range(120):
    manifest=json.loads((ROOT/'data/licensed-10b/manifest.json').read_text())
    if manifest['status']=='complete':
        print('10B-token corpus assembly complete',manifest['tokens'],flush=True);break
    print('Restarting resumable corpus assembly:',attempt+1,flush=True)
    with open(ROOT/'runs-licensed-build.log','a') as log:
        p=subprocess.run(['python','scripts/build_licensed_corpus.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if p.returncode==0:break
    tail=(ROOT/'runs-licensed-build.log').read_text()[-3000:]
    if attempt>=5 and not any(s in tail for s in ('disk headroom','exhausting the workspace disk')):
        print('Assembly stopped after repeated non-storage failures',flush=True)
        break
    time.sleep(30)
