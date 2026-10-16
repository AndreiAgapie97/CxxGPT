"""Independent fail-safe: stop only this run's Pod at its absolute deadline."""
import datetime
import json
from pathlib import Path
import subprocess
import time

config = Path('/workspace/cxxgpt-ops/run.json')
while not config.exists():
    time.sleep(10)
run = json.loads(config.read_text())
deadline = datetime.datetime.fromisoformat(run['hard_stop_utc']).timestamp()
print('Budget guard armed:', run['pod_id'], run['hard_stop_utc'], flush=True)
while time.time() < deadline:
    time.sleep(min(30, max(0, deadline - time.time())))
while True:
    try:
        args = json.dumps({'id':run['pod_id'], 'body':{'action':'stop'}})
        p = subprocess.run(['python','/workspace/cxxgpt-ops/mcp_bridge.py','pod-action',args],capture_output=True,text=True,timeout=120)
        print('Budget stop:', p.returncode, 'request returned',flush=True)
        p = subprocess.run(['python','/workspace/cxxgpt-ops/mcp_bridge.py','get-pod',json.dumps({'id':run['pod_id']})],capture_output=True,text=True,timeout=120)
        result = json.loads(p.stdout)
        for content in result.get('content',[]):
            if content.get('type')=='text':
                pod = json.loads(content['text'])
                if pod.get('status') in ('EXITED','ERROR'):
                    print('GPU STOP VERIFIED',pod['id'],pod['status'],flush=True)
                    raise SystemExit(0)
    except Exception as e:
        print('Budget-stop retry:',type(e).__name__,str(e)[:500],flush=True)
    time.sleep(30)
