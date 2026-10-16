"""On-Pod budget stop independent of the chat workspace; uses Pod-scoped runpodctl auth."""
import datetime,json,os,subprocess,time
from pathlib import Path
root=Path('/workspace/CxxGPT')
run=json.loads((root/'run-budget.json').read_text())
deadline=datetime.datetime.fromisoformat(run['hard_stop_utc']).timestamp()
print('ON-POD BUDGET GUARD ARMED',run['pod_id'],run['hard_stop_utc'],flush=True)
while time.time()<deadline:time.sleep(min(30,max(0,deadline-time.time())))
while True:
    result=subprocess.run(['runpodctl','stop','pod',run['pod_id']],capture_output=True,text=True,timeout=120,env={**os.environ,'APIKEY':os.environ['RUNPOD_API_KEY']})
    # CLI output is intentionally omitted: no keys or environment are logged.
    print('Budget-stop request returned',result.returncode,flush=True)
    time.sleep(30)
