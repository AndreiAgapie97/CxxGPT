"""Run all authorized helpers under a foreground execution session."""
from pathlib import Path
import subprocess,time,datetime,json
root=Path('/workspace/cxxgpt-ops')
scripts=['budget_guard.py','recovery_supervisor.py','sync_daemon.py','build_daemon.py','status_reporter.py']
children={}
while True:
 for script in scripts:
  if script not in children or children[script].poll() is not None:
   if script=='recovery_supervisor.py' and (root/'completed.json').exists():continue
   children[script]=subprocess.Popen(['python',str(root/script)],stdout=open(root/(script[:-3].replace('_','-')+'.log'),'a'),stderr=subprocess.STDOUT)
   print('Worker started',script,children[script].pid,flush=True)
 if (root/'completed.json').exists():break
 time.sleep(30)
