"""Bounded batch benchmark; checkpoints remain intact and training resumes in finally."""
import datetime,json,os,pathlib,re,signal,statistics,subprocess,time
root=pathlib.Path('/workspace/CxxGPT');r=root/'runs'
original=json.loads((r/'training-status.json').read_text())
controller=int((r/'controller.pid').read_text());trainer=int((r/'trainer.pid').read_text())
args=original['args'];results=[];choice={'batch':16,'accum':8}
os.kill(controller,signal.SIGSTOP)
try:
 os.kill(trainer,signal.SIGINT)
 for _ in range(180):
  p=pathlib.Path('/proc')/str(trainer)/'stat'
  if not p.exists() or p.read_text().split()[2]=='Z':break
  time.sleep(1)
 else:raise RuntimeError('Trainer did not finish saving; no benchmark started')
 original['phase']='benchmark';original['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (r/'training-status.json').write_text(json.dumps(original,indent=2))
 baseline=r/'fresh-eu.ckpt'
 for batch,accum in [(16,8),(32,4)]:
  ck=r/f'batch-probe-{batch}.ckpt';ck.unlink(missing_ok=True);os.link(baseline,ck)
  a=args.copy()
  for flag,value in {'--batch':batch,'--accum':accum,'--out':str(ck),'--eval-every':0,'--save-every':0,'--log-every':1,'--time-limit':0.25,'--log':str(r/f'batch-probe-{batch}.csv'),'--load-log':str(r/f'batch-probe-{batch}.load.csv')}.items():
   a[a.index(flag)+1]=str(value)
  with (r/f'batch-probe-{batch}.log').open('w') as log:
   p=subprocess.Popen(a,cwd=root,stdout=log,stderr=subprocess.STDOUT)
   deadline=time.time()+100;peak=0
   while p.poll() is None:
    if time.time()>deadline:p.send_signal(signal.SIGINT);p.wait(timeout=60);break
    x=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
    try:peak=max(peak,int(x.stdout.strip()))
    except ValueError:pass
    time.sleep(.5)
  text=(r/f'batch-probe-{batch}.log').read_text()
  speeds=[float(x) for x in re.findall(r'\|\s*([0-9]+) tok/s',text)]
  results.append({'batch':batch,'accum':accum,'returncode':p.returncode,'median_tokens_per_second':statistics.median(speeds[-10:]) if speeds else 0,'samples':len(speeds),'peak_mib':peak})
  ck.unlink(missing_ok=True)
  print('BATCH PROBE',json.dumps(results[-1]),flush=True)
 if results[1]['returncode']==0 and results[1]['median_tokens_per_second']>results[0]['median_tokens_per_second']*1.03:
  choice={'batch':32,'accum':4}
finally:
 (r/'batch-choice.json').write_text(json.dumps({**choice,'results':results},indent=2))
 # The stopped controller cannot relaunch its previous batch while the replacement starts.
 try:os.kill(controller,signal.SIGTERM);os.kill(controller,signal.SIGCONT)
 except ProcessLookupError:pass
 p=subprocess.Popen(['python3',str(root/'resume_controller.py')],cwd=root,stdout=(r/'controller.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
 (r/'controller.pid').write_text(str(p.pid))
 print('TRAINING RESUMED',json.dumps(choice),'controller',p.pid,flush=True)
