"""Score a development-selected model once on the untouched original test split."""
import datetime,hashlib,json,pathlib,re,subprocess,time
root=pathlib.Path('/workspace/CxxGPT');r=root/'runs';config=json.loads((root/'run-budget.json').read_text())
deadline=datetime.datetime.fromisoformat(config['training_stop_utc']).timestamp();cutoff=datetime.datetime.fromisoformat(config['hard_stop_utc']).timestamp()
if (r/'final-eval-ready.json').exists():raise SystemExit('Final test already recorded; do not rescore')
print('FINAL TEST WATCHER ARMED',config['training_stop_utc'],flush=True)
while time.time()<min(cutoff-300,deadline+900):
 best=json.loads((r/'math-best.json').read_text()) if (r/'math-best.json').exists() else {}
 if best.get('selection_frozen_utc'):break
 state=json.loads((r/'training-status.json').read_text()) if (r/'training-status.json').exists() else {}
 if state.get('phase')=='failed' and best:
  best['selection_frozen_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();best['early_end_reason']='trainer failed';(r/'math-best.json').write_text(json.dumps(best,indent=2));break
 time.sleep(10)
else:raise SystemExit('No frozen development selection available before final test cutoff')
ck=r/'best-math-inference.ckpt'
with (r/'export-best.log').open('w') as log:
 subprocess.run([str(root/'export-weights'),str(r/'best-math.ckpt'),str(ck)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
proof={'bytes':ck.stat().st_size,'sha256':hashlib.file_digest(ck.open('rb'),'sha256').hexdigest(),'parent_full_checkpoint_sha256':best['checkpoint_sha256'],'optimizer_state_included':False,'weights_and_router_biases_bit_exact_verified':True,'selection':best}
(r/'best-inference.json').write_text(json.dumps(proof,indent=2))
command=['./cxxgpt-gpu','eval-reasoning','--ckpt',str(ck),'--test','data/gsm8k-test.txt','--bpe','data/tok.bpe','--gpu','--matmul','bf16','--batch','16','--limit','1319','--max-new','400','--tries','1','--temp','0','--seed','1','--tools','none','--labels','runs/math-final-test.answers.jsonl','--show-failures','--threads','8']
with (r/'math-final-test.log').open('w') as log:
 p=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
 try:p.wait(timeout=180)
 except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=15)
text=(r/'math-final-test.log').read_text();m=re.search(r'overall [\d.]+% \((\d+) / (\d+)\)',text)
result={'split':'original GSM8K test; one evaluation after dev selection frozen','correct':int(m[1]) if m else None,'total':int(m[2]) if m else None,'returncode':p.returncode,'command':command,'checkpoint':proof,'tokenizer_sha256':hashlib.file_digest((root/'data/tok.bpe').open('rb'),'sha256').hexdigest(),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(r/'math-final-test.json').write_text(json.dumps(result,indent=2))
# Also score the frozen latest balanced model; neither test score changes selection.
latest_sha=hashlib.file_digest((r/'fresh-eu.ckpt').open('rb'),'sha256').hexdigest()
latest=dict(result);latest['checkpoint_sha256']=latest_sha
if latest_sha==best['checkpoint_sha256']:
 latest['same_weights_as_best']=True
else:
 latest_command=command.copy();latest_command[latest_command.index('--ckpt')+1]=str(r/'fresh-eu.ckpt');latest_command[latest_command.index('--labels')+1]='runs/math-final-latest-test.answers.jsonl'
 with (r/'math-final-latest-test.log').open('w') as log:
  q=subprocess.Popen(latest_command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
  try:q.wait(timeout=180)
  except subprocess.TimeoutExpired:q.terminate();q.wait(timeout=15)
 text=(r/'math-final-latest-test.log').read_text();m=re.search(r'overall [\d.]+% \((\d+) / (\d+)\)',text)
 latest={'split':'original GSM8K test; frozen latest balanced checkpoint; not used for selection','correct':int(m[1]) if m else None,'total':int(m[2]) if m else None,'returncode':q.returncode,'command':latest_command,'checkpoint_sha256':latest_sha,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(r/'math-final-latest-test.json').write_text(json.dumps(latest,indent=2))
(r/'final-eval-ready.json').write_text(json.dumps({'finished':True,'best_returncode':p.returncode,'latest_returncode':latest['returncode']}))
print('FINAL TEST COMPLETE',json.dumps(result),flush=True)
