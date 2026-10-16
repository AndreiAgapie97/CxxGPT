"""Continue existing weights on worked math; record every dev answer and keep best/latest."""
import datetime,hashlib,json,os,pathlib,re,signal,subprocess,time
root=pathlib.Path('/workspace/CxxGPT');r=root/'runs';math=root/'data/targeted-math'
config=json.loads((root/'run-budget.json').read_text());deadline=datetime.datetime.fromisoformat(config['training_stop_utc']).timestamp()

def atomic_json(path,obj):
 p=path.with_suffix(path.suffix+'.tmp');p.write_text(json.dumps(obj,indent=2));p.replace(path)
def status(**state):
 state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();atomic_json(r/'training-status.json',state)
def alive(pid):
 p=pathlib.Path('/proc')/str(pid)/'stat'
 return p.exists() and p.read_text().split()[2]!='Z'
def general_loss(name,source=None):
    source=source or r/'fresh-eu.ckpt';tmp=r/(name+'.probe.ckpt');tmp.unlink(missing_ok=True);os.link(source,tmp)
    import struct
    with tmp.open('rb') as f:
        f.read(8);n=struct.unpack('<i',f.read(4))[0];assert 11<=n<=13;f.read(n*4);step=struct.unpack('<q',f.read(8))[0]
    command=['./cxxgpt-gpu','train','--gpu','--matmul','bf16','--muon','--resume','--out',str(tmp),'--data','data/corpus/textbooks-libretexts-0000/train-0000.tok','--val','data/corpus/doab-doab-0001/val-0000.tok','--seq','512','--batch','16','--accum','8','--steps',str(step),'--warmup','0','--eval-every','1','--eval-batches','8','--save-every','0','--log-every','1','--threads','8']
    with (r/(name+'.log')).open('w') as log:
        proc=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        try:proc.wait(timeout=90)
        except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=15)
    text=(r/(name+'.log')).read_text();match=re.search(r'val ([0-9.]+) \(final validation\)',text)
    score=float(match[1]) if match and proc.returncode==0 else None
    assert 'tok/s' not in text, 'General validation must not perform any training updates'
    tmp.unlink(missing_ok=True)
    atomic_json(r/(name+'.json'),{'general_text_loss':score,'source_step':step,'updates':0,'fixed_validation':'DOAB0001 held-out book tokens, B16/T512/8batches/seed12345','returncode':proc.returncode})
    return score

def evaluate(name,test='data/gsm8k-dev.txt',limit=500,timeout=120):
 ck=r/'fresh-eu.ckpt';p=r/(name+'.ckpt');p.unlink(missing_ok=True);os.link(ck,p)
 command=['./cxxgpt-gpu','eval-reasoning','--ckpt',str(p),'--test',test,'--bpe','data/tok.bpe','--gpu','--matmul','bf16','--batch','16','--limit',str(limit),'--max-new','400','--tries','1','--temp','0','--seed','1','--tools','none','--labels',str(r/(name+'.answers.jsonl')),'--show-failures','--threads','8']
 with (r/(name+'.log')).open('w') as log:
  proc=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
  try:proc.wait(timeout=timeout)
  except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=15)
 text=(r/(name+'.log')).read_text();match=re.search(r'overall [\d.]+% \((\d+) / (\d+)\)',text)
 result={'name':name,'split':'reserved development' if 'dev' in test else 'original test','returncode':proc.returncode,'correct':int(match[1]) if match else None,'total':int(match[2]) if match else None,'checkpoint_sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'tokenizer_sha256':hashlib.file_digest((root/'data/tok.bpe').open('rb'),'sha256').hexdigest(),'command':command,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 result['general_text_loss']=general_loss(name+'-general',p)
 atomic_json(r/(name+'.json'),result)
 if result['correct'] is not None and result['total']==500:
  best=json.loads((r/'math-best.json').read_text())
  if result['correct']>best['correct']:
   target=r/'best-math.ckpt';tmp=r/'best-math.ckpt.link';tmp.unlink(missing_ok=True);os.link(p,tmp);tmp.replace(target);atomic_json(r/'math-best.json',result)
  atomic_json(r/'math-latest-eval.json',result)
 p.unlink(missing_ok=True);print('MATH EVALUATION',json.dumps(result),flush=True)
 return result

print('Waiting for verified word-problem corpus while broad training continues',flush=True)
while not (math/'manifest.json').exists():
 if time.time()>=deadline-300:raise SystemExit('Math preparation did not finish before deadline')
 time.sleep(10)
manifest=json.loads((math/'manifest.json').read_text())
for split in ('train','dev'):
 p=math/('word-'+split+'.tok');assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==manifest['splits'][split]['sha256']
assert (math/'gsm8k-train.tok').exists()
# Preserve a deterministic replay subset, not the entire growing literature corpus.
inv=json.loads((root/'data/inventory.json').read_text());replay=[];replay_tokens=0
for f in inv['files']:
 if f['split']=='train':
  replay.append(f['path']);replay_tokens+=f['tokens']
  if replay_tokens>=750_000_000:break
word_tokens=manifest['splits']['train']['tokens'];gsm_tokens=1_238_589
word_weight=4;gsm_weight=64
data=','.join([f'data/targeted-math/word-train.tok*{word_weight}',f'data/targeted-math/gsm8k-train.tok*{gsm_weight}']+replay)
recipe={'priority':'balanced general learning and math accuracy','word_weight':word_weight,'gsm8k_weight':gsm_weight,'replay_tokens':replay_tokens,'weighted_word_tokens':word_tokens*word_weight,'weighted_gsm8k_tokens':gsm_tokens*gsm_weight,'data':data,'sampler':'qa','validation':'reserved GSM8K dev; answer-token loss','selection':'best exact-answer count on same500 dev; strict improvements only; officialtest not used for selection','previous_laptop_comparison':'night2 202/1319 test vs current dev500; not identical evaluation split'}
atomic_json(r/'math-focus-recipe.json',recipe)
old_controller=int((r/'controller.pid').read_text());old_trainer=int((r/'trainer.pid').read_text());os.kill(old_controller,signal.SIGSTOP)
saved=False
try:
 os.kill(old_trainer,signal.SIGINT)
 for _ in range(180):
  if not alive(old_trainer):saved=True;break
  time.sleep(1)
 if not saved:raise RuntimeError('Old trainer still saving; refusing concurrent launch')
 os.kill(old_controller,signal.SIGTERM);os.kill(old_controller,signal.SIGCONT)
except Exception:
 try:os.kill(old_controller,signal.SIGCONT)
 except ProcessLookupError:pass
 raise
(r/'controller.pid').write_text(str(os.getpid()))
initial=r/'math-focus-initial.ckpt'
if not initial.exists():os.link(r/'fresh-eu.ckpt',initial)
base=r/'math-dev500-snapshot.ckpt'
base_info=json.loads((r/'math-dev500-metadata.json').read_text())
best={'correct':6,'total':500,'name':'broad-pretraining-dev500','checkpoint_sha256':base_info['checkpoint_sha256'],'split':'reserved development'}
assert base.exists() and hashlib.file_digest(base.open('rb'),'sha256').hexdigest()==best['checkpoint_sha256']
target=r/'best-math.ckpt'
if not (r/'math-best.json').exists():
 target.unlink(missing_ok=True);os.link(base,target);atomic_json(r/'math-best.json',best)
if not (r/'math-general-baseline.json').exists():
 baseline_loss=general_loss('math-general-baseline',base)
 best=json.loads((r/'math-best.json').read_text());best['general_text_loss']=baseline_loss;atomic_json(r/'math-best.json',best)
print('BALANCED CONTINUATION',json.dumps(recipe),flush=True)
stage=max([int(p.stem.split('-')[-1]) for p in r.glob('stage-*.json')]+[0]);goal=50000
# Continue the same optimizer and RNG. Set the new cosine schedule to approach the previous LR smoothly.
lr=.0011;failures=0
while time.time()<deadline-60:
 stage+=1
 minutes=min(20,max(1,(deadline-time.time()-15)/60))
 args=['./cxxgpt-gpu','train','--gpu','--matmul','bf16','--muon','--resume','--out','runs/fresh-eu.ckpt','--data',data,'--val','data/gsm8k-dev.tok','--bpe','data/tok.bpe','--seq','512','--batch','16','--accum','8','--sampler','qa','--threads','8','--clip','1.0','--seed','1','--steps',str(goal),'--lr',str(lr),'--min-lr-frac','0.1','--warmup','500','--eval-every','250','--eval-batches','8','--save-every','250','--log-every','25','--time-limit',str(minutes),'--log','runs/math-focus.csv','--load-log','runs/math-focus.load.csv']
 state={'phase':'training','priority':'balanced general learning and math accuracy','stage':stage,'source_training_tokens':manifest['splits']['train']['tokens']+gsm_tokens+replay_tokens,'args':args,'metrics_log':'math-focus.csv','training_stop_utc':config['training_stop_utc'],'measured_tokens_per_second':94393.6667}
 status(**state);atomic_json(r/f'stage-{stage:03d}.json',state)
 with (r/'math-focus.log').open('a') as log:
  p=subprocess.Popen(args,cwd=root,stdout=log,stderr=subprocess.STDOUT);(r/'trainer.pid').write_text(str(p.pid))
  while p.poll() is None:
   if time.time()>=deadline:
    p.send_signal(signal.SIGINT)
    try:p.wait(timeout=180)
    except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=30)
    break
   time.sleep(3)
 if p.returncode:
  failures+=1;lr*=.5
  if failures>2:status(phase='failed',priority='balanced general learning and math accuracy',reason='repeated math continuation failure');raise SystemExit(p.returncode)
 else:failures=0
 status(phase='benchmark',priority='balanced general learning and math accuracy',stage=stage,metrics_log='math-focus.csv',source_training_tokens=state['source_training_tokens'])
 evaluate(f'math-stage-{stage:03d}')
 text=(r/'math-focus.log').read_text();steps=re.findall(r'at step (\d+)',text)
 if steps and int(steps[-1])>=goal:goal=int(steps[-1])+max(100,int(.85*94394*(deadline-time.time())/65536))
status(phase='benchmark',priority='balanced general learning and math accuracy',metrics_log='math-focus.csv',reason='final math assessment')
# Final dev scores select a checkpoint; the reserved official test is scored once after selection.
if time.time()<deadline+150:evaluate('math-final-dev')
best=json.loads((r/'math-best.json').read_text());best['selection_frozen_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();atomic_json(r/'math-best.json',best)
status(phase='finished',priority='balanced general learning and math accuracy',metrics_log='math-focus.csv',checkpoint='runs/fresh-eu.ckpt',best_checkpoint='runs/best-math.ckpt',best_dev=best)
print('MATH-FOCUSED TRAINING FINISHED',flush=True)
