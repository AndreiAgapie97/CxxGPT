"""On-Pod trainer: stage data refreshes while preserving optimizer and RNG state."""
import datetime
import json
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT=Path('/workspace/CxxGPT')
RUNS=ROOT/'runs'
RUNS.mkdir(exist_ok=True)
config=json.loads((ROOT/'run-budget.json').read_text())
deadline=datetime.datetime.fromisoformat(config['training_stop_utc']).timestamp()
shape=['--dim','768','--layers','12','--heads','12','--experts','8','--topk','2']
batch=16
accum=8
seq=512
lr=1.5e-3

def status(**values):
    values.update(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    p=RUNS/'training-status.json.tmp';p.write_text(json.dumps(values,indent=2));p.replace(RUNS/'training-status.json')

def dataset():
    inv=json.loads((ROOT/'data/inventory.json').read_text())
    files=inv['files']
    # Increase math/science paper exposure without reporting repetitions as new corpus tokens.
    train=[f['path']+('*3' if 'arxiv' in f['source'] else '') for f in files if f['split']=='train']
    train.append('data/math-boost.tok*4')
    val=max((f for f in files if f['split']=='val'),key=lambda f:f['tokens'])['path']
    return ','.join(train),val,sum(f['tokens'] for f in files if f['split']=='train')

def common(data,val):
    return ['./cxxgpt-gpu','train','--gpu','--matmul','bf16','--muon','--data',data,'--val',val,
            '--bpe','data/tok.bpe','--seq',str(seq),'--batch',str(batch),'--accum',str(accum),
            '--threads','8','--clip','1.0','--seed','1']

def launch(args,log_name):
    with open(RUNS/log_name,'a',buffering=1) as log:
        p=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        (RUNS/'trainer.pid').write_text(str(p.pid))
        while p.poll() is None:
            if time.time()>=deadline:
                p.send_signal(signal.SIGINT)
                try:return p.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    p.terminate()
                    return p.wait(timeout=60)
            time.sleep(5)
        return p.returncode

if (RUNS/'targeted-gpu.exit').read_text().strip()!='0':
    raise RuntimeError('The selected BF16 MoE path did not pass its targeted GPU check')
while not (ROOT/'data/inventory.json').exists():
    if time.time()>=deadline:raise SystemExit('Data unavailable before deadline')
    time.sleep(15)

data,val,unique_tokens=dataset()
status(phase='benchmark',source_training_tokens=unique_tokens,shape=shape,batch=batch,accum=accum,seq=seq)
probe=common(data,val)+shape+['--steps','20','--lr',str(lr),'--warmup','20','--eval-every','0',
        '--save-every','0','--log-every','5','--time-limit','3','--out','runs/probe.ckpt']
probe_start=time.time()
rc=launch(probe,'probe.log')
if rc!=0:
    text=(RUNS/'probe.log').read_text()
    if 'out of memory' in text.lower() and batch>4:
        batch=8;accum=16
        rc=launch(common(data,val)+shape+['--steps','20','--lr',str(lr),'--warmup','20','--eval-every','0',
            '--save-every','0','--log-every','5','--time-limit','3','--out','runs/probe.ckpt'],'probe.log')
    if rc!=0:
        status(phase='failed',reason='benchmark failed',returncode=rc)
        raise SystemExit(rc)
speeds=[float(x) for x in re.findall(r'\|\s*([0-9]+) tok/s',(RUNS/'probe.log').read_text())]
if not speeds:raise RuntimeError('Benchmark did not report throughput')
speed=sum(speeds[-3:])/len(speeds[-3:])
# Continue the benchmark's fresh weights/optimizer; schedule long training from their saved state.
(RUNS/'probe.ckpt').replace(RUNS/'fresh-eu.ckpt')
per_step=batch*accum*seq
goal_steps=max(200,20+int(.88*speed*max(0,deadline-time.time())/per_step))
stage=0
failures=0
while time.time()<deadline-60:
    stage+=1
    data,val,unique_tokens=dataset()
    minutes=min(20,max(1,(deadline-time.time()-20)/60))
    args=common(data,val)+['--resume','--out','runs/fresh-eu.ckpt','--steps',str(goal_steps),
        '--lr',str(lr),'--min-lr-frac','0.1','--warmup','500','--eval-every','250',
        '--eval-batches','8','--save-every','250','--log-every','25','--time-limit',str(minutes),
        '--log','runs/fresh-eu.csv','--load-log','runs/fresh-eu.load.csv']
    state={'phase':'training','stage':stage,'source_training_tokens':unique_tokens,'args':args,
           'measured_tokens_per_second':speed,'goal_steps':goal_steps,'training_stop_utc':config['training_stop_utc']}
    status(**state)
    (RUNS/f'stage-{stage:03d}.json').write_text(json.dumps(state,indent=2))
    rc=launch(args,'fresh-eu.log')
    if rc!=0:
        failures+=1
        lr*=.5
        if failures>2:
            status(phase='failed',reason='repeated training failure',returncode=rc)
            raise SystemExit(rc)
    else:
        failures=0
    # If the estimate finished early, extend the existing run rather than pay for an idle GPU.
    log=(RUNS/'fresh-eu.log').read_text()
    steps=re.findall(r'at step (\d+)',log)
    if steps and int(steps[-1])>=goal_steps:
        goal_steps=int(steps[-1])+max(100,int(.85*speed*(deadline-time.time())/per_step))
status(phase='finished',checkpoint='runs/fresh-eu.ckpt',training_stop_utc=config['training_stop_utc'])
print('TRAINING FINISHED; CHECKPOINT SAVED',flush=True)
