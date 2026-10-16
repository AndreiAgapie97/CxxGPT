"""Keep verified off-Pod checkpoints and stop/remove only this run's resources."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import urllib.error
from jupyter_client import Client,ROOT

RECOVERY=ROOT/'recovery'
LOGS=RECOVERY/'logs'
LOGS.mkdir(parents=True,exist_ok=True)
client=Client()
config=client.run
backup_start=datetime.datetime.fromisoformat(config['backup_start_utc']).timestamp()
hard_stop=datetime.datetime.fromisoformat(config['hard_stop_utc']).timestamp()
last_backup=0
ever_saved=False
finished_backup=False
proof=LOGS/'checkpoint-verification.json'
archive=Path('/workspace/library-files/CxxGPT-recovery.tar.gz')
if proof.exists() and archive.exists() and archive.with_name(archive.name+'.sha256').exists():
    previous=json.loads(proof.read_text())
    last_backup=datetime.datetime.fromisoformat(previous['verified_utc']).timestamp()
    ever_saved=True

def rpc(tool,args):
    p=subprocess.run(['python','/workspace/cxxgpt-ops/mcp_bridge.py',tool,json.dumps(args)],capture_output=True,text=True,timeout=120)
    if p.returncode:raise RuntimeError(p.stderr[-1000:])
    r=json.loads(p.stdout)
    if r.get('isError'):raise RuntimeError(str(r.get('content'))[:1000])
    for c in r.get('content',[]):
        if c.get('type')=='text':
            try:return json.loads(c['text'])
            except ValueError:return c['text']
    return r

def logs():
    names={'build.log','build.exit','gpu-tests.log','gpu-tests.exit','targeted-gpu.log',
           'targeted-gpu.exit','probe.log','controller.log','training-status.json','fresh-eu.log',
           'fresh-eu.csv','fresh-eu.load.csv','batch-choice.json','batch-tuning.log'}
    listing=client.request('/api/contents/workspace/CxxGPT/runs?content=1')
    for item in listing['content']:
        name=item['name']
        if name.startswith(('math-','eval-','final-test','export-best','targeted-math')) and Path(name).suffix in ('.log','.csv','.json','.jsonl'):
            names.add(name)
    for name in sorted(names):
        try:
            raw=client.request('/api/contents/workspace/CxxGPT/runs/'+name+'?content=1')
            if raw.get('format')=='base64':
                import base64
                (LOGS/name).write_bytes(base64.b64decode(raw['content']))
            else:(LOGS/name).write_text(raw['content'])
        except urllib.error.HTTPError as e:
            if e.code!=404:raise
    try:
        raw=client.request('/api/contents/workspace/CxxGPT/data/targeted-math/manifest.json?content=1')
        (LOGS/'targeted-math-manifest.json').write_text(raw['content'])
    except urllib.error.HTTPError as e:
        if e.code!=404:raise
    (LOGS/'run.json').write_text(json.dumps(config,indent=2))

def backup(final=False):
    global ever_saved
    checkpoint=RECOVERY/'fresh-eu.ckpt'
    expected=RECOVERY/'snapshot-expected.json'
    if final and (not expected.exists() or not json.loads(expected.read_text()).get('final')):
        Path(str(checkpoint)+'.part.ranges.json').unlink(missing_ok=True)
        Path(str(checkpoint)+'.part').unlink(missing_ok=True)
        expected.unlink(missing_ok=True)
    if Path(str(checkpoint)+'.part').exists() and expected.exists():
        info=json.loads(expected.read_text())
    else:
        text=client.execute("import pathlib,os,hashlib,json; r=pathlib.Path('/workspace/CxxGPT/runs'); p=r/'fresh-eu.ckpt'; s=r/'backup-snapshot.ckpt'; s.unlink(missing_ok=True); os.link(p,s); print(json.dumps({'bytes':s.stat().st_size,'sha256':hashlib.file_digest(open(s,'rb'),'sha256').hexdigest()}))")
        info=json.loads(text.strip().splitlines()[-1])
        info['final']=final and time.time()>=backup_start
        info['snapshot_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        expected.write_text(json.dumps(info))
    # The verified archive keeps the previous checkpoint while the next one
    # downloads. Avoid holding two raw 3.2GB checkpoints on the 32GB workspace.
    if ever_saved and checkpoint.exists() and not Path(str(checkpoint)+'.part').exists():
        checkpoint.unlink()
    client.download_parallel('workspace/CxxGPT/runs/backup-snapshot.ckpt',checkpoint)
    if not final:info['final']=False
    with open(checkpoint,'rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    if checkpoint.stat().st_size!=info['bytes'] or digest!=info['sha256']:
        raise RuntimeError('Off-Pod checkpoint verification failed')
    (LOGS/'checkpoint-verification.json').write_text(json.dumps({**info,'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
    best_checkpoint=None
    if final:
        # The independently running final evaluator exports a smaller best model
        # with exactly the same weights and biases, omitting optimizer moments.
        evaluation_finished=False
        for _ in range(18):
            try:
                ready=client.request('/api/contents/workspace/CxxGPT/runs/final-eval-ready.json?content=1')
                if json.loads(ready['content']).get('finished'):
                    evaluation_finished=True
                    break
            except urllib.error.HTTPError as e:
                if e.code!=404:raise
            if time.time()>hard_stop-600:break
            time.sleep(10)
        if not evaluation_finished:
            raise RuntimeError('Final evaluation is not ready; retain Pod storage and retry recovery. Independent GPU cutoff remains armed.')
        raw=client.request('/api/contents/workspace/CxxGPT/runs/best-inference.json?content=1')
        best_info=json.loads(raw['content'])
        client.execute("import pathlib,os; r=pathlib.Path('/workspace/CxxGPT/runs'); p=r/'best-math-inference.ckpt'; s=r/'backup-best-inference.ckpt'; s.unlink(missing_ok=True); os.link(p,s)")
        best_checkpoint=RECOVERY/'best-math-inference.ckpt'
        client.download_parallel('workspace/CxxGPT/runs/backup-best-inference.ckpt',best_checkpoint)
        with best_checkpoint.open('rb') as f:best_digest=hashlib.file_digest(f,'sha256').hexdigest()
        if best_checkpoint.stat().st_size!=best_info['bytes'] or best_digest!=best_info['sha256']:
            raise RuntimeError('Off-Pod best math model verification failed')
        (LOGS/'best-checkpoint-verification.json').write_text(json.dumps(best_info,indent=2))
    logs()
    command=['python','/workspace/CxxGPT/scripts/package_recovery.py','--checkpoint',str(checkpoint),'--logs',str(LOGS)]
    if best_checkpoint:command+=['--best-checkpoint',str(best_checkpoint)]
    subprocess.run(command,check=True)
    archive=Path('/workspace/library-files/CxxGPT-recovery.tar.gz')
    with open(archive,'rb') as f:archive_hash=hashlib.file_digest(f,'sha256').hexdigest()
    archive.with_name(archive.name+'.sha256').write_text(archive_hash+'  '+archive.name+'\n')
    ever_saved=True
    expected.unlink(missing_ok=True)
    print('OFF-POD CHECKPOINT AND RECOVERY ARCHIVE VERIFIED',info,flush=True)

def stop_and_cleanup():
    global finished_backup
    # A resumed trainer may have replaced a failed controller during a long backup.
    if time.time()<backup_start:
        current=json.loads(client.request('/api/contents/workspace/CxxGPT/runs/training-status.json?content=1')['content'])
        if current.get('phase') in ('training','benchmark'):
            finished_backup=False
            print('Active training rechecked; cleanup deferred',flush=True)
            return False
    def get_or_missing(tool,id):
        try:return rpc(tool,{'id':id})
        except RuntimeError as e:
            if any(s in str(e).lower() for s in ('not found','not_found','404')):return None
            raise
    pod=get_or_missing('get-pod',config['pod_id'])
    if pod is not None and pod.get('status')!='EXITED':
        rpc('pod-action',{'id':config['pod_id'],'body':{'action':'stop'}})
        for _ in range(15):
            pod=get_or_missing('get-pod',config['pod_id'])
            if pod is None or pod.get('status')=='EXITED':break
            time.sleep(3)
    print('GPU stop readback',pod.get('status') if pod else 'absent',flush=True)
    if pod is not None and pod.get('status')!='EXITED':return False
    if ever_saved and pod is not None:
        try:rpc('delete-pod',{'id':config['pod_id']})
        except RuntimeError as e:
            if 'unexpected end of json' not in str(e).lower():raise
        if get_or_missing('get-pod',config['pod_id']) is not None:
            raise RuntimeError('Pod still present after termination request')
    if ever_saved:
        print('POD DELETION VERIFIED',config['pod_id'],flush=True)
        volume=get_or_missing('get-network-volume',config['volume_id'])
        if volume is not None:
            try:rpc('delete-network-volume',{'id':config['volume_id']})
            except RuntimeError as e:
                if 'unexpected end of json' not in str(e).lower():raise
        if get_or_missing('get-network-volume',config['volume_id']) is not None:
            raise RuntimeError('Storage still present after deletion request')
        print('STORAGE DELETION VERIFIED',config['volume_id'],flush=True)
    return True

while time.time()<hard_stop:
    try:
        if finished_backup:
            if stop_and_cleanup():
                (ROOT/'completed.json').write_text(json.dumps({'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'archive':'/workspace/library-files/CxxGPT-recovery.tar.gz','checkpoint_verified':True},indent=2))
                print('TRAINING AND RECOVERY COMPLETE',flush=True)
                break
            time.sleep(30)
            continue
        raw=client.request('/api/contents/workspace/CxxGPT/runs/training-status.json?content=1')
        state=json.loads(raw['content'])
        updated=datetime.datetime.fromisoformat(state['updated_utc']).timestamp()
        if state.get('phase') in ('training','benchmark') and time.time()-updated>27*60:
            health=client.execute("import pathlib,json; r=pathlib.Path('/workspace/CxxGPT/runs'); p=pathlib.Path('/proc')/(r/'controller.pid').read_text().strip()/'stat'; print(json.dumps({'alive':p.exists() and p.read_text().split()[2]!='Z'}))")
            if not json.loads(health.strip().splitlines()[-1])['alive']:
                state['phase']='failed'
        final=time.time()>=backup_start or state.get('phase') in ('finished','failed')
        if final:
            client.execute("import pathlib,os,signal\nr=pathlib.Path('/workspace/CxxGPT/runs'); p=r/'trainer.pid'\ntry:\n    if p.exists():os.kill(int(p.read_text()),signal.SIGINT)\nexcept ProcessLookupError:\n    pass")
            for _ in range(12):
                time.sleep(10)
                model=client.request('/api/contents/workspace/CxxGPT/runs/training-status.json?content=1')
                if json.loads(model['content']).get('phase') in ('finished','failed'):break
            # Never interrupt a checkpoint writer; a previous periodic save remains recoverable.
            backup(final=True)
            finished_backup=True
            if stop_and_cleanup():
                (ROOT/'completed.json').write_text(json.dumps({'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'archive':'/workspace/library-files/CxxGPT-recovery.tar.gz','checkpoint_verified':True},indent=2))
                print('TRAINING AND RECOVERY COMPLETE',flush=True)
                break
        elif time.time()-last_backup>=3600 and state.get('phase')=='training':
            backup()
            last_backup=time.time()
    except urllib.error.HTTPError as e:
        print('Recovery status:',e.code,flush=True)
    except Exception:
        traceback.print_exc()
    time.sleep(30)
