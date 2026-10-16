"""Upload only committed token shards and publish an atomic training inventory."""
import datetime
import hashlib
import json
from pathlib import Path
import time
from jupyter_client import Client,ROOT

corpus=Path('/workspace/CxxGPT/data/licensed-10b')
state_path=ROOT/'uploaded.json'
state=json.loads(state_path.read_text()) if state_path.exists() else {'jobs':{},'files':[]}
client=Client()
client.execute("from pathlib import Path; Path('/workspace/CxxGPT/data/corpus').mkdir(parents=True,exist_ok=True)")
# Immutable tokenizer and math supplement were installed atomically at setup.
# Corpus refreshes never rewrite either file while a trainer can load them.
while True:
    manifest=json.loads((corpus/'manifest.json').read_text())
    for job,record in manifest['completed'].items():
        if job in state['jobs']:continue
        client.execute(f"from pathlib import Path; Path('/workspace/CxxGPT/data/corpus/{job}').mkdir(exist_ok=True)")
        for info in record['files']:
            local=corpus/'blocks'/job/info['file']
            relative=f'data/corpus/{job}/{info["file"]}'
            remote='workspace/CxxGPT/'+relative
            if any(f['path']==relative for f in state['files']):continue
            client.put(local,remote+'.upload')
            code=f"import hashlib,os; p='/workspace/CxxGPT/{relative}.upload'; h=hashlib.file_digest(open(p,'rb'),'sha256').hexdigest(); assert h=={info['sha256']!r}, 'Uploaded corpus checksum mismatch'; os.replace(p,p.removesuffix('.upload')); print('TOKEN SHARD VERIFIED', {relative!r})"
            client.execute(code)
            state['files'].append({**info,'path':relative,'source':record['source']})
            state_path.write_text(json.dumps(state,indent=2))
            inventory=ROOT/'training-inventory.json'
            inventory.write_text(json.dumps({'files':state['files'],'math_boost':'data/math-boost.tok','corpus_manifest':manifest},indent=2))
            client.put(inventory,'workspace/CxxGPT/data/inventory.json.upload')
            client.execute("import os; os.replace('/workspace/CxxGPT/data/inventory.json.upload','/workspace/CxxGPT/data/inventory.json')")
        state['jobs'][job]=True
        state_path.write_text(json.dumps(state,indent=2))
        print('CORPUS JOB READY',job,flush=True)
    if manifest['status']=='complete':
        print('FULL CORPUS UPLOAD COMPLETE',manifest['tokens'],flush=True)
        break
    if datetime.datetime.now(datetime.timezone.utc).isoformat()>=client.run['training_stop_utc']:
        break
    time.sleep(30)
