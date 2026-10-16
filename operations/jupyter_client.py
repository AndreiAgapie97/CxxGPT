"""Authenticated Jupyter transport for this training Pod; never prints tokens."""
import base64
import datetime
import json
import re
import time
from pathlib import Path
import urllib.error
import urllib.request
import uuid
from websockets.sync.client import connect

ROOT = Path('/workspace/cxxgpt-ops')

class Client:
    def __init__(self):
        self.run = json.loads((ROOT/'run.json').read_text())
        self.base = f"https://{self.run['pod_id']}-8888.proxy.runpod.net"
        self.token = (ROOT/'jupyter-token').read_text()
        self.headers = {'Authorization':'token '+self.token,'User-Agent':'CxxGPT-cloud-client/1'}

    def request(self, path, method='GET', body=None, timeout=90):
        headers = self.headers.copy()
        if body is not None:
            body = json.dumps(body).encode()
            headers['Content-Type']='application/json'
        req = urllib.request.Request(self.base+path, data=body, headers=headers, method=method)
        with urllib.request.urlopen(req,timeout=timeout) as r:
            raw=r.read()
            return json.loads(raw) if raw else None

    def mkdir(self,path):
        try:
            return self.request('/api/contents/'+path,'PUT',{'type':'directory'})
        except urllib.error.HTTPError as e:
            if e.code != 409: raise

    def put(self,local,remote):
        # The protocol supports numbered chunks; commit on chunk=-1.
        size=Path(local).stat().st_size
        with open(local,'rb') as f:
            number=1
            while True:
                data=f.read(8<<20)
                last=f.tell()==size
                body={'type':'file','format':'base64','content':base64.b64encode(data).decode()}
                # A single final chunk appends in Jupyter; ordinary PUT overwrites.
                if not (number==1 and last):body['chunk']=-1 if last else number
                self.request('/api/contents/'+remote,'PUT',body,timeout=120)
                if last:break
                number+=1
        print('Uploaded',remote,size,'bytes',flush=True)

    def download(self,remote,local):
        local=Path(local);local.parent.mkdir(parents=True,exist_ok=True)
        partial=Path(str(local)+'.part')
        total=None;last_report=0;failures=0
        while total is None or (partial.stat().st_size if partial.exists() else 0)<total:
            offset=partial.stat().st_size if partial.exists() else 0
            req=urllib.request.Request(self.base+'/files/'+remote,headers={**self.headers,'Range':f'bytes={offset}-{offset+(8<<20)-1}'})
            try:
                with urllib.request.urlopen(req,timeout=45) as r:
                    cr=r.headers.get('Content-Range')
                    if r.status==206:
                        start,end,total=map(int,re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',cr).groups())
                        if start!=offset:raise RuntimeError('Wrong byte range returned')
                        mode='ab'
                    else:
                        if offset:raise RuntimeError('Server refused resume byte range')
                        total=int(r.headers['Content-Length']);mode='wb'
                    with open(partial,mode) as f:
                        for block in iter(lambda:r.read(1<<20),b''):f.write(block)
                failures=0
            except (OSError,urllib.error.URLError) as e:
                failures+=1
                if isinstance(e,urllib.error.HTTPError) and e.code==416:
                    match=re.fullmatch(r'bytes \*/(\d+)',e.headers.get('Content-Range',''))
                    if match and int(match.group(1))==offset:
                        total=offset;partial.touch(exist_ok=True);break
                if failures>=5:raise
                print('Download retry:',type(e).__name__,flush=True);time.sleep(3)
            if partial.exists() and partial.stat().st_size-last_report>=256<<20:
                print('Download progress',remote,partial.stat().st_size,'/',total,flush=True)
                last_report=partial.stat().st_size
        partial.replace(local)
        print('Downloaded',remote,local.stat().st_size,'bytes',flush=True)

    def download_parallel(self,remote,local):
        """Copy an immutable checkpoint with four resumable range workers."""
        import concurrent.futures,os,threading
        local=Path(local);local.parent.mkdir(parents=True,exist_ok=True)
        partial=Path(str(local)+'.part');ledger=Path(str(partial)+'.ranges.json')
        req=urllib.request.Request(self.base+'/files/'+remote,headers={**self.headers,'Range':'bytes=0-0'})
        with urllib.request.urlopen(req,timeout=45) as r:
            match=re.fullmatch(r'bytes 0-0/(\d+)',r.headers.get('Content-Range',''))
            if r.status!=206 or not match:raise RuntimeError('Range support required')
            total=int(match.group(1));r.read()
        chunk=8<<20;count=(total+chunk-1)//chunk
        if ledger.exists():
            state=json.loads(ledger.read_text())
            if state['total']!=total or state['remote']!=remote:raise RuntimeError('Snapshot range ledger mismatch')
            done=set(state['done'])
        else:
            previous=partial.stat().st_size if partial.exists() else 0
            if previous>total:raise RuntimeError('Partial checkpoint exceeds snapshot')
            done=set(range(count if previous==total else previous//chunk))
        lock=threading.Lock()
        def save():
            temp=ledger.with_suffix('.tmp')
            temp.write_text(json.dumps({'total':total,'remote':remote,'done':sorted(done)}));temp.replace(ledger)
        save()
        fd=os.open(partial,os.O_RDWR|os.O_CREAT,0o600);os.ftruncate(fd,total)
        def fetch(index):
            start=index*chunk;end=min(total,start+chunk)-1
            for attempt in range(5):
                try:
                    q=urllib.request.Request(self.base+'/files/'+remote,headers={**self.headers,'Range':f'bytes={start}-{end}'})
                    with urllib.request.urlopen(q,timeout=60) as response:
                        expected=f'bytes {start}-{end}/{total}'
                        valid=(response.status==206 and response.headers.get('Content-Range')==expected) or (response.status==200 and start==0 and end==total-1)
                        if not valid:raise RuntimeError('Wrong checkpoint range')
                        data=response.read()
                    if len(data)!=end-start+1:raise OSError('Short checkpoint range')
                    view=memoryview(data);offset=start
                    while view:
                        written=os.pwrite(fd,view,offset);offset+=written;view=view[written:]
                    with lock:
                        done.add(index);save()
                        if len(done)%32==0:print('Parallel checkpoint progress',len(done),'/',count,'chunks',flush=True)
                    return
                except (OSError,urllib.error.URLError):
                    if attempt==4:raise
                    time.sleep(2)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                for _ in pool.map(fetch,[i for i in range(count) if i not in done]):pass
            os.fsync(fd)
        finally:os.close(fd)
        if len(done)!=count:raise RuntimeError('Missing checkpoint ranges')
        partial.replace(local);ledger.unlink(missing_ok=True)
        print('Parallel checkpoint copied',total,'bytes',flush=True)

    def kernel(self):
        path=ROOT/'kernel-id'
        if path.exists():return path.read_text()
        kernel=self.request('/api/kernels','POST',{'name':'python3'})
        path.write_text(kernel['id'])
        return kernel['id']

    def execute(self,code,timeout=120):
        kid=self.kernel();session=uuid.uuid4().hex;mid=uuid.uuid4().hex
        uri=self.base.replace('https://','wss://')+'/api/kernels/'+kid+'/channels?session_id='+session
        message={'header':{'msg_id':mid,'username':'cxxgpt','session':session,'msg_type':'execute_request',
                 'version':'5.3','date':datetime.datetime.now(datetime.timezone.utc).isoformat()},
                 'parent_header':{},'metadata':{},'channel':'shell',
                 'content':{'code':code,'silent':False,'store_history':False,'user_expressions':{},'allow_stdin':False,'stop_on_error':True}}
        outputs=[];error=None
        with connect(uri,additional_headers=self.headers,origin=self.base,open_timeout=30,max_size=32<<20) as ws:
            ws.send(json.dumps(message))
            while True:
                raw=ws.recv(timeout=timeout)
                if isinstance(raw,bytes):continue
                item=json.loads(raw)
                if item.get('parent_header',{}).get('msg_id')!=mid:continue
                kind=item['header']['msg_type'];content=item['content']
                if kind=='stream':outputs.append(content['text'])
                elif kind in ('execute_result','display_data'):
                    outputs.append(content.get('data',{}).get('text/plain',''))
                elif kind=='error':error=content.get('evalue','Kernel error')
                elif kind=='status' and content['execution_state']=='idle':break
        text=''.join(outputs)
        if text: print(text,end='' if text.endswith('\n') else '\n',flush=True)
        if error:raise RuntimeError(error)
        return text

if __name__=='__main__':
    import sys
    client=Client()
    if sys.argv[1]=='exec':client.execute(Path(sys.argv[2]).read_text(),timeout=300)
    elif sys.argv[1]=='put':client.put(sys.argv[2],sys.argv[3])
    elif sys.argv[1]=='get':client.download(sys.argv[2],sys.argv[3])
    elif sys.argv[1]=='status':print(json.dumps(client.request('/api/status')))
