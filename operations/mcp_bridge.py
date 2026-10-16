import json
import subprocess
import sys
import queue
import threading

p = subprocess.Popen(['codex', 'app-server', '--stdio'], stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, stderr=open('/tmp/runpod-mcp-bridge.log', 'a'), text=True)
messages = queue.Queue()
def receive():
    for line in p.stdout:
        try:
            messages.put(json.loads(line))
        except json.JSONDecodeError:
            pass
threading.Thread(target=receive, daemon=True).start()
seq = 0
def send(obj):
    p.stdin.write(json.dumps(obj) + '\n')
    p.stdin.flush()
def request(method, params):
    global seq
    seq += 1
    request_id = seq
    send({'id': request_id, 'method': method, 'params': params})
    while True:
        msg = messages.get(timeout=90)
        if msg.get('id') == request_id:
            if 'error' in msg:
                raise RuntimeError(json.dumps(msg['error']))
            return msg.get('result')
        if 'id' in msg and 'method' in msg:
            raise RuntimeError('Unexpected server request: ' + msg['method'])
try:
    request('initialize', {'clientInfo': {'name':'runpod-readonly-bridge','version':'1'},
                           'capabilities': {'experimentalApi':True}})
    send({'method':'initialized'})
    thread = request('thread/start', {'ephemeral': True, 'cwd':'/workspace/CxxGPT'})
    tid = thread['thread']['id']
    catalog = request('mcpServerStatus/list', {'serverName':'runpod','threadId':tid,'detail':'toolsAndAuthOnly'})
    with open('/tmp/runpod-mcp-catalog.json','w') as f:
        json.dump(catalog,f)
    if len(sys.argv) == 1:
        for server in catalog['data']:
            print(json.dumps({k:v for k,v in server.items() if k not in ('tools','resources','resourceTemplates')}))
            for key, tool in server.get('tools',{}).items():
                if any(s in tool['name'] for s in ('list-pods','list-gpu-types','get-gpu-type')):
                    print(json.dumps({'key':key,**tool}))
    else:
        tool = sys.argv[1]
        arguments = (json.load(open(sys.argv[2][1:])) if sys.argv[2].startswith('@') else json.loads(sys.argv[2])) if len(sys.argv) > 2 else {}
        result = request('mcpServer/tool/call', {'threadId':tid,'server':'runpod','tool':tool,'arguments':arguments})
        print(json.dumps(result))
finally:
    p.terminate()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()
