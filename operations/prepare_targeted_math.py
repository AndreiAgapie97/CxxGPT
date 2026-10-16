"""Build verified-generator word problems; never uses held-out answers for training."""
import datetime,hashlib,json,pathlib,re,struct,subprocess
root=pathlib.Path('/workspace/CxxGPT');out=root/'data/targeted-math';out.mkdir(exist_ok=True)
bin=str(root/'cxxgpt-gpu');tok=str(root/'data/tok.bpe')
def run(args):subprocess.run(args,cwd=root,check=True)
def unescape(s):
 esc={'n':'\n','t':'\t','\\':'\\'};out=[];i=0
 while i<len(s):
  if s[i]=='\\' and i+1<len(s):out.append(esc[s[i+1]]);i+=2
  else:out.append(s[i]);i+=1
 return ''.join(out)
def grams(s):
 words=re.findall(r'[a-z0-9]+',s.lower())
 return {' '.join(words[i:i+13]) for i in range(len(words)-12)}
held=set()
for line in (out/'heldout.jsonl').read_text().splitlines():held.update(grams(json.loads(line)['question']))
records={}
for split,seed,count,style in [('dev',202610161,1000,'test'),('train',202610160,1000000,'train')]:
 prefix=out/('word-'+split)
 if prefix.with_suffix('.tok').exists():continue
 raw=out/('generated-'+split)
 run([bin,'gen-word','--out',str(raw),'--count',str(count),'--seed',str(seed),'--split',style,'--notes','off'])
 kept=dropped=0
 with raw.with_suffix('.txt').open() as f,prefix.with_suffix('.jsonl').open('w') as dest:
  for line in f:
   task,difficulty,q,steps,answer=line.rstrip('\n').split('\t');q=unescape(q);steps=unescape(steps);answer=unescape(answer)
   if split=='train' and grams(q+' '+steps) & held:dropped+=1;continue
   dest.write(json.dumps({'question':q,'answer':steps+'\n#### '+answer},ensure_ascii=False)+'\n');kept+=1
 run([bin,'prep-qa','--in',str(prefix.with_suffix('.jsonl')),'--out',str(prefix),'--bpe',tok,'--task','word'])
 if split=='dev':
  # Keep synthetic dev separate too, including near-copies.
  for line in prefix.with_suffix('.jsonl').read_text().splitlines():held.update(grams(json.loads(line)['question']))
 with prefix.with_suffix('.tok').open('rb') as f:
  magic,vocab,masked,tokens=struct.unpack('<8sIIQ',f.read(24))
 assert magic==b'CXTOKEN1' and masked==1
 records[split]={'seed':seed,'questions':kept,'dropped_near_heldout':dropped,'tokens':tokens,'sha256':hashlib.file_digest(prefix.with_suffix('.tok').open('rb'),'sha256').hexdigest()}
 for path in [raw.with_suffix('.text'),raw.with_suffix('.txt')]:path.unlink(missing_ok=True)
 print('TARGETED MATH READY',split,records[split],flush=True)
metadata={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'generator':'CxxGPT gen-word; exact arithmetic; no calculator notes','source':'locally generated from repository templates, not an externally downloaded dataset','generator_source_sha256':hashlib.file_digest((root/'src/wordgen.h').open('rb'),'sha256').hexdigest(),'tokenizer_sha256':hashlib.file_digest(pathlib.Path(tok).open('rb'),'sha256').hexdigest(),'split':'generator train/test sentence templates; distinct seeds; train rejects 13-word overlaps with GSM8K dev/test and generated word dev','splits':records}
(out/'manifest.json').write_text(json.dumps(metadata,indent=2))
print('MATH PREPARATION COMPLETE',flush=True)
