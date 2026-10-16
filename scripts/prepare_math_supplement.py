#!/usr/bin/env python3
"""Add human-written MIT-licensed GSM8K solutions to the math sampling file.

Keep the 500-question dev split and original test split out of training; remove
training rows sharing a 13-word run with a held-out question. Preserve GSM8K's
answer-only loss masks. The math subset is a weighted sample of existing corpus
documents, not additional unique tokens toward the 10B target.
"""
import gzip
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

ROOT=Path(__file__).resolve().parents[1]
CORPUS=ROOT/'data/licensed-10b'
GSM=ROOT/'data/fresh-pilot/gsm8k'
HEADER=struct.Struct('<8sIIQ')


def runs(text):
    words=re.findall(r'[a-z0-9]+',text.lower())
    return {tuple(words[i:i+13]) for i in range(len(words)-12)}


def main():
    held=set()
    for name in ('dev','test'):
        for line in (GSM/f'{name}.jsonl').read_text().split('\n'):
            if not line.strip():continue
            held.update(runs(json.loads(line)['question']))
    kept=dropped=0
    filtered=CORPUS/'gsm8k-clean-train.jsonl'
    with open(filtered,'w',encoding='utf-8') as out:
        for line in (GSM/'train.jsonl').read_text().split('\n'):
            if not line.strip():continue
            record=json.loads(line)
            if runs(record['question']+' '+record['answer']) & held:
                dropped+=1
                continue
            out.write(json.dumps(record,ensure_ascii=False)+'\n');kept+=1
    subprocess.run([str(ROOT/'cxxgpt'),'prep-qa','--in',str(filtered),'--out',str(CORPUS/'gsm8k-train'),
        '--bpe',str(CORPUS/'tok.bpe'),'--task','gsm8k'],check=True)
    manifest=json.loads((CORPUS/'manifest.json').read_text())
    raw=CORPUS/'math-base.tokens'
    count=0
    with open(raw,'wb') as out:
        for job in manifest['completed']:
            with gzip.open(CORPUS/'blocks'/job/'attribution.jsonl.gz','rt') as f:
                for line in f:
                    record=json.loads(line)
                    if record['split']!='train' or record.get('math_label','general')=='general' or count>=50_000_000:
                        continue
                    with open(CORPUS/'blocks'/job/record['token_file'],'rb') as tokens:
                        tokens.seek(24+record['token_offset']*2)
                        data=tokens.read(record['tokens']*2)
                        if len(data)!=record['tokens']*2:raise RuntimeError('Truncated math source')
                    out.write(data);count+=record['tokens']
    with open(CORPUS/'gsm8k-train.tok','rb') as f:
        magic,vocab,has_mask,qa_count=HEADER.unpack(f.read(24))
        assert magic==b'CXTOKEN1' and has_mask==1
        qa_tokens=f.read(qa_count*2);qa_masks=f.read(qa_count)
        assert len(qa_masks)==qa_count and not f.read(1)
    partial=CORPUS/'math-boost.tok.part'
    with open(partial,'wb') as out:
        out.write(HEADER.pack(b'CXTOKEN1',vocab,1,count+qa_count))
        with open(raw,'rb') as f:
            for chunk in iter(lambda:f.read(4<<20),b''):out.write(chunk)
        out.write(qa_tokens)
        for i in range(0,count,4<<20):out.write(b'\1'*min(4<<20,count-i))
        out.write(qa_masks)
    partial.replace(CORPUS/'math-boost.tok');raw.unlink()
    info={'gsm8k_license':'MIT; OpenAI 2021','training_questions':kept,'removed_near_heldout_questions':dropped,
        'gsm8k_tokens':qa_count,'existing_corpus_math_tokens':count,
        'combined_tokens':count+qa_count,'weighted_subset_not_counted_as_additional_unique_corpus':True,
        'math_boost_sha256':hashlib.sha256((CORPUS/'math-boost.tok').read_bytes()).hexdigest()}
    (CORPUS/'math-supplement.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))


if __name__=='__main__':main()
