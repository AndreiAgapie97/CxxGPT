"""Publish a small, non-secret status file beside the recovery archive."""
import datetime
import json
from pathlib import Path
import time
from jupyter_client import Client,ROOT

client=Client()
out=Path('/workspace/library-files/CxxGPT-status.md')
stop=datetime.datetime.fromisoformat(client.run['hard_stop_utc']).timestamp()+600
while time.time()<stop:
    now=datetime.datetime.now(datetime.timezone.utc)
    try:
        m=json.loads(Path('/workspace/CxxGPT/data/licensed-10b/manifest.json').read_text())
        math=sum(v for j in m['completed'].values() for k,v in j.get('math_training_tokens',{}).items() if k!='general')
        lines=['# CxxGPT cloud training status','',f'Updated: {now.isoformat()}','',
            '- GPU: RTX 4090, EU-RO-1 (Romania).',
            '- Model: 267.480M total / 97.610M active parameters; BF16 MoE with Muon.',
            '- Budget: $10 total; planned compute/storage through hard stop about $9.25.',
            '- Training stop: 2026-10-16 09:10 CEST (07:10 UTC); independent GPU stop: 09:45 CEST (07:45 UTC).',
            f"- Corpus assembled: {m['tokens'].get('train',0):,} unique training tokens / target 10,000,000,000.",
            f'- Math/proof-labeled corpus tokens: {math:,} (includes a proof-text heuristic).',
            '- Human math supplement: 6,966 MIT-licensed GSM8K training questions; dev/test excluded.','']
        try:
            status=client.request('/api/contents/workspace/CxxGPT/runs/training-status.json?content=1')
            state=json.loads(status['content'])
            lines.extend([f"Training state: {state.get('phase')}; stage {state.get('stage','-')}.",
                f"Unique data tokens available at stage start: {state.get('source_training_tokens','-'):,}." if isinstance(state.get('source_training_tokens'),int) else '',
                ''])
            metrics=state.get('metrics_log','fresh-eu.csv')
            rows=client.request('/api/contents/workspace/CxxGPT/runs/'+metrics+'?content=1')['content'].splitlines()
            try:
                scored=json.loads(client.request('/api/contents/workspace/CxxGPT/runs/math-latest-eval.json?content=1')['content'])
                lines.extend([f"Latest reserved math dev score: {scored.get('correct')}/{scored.get('total')}; {scored.get('utc')}.",f"Fixed held-out book loss at that checkpoint: {scored.get('general_text_loss','not recorded')}.",''])
                best=json.loads(client.request('/api/contents/workspace/CxxGPT/runs/math-best.json?content=1')['content'])
                lines.extend([f"Best preserved reserved math dev score: {best.get('correct')}/{best.get('total')}; held-out book loss {best.get('general_text_loss','not recorded')}.",
                              'Baseline: 6/500 (1.2%); fixed held-out book loss 2.5745.',''])
                marker=ROOT/'published-eval-name'
                name=scored.get('name')
                if name and (not marker.exists() or marker.read_text()!=name):
                    raw=client.request('/api/contents/workspace/CxxGPT/runs/'+name+'.answers.jsonl?content=1')['content']
                    records=[json.loads(x) for x in raw.splitlines()][1:]
                    dev=Path('/workspace/CxxGPT/data/licensed-10b/gsm8k-dev.txt').read_text().splitlines()
                    destination=Path('/workspace/library-files/CxxGPT-math-latest-answers.jsonl')
                    partial=destination.with_suffix('.jsonl.part')
                    with partial.open('w') as f:
                        f.write(json.dumps({'evaluation':scored,'baseline_correct':6,'baseline_total':500,'split':'reserved development; test is not used for selection'})+'\n')
                        for item in records:
                            q=dev[item['problem']].split('\t');f.write(json.dumps({**item,'question':q[2]},ensure_ascii=False)+'\n')
                    partial.replace(destination);marker.write_text(name)
                    progress=Path('/workspace/library-files/CxxGPT-math-progress.md')
                    progress.write_text(f"# CxxGPT balanced-training progress\n\nUpdated: {scored.get('utc')}.\n\nBaseline: 6/500 (1.2%) reserved GSM8K dev; held-out book loss 2.5745.\n\nLatest: **{scored.get('correct')}/{scored.get('total')}** on the same reserved dev questions, greedy, no calculator, max 400 generated tokens. Held-out book loss: **{scored.get('general_text_loss')}**.\n\nTraining uses approximately 53% worked math and 47% general text by input sampling weight. Loss masks change the fraction of supervised targets. Dev scores guide selection; the original test is reserved until selection freezes.\n\nAll latest responses are in CxxGPT-math-latest-answers.jsonl.\n")
            except Exception:pass
            numeric=[row for row in rows if row and row[0].isdigit()]
            if numeric:
                last=numeric[-1].split(',');step=int(last[0])
                lines.extend([f'Latest logged optimizer step: {step:,}.',f'Approximately {step*65536:,} sampled training tokens (sampling repetitions included).',f'Latest training loss: {last[1]}.',''])
        except Exception as e:lines.extend(['Live Pod read: '+type(e).__name__+'.',''])
        proof=ROOT/'recovery/logs/checkpoint-verification.json'
        if proof.exists():
            info=json.loads(proof.read_text())
            lines.extend(['Off-Pod checkpoint SHA-256 verified: '+info['sha256'],
                          'Last verified copy: '+info['verified_utc']+'.',''])
        else:lines.extend(['First full checkpoint copy is still in progress; the existing archive currently contains source and setup artifacts.',''])
        if (ROOT/'completed.json').exists():
            lines.extend(['Training/recovery supervisor has completed shutdown and cleanup.',''])
        lines.extend(['Recovery archive: `CxxGPT-recovery.tar.gz` (beside this status file).',
                      'The archive contains source, Git history, tokenizer, metadata, math supplement, logs and the most recently verified copied checkpoint; inspect RECOVERY.json for its exact contents.',
                      'The bulk token corpus is retained separately in the cloud workspace.'])
        temp=out.with_suffix('.md.part');temp.write_text('\n'.join(lines)+'\n');temp.replace(out)
        if (ROOT/'completed.json').exists():break
    except Exception as e:print('Status update:',type(e).__name__,str(e)[:300],flush=True)
    for _ in range(4):time.sleep(30)
