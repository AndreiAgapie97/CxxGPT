#!/usr/bin/env python3
"""Package the working source plus explicitly selected training artifacts.

Credentials and the bulk training corpus are deliberately excluded. The corpus
manifest and source inventory allow downloads to be reproduced independently.
"""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('/workspace/library-files/CxxGPT-recovery.tar.gz'))
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--logs',type=Path)
    parser.add_argument('--best-checkpoint',type=Path)
    args=parser.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    paths=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=ROOT).split(b'\0')
    partial=args.out.with_name(args.out.name+'.part')
    with tarfile.open(partial,'w:gz',compresslevel=1) as archive:
        for raw in paths:
            if not raw:continue
            relative=Path(raw.decode())
            if relative.suffix in ('.log','.tmp','.part') or relative.parts[0] in ('data','runs','.venv','.git'):
                continue
            path=ROOT/relative
            if path.is_file():archive.add(path,arcname=str(Path('CxxGPT')/relative))
        for name in ('manifest.json','tok.bpe','inventory.json','math-boost.tok','math-supplement.json','gsm8k-provenance.json','processing-code-sha256.json'):
            path=ROOT/'data/licensed-10b'/name
            if path.is_file():archive.add(path,arcname='CxxGPT/data/licensed-10b/'+name)
        path=ROOT/'data/licensed-10b/licenses'
        if path.exists():archive.add(path,arcname='CxxGPT/data/licensed-10b/licenses')
        for path in (ROOT/'data/licensed-10b').glob('gsm8k-*'):
            if path.is_file() and path.suffix in ('.tok','.txt','.jsonl'):
                archive.add(path,arcname='CxxGPT/data/licensed-10b/'+path.name)
        path=ROOT/'data/licensed-10b/cards'
        if path.exists():archive.add(path,arcname='CxxGPT/data/licensed-10b/cards')
        # Only these non-secret operational files are included.
        ops=Path('/workspace/cxxgpt-ops')
        history=ops/'repository-history.bundle'
        if history.is_file():archive.add(history,arcname='CxxGPT/repository-history.bundle')
        for name in ('train_controller.py','resume_controller.py','tune_batch.py','evaluate_current.py','evaluate_dev500.py','math_focus_controller.py','prepare_targeted_math.py','final_math_evaluation.py','math-heldout.jsonl','pod_budget_guard.py','keep_workers.py','status_reporter.py','sync_corpus.py','sync_daemon.py','build_daemon.py','jupyter_client.py','budget_guard.py','recovery_supervisor.py','mcp_bridge.py','run.json'):
            path=ops/name
            if path.is_file():archive.add(path,arcname='CxxGPT/operations/'+name)
        for job in sorted((ROOT/'data/licensed-10b/blocks').glob('*')):
            path=job/'attribution.jsonl.gz'
            if path.is_file() and not job.name.endswith('.work'):
                archive.add(path,arcname='CxxGPT/data/licensed-10b/blocks/'+job.name+'/'+path.name)
        if args.checkpoint and args.checkpoint.exists():
            archive.add(args.checkpoint,arcname='CxxGPT/runs/fresh-eu.ckpt')
        if args.best_checkpoint and args.best_checkpoint.exists():
            archive.add(args.best_checkpoint,arcname='CxxGPT/runs/best-math-inference.ckpt')
        if args.logs and args.logs.exists():
            for path in args.logs.iterdir():
                if path.is_file() and path.suffix in ('.log','.csv','.json','.jsonl','.txt','.exit'):
                    archive.add(path,arcname='CxxGPT/runs/'+path.name)
        description={
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'includes_working_tree_changes':True,
            'includes_git_history_bundle':history.is_file(),
            'includes_checkpoint':bool(args.checkpoint and args.checkpoint.exists()),
            'bulk_corpus_included':False,
            'includes_best_math_inference_checkpoint':bool(args.best_checkpoint and args.best_checkpoint.exists()),
            'best_math_restore':'best-math-inference.ckpt has bit-identical weights/router biases, without optimizer moments. Use sample/eval-reasoning; use --init for further training. fresh-eu.ckpt retains full latest optimizer/RNG state for --resume.',
            'restore':'Extract this archive. Rebuild with make (CPU) or make gpu (CUDA). Use runs/fresh-eu.ckpt with data/licensed-10b/tok.bpe. Corpus source revisions, licenses, exact counts, and checksums are in data/licensed-10b/manifest.json.',
        }
        content=(json.dumps(description,indent=2)+'\n').encode()
        member=tarfile.TarInfo('CxxGPT/RECOVERY.json');member.size=len(content)
        archive.addfile(member,io.BytesIO(content))
    partial.replace(args.out)
    print(args.out,args.out.stat().st_size,'bytes')


if __name__=='__main__':main()
