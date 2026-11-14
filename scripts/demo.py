"""A real offline SFT → preference learning → evaluation demonstration."""
import json
import argparse
from datetime import datetime
import subprocess
import sys
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--output',default='artifacts/demo-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
parser.add_argument('--public-results')
args=parser.parse_args()

print('DPO / APPLE SILICON · actual offline engineering demo',flush=True)
print('Synthetic fixtures + a small randomly initialized transformer. Not paper benchmark results.',flush=True)
subprocess.run([sys.executable,'-m','dpo_lab','doctor'],check=True)
subprocess.run([sys.executable,'-u','-m','dpo_lab','run','--config','experiments/smoke.json',
                '--output',args.output,'--device','mps'],check=True)
for task,beta in [('sentiment','0.1'),('summarization','0.5'),('dialogue','0.1')]:
    root=(Path(args.output)/'seed_0')/task/f'dpo_beta_{beta}'/'evaluation'
    result=json.loads((root/'summary.json').read_text())
    samples=[json.loads(line) for line in (root/'samples.jsonl').read_text().splitlines()]
    print(f'\n=== SAVED RESULT: {task} ===',flush=True)
    print(f'Pair accuracy: {result["pairs"]["preference_accuracy"]["mean"]:.3f}',flush=True)
    print('Prompt: '+samples[0]['prompt'].replace('\n',' '),flush=True)
    print('Generated: '+samples[0]['completion'].replace('\n',' '),flush=True)
print(f'\nCompleted. Checkpoints, data, metrics and samples saved in {args.output}.',flush=True)

if args.public_results:
    subprocess.run([sys.executable,'-u','scripts/demo_public.py','--results',args.public_results],check=True)
