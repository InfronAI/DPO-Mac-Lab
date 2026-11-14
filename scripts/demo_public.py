"""Show real saved public-data measurements and load each DPO checkpoint for inference."""
import argparse,json,time
from pathlib import Path
import torch
from dpo_lab.models import load_model,load_tokenizer
from dpo_lab.evaluation import generate
from dpo_lab.runtime import read_jsonl,select_device,seed_everything,release_memory
p=argparse.ArgumentParser();p.add_argument('--results',default='results/m1-public');a=p.parse_args();root=Path(a.results)
meta=json.loads((root/'manifest.json').read_text())
if meta['status']!='completed':raise ValueError('Public run must be complete before recording')
print('\n=== PUBLIC-DATA CHECKPOINT DEMO ===',flush=True)
print(f"Model: {meta['config']['model']} | device: {meta['environment']['device']} | {meta['completed_runs']} evaluated variants",flush=True)
print('Reduced public-data experiment. No GPT-4 or human win rates.',flush=True)
device=select_device();seed_everything(0)
for task,beta in [('sentiment','0.1'),('summarization','0.5'),('dialogue','0.1')]:
 path=root/'seed_0'/task/f'dpo_beta_{beta}'
 metrics=json.loads((path/'evaluation'/'summary.json').read_text())
 sample=read_jsonl(path/'evaluation'/'samples.jsonl')[0]
 model=load_model(path/'model',device);tok=load_tokenizer(path/'model')
 text=generate(model,tok,sample['prompt'],meta['config'],device,task,0.)[0][1]
 print(f'\n--- {task.upper()} / restored DPO checkpoint ---',flush=True)
 print(f"Held-out pair accuracy: {metrics['pairs']['preference_accuracy']['mean']:.3f} (not a generated-answer win rate)",flush=True)
 print('Prompt: '+sample['prompt'].replace('\n',' ')[:160],flush=True)
 print('Live generation: '+text.replace('\n',' '),flush=True)
 del model;release_memory();time.sleep(1)
print('\nCompleted: actual inference from all three saved public-data DPO checkpoints.',flush=True)
print(f'Tiny models and short contexts limit answer quality. Full evidence: {root}',flush=True)
