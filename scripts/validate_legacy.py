"""Execute the original Hydra entry with a tail batch, SFT handoff, DPO and IPO."""
import argparse,json,subprocess,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--device',default='mps');a=p.parse_args()
base=Path('artifacts/legacy-current-'+a.device);logs=Path('results/validation');logs.mkdir(parents=True,exist_ok=True)
common=[sys.executable,'train.py','model=blank_model','model.name_or_path=results/smoke-cpu/fixture_model',
        'model.reference_dtype=float32','datasets=[results/smoke-cpu/data/dialogue]',f'device={a.device}',
        'batch_size=3','gradient_accumulation_steps=2','eval_batch_size=3','n_epochs=1','n_examples=5','n_eval_examples=4',
        'max_length=64','max_prompt_length=40','eval_every=3','sample_during_eval=false','wandb.enabled=false']
records=[]
for method in ('sft','dpo','ipo'):
 output=base/method
 command=common+[f'loss={method}',f'local_run_dir={output}',f'exp_name=legacy_{method}']
 if method!='sft':command += ['loss.beta=0.1',f'model.archive={base}/sft/LATEST/policy.pt']
 with (logs/f'legacy-{a.device}-{method}.txt').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
 state=torch.load(output/'LATEST'/'policy.pt',map_location='cpu',weights_only=True)
 assert state['step_idx']==5
 assert all(torch.isfinite(t).all() for t in state['state'].values())
 metrics=[json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
 assert metrics[-1]['phase']=='final_eval'
 records.append({'method':method,'device':a.device,'examples':state['step_idx'],'final_metrics':metrics[-1]['metrics'],'passed':True})
 print(f'{a.device} legacy {method}: 5 examples, tail batch and final checkpoint verified',flush=True)
(logs/f'legacy-{a.device}.json').write_text(json.dumps(records,indent=2)+'\n')
