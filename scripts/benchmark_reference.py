"""Measure cached vs live frozen-reference DPO steps; full checkpoint and real data."""
import argparse,json,time
from pathlib import Path
import torch
from dpo_lab.models import load_model,load_tokenizer
from dpo_lab.engine import encode_rows,batch_for,logps
from dpo_lab.objectives import preference_loss
from dpo_lab.runtime import read_jsonl,seed_everything,synchronize,write_json
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--pairs',required=True);p.add_argument('--output',required=True);a=p.parse_args()
seed_everything(0);device=torch.device('mps');tok=load_tokenizer(a.checkpoint)
policy=load_model(a.checkpoint,device).eval();ref=load_model(a.checkpoint,device).eval().requires_grad_(False)
items=encode_rows(read_jsonl(a.pairs)[:1],tok,{'max_length':128,'max_prompt_length':88},'sentiment');b=batch_for(items,tok,device)
with torch.no_grad():rc,rr=logps(ref,b),logps(ref,b,'rejected')
results={}
for live in (True,False):
 if not live:ref.to('cpu');torch.mps.empty_cache()
 times=[]
 for step in range(12):
  policy.zero_grad(set_to_none=True);synchronize(device);start=time.perf_counter()
  if live:
   with torch.no_grad():rc,rr=logps(ref,b),logps(ref,b,'rejected')
  c,r=logps(policy,b),logps(policy,b,'rejected');loss,*_=preference_loss(c,r,rc,rr,.1);loss.mean().backward()
  synchronize(device);elapsed=time.perf_counter()-start
  if step>=2:times.append(elapsed)
 results['live' if live else 'cached']={'mean_seconds':sum(times)/len(times),'step_seconds':times,'allocated_mps_bytes':torch.mps.current_allocated_memory()}
results['speedup']=results['live']['mean_seconds']/results['cached']['mean_seconds']
results['scope']='10 measured single-pair forward/backward steps after 2 warmups; excludes cache construction, optimizer and generation'
write_json(a.output,results);print(json.dumps(results,indent=2))
