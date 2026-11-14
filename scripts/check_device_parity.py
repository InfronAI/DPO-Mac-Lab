"""Numerical CPU/MPS checks on the actual pretrained model and variable-length data."""
import argparse,json
import torch
from dpo_lab.models import load_model,load_tokenizer
from dpo_lab.engine import encode_rows,batch_for,logps
from dpo_lab.runtime import read_jsonl,seed_everything,write_json
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--model',default='EleutherAI/pythia-70m-deduped');p.add_argument('--revision',default='e93a9faa9c77e5d09219f6c868bfc7a1bd65593c');p.add_argument('--data',default='artifacts/public-data/dialogue/train.jsonl');a=p.parse_args()
seed_everything(0);tok=load_tokenizer(a.model,a.revision)
rows=read_jsonl(a.data)[:3]
model=load_model(a.model,torch.device('cpu'),a.revision).eval()
records=[]
for maxlen in (48,80,128):
 items=encode_rows(rows,tok,{'max_length':maxlen,'max_prompt_length':maxlen//2},'dialogue')
 # Identical forward/backward computation on both backends, no parameter updates.
 model.to('cpu');model.zero_grad(set_to_none=True)
 cpu_loss=-logps(model,batch_for(items,tok,'cpu')).mean();cpu_loss.backward()
 cpu_grads={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
 model.zero_grad(set_to_none=True);model.to('mps')
 mps_loss=-logps(model,batch_for(items,tok,'mps')).mean();mps_loss.backward();torch.mps.synchronize()
 errors=[]
 total_error_sq=0.;total_cpu_sq=0.;total_mps_sq=0.;dot=0.
 for n,param in model.named_parameters():
  if param.grad is not None:
   x=param.grad.detach().cpu();y=cpu_grads[n]
   difference=x-y
   total_error_sq+=float(difference.square().sum());total_cpu_sq+=float(y.square().sum());total_mps_sq+=float(x.square().sum());dot+=float((x*y).sum())
   errors.append((n,float(difference.norm()/y.norm().clamp_min(1e-6)),float(y.norm()),float(difference.abs().max())))
 worst=max(errors,key=lambda x:x[1]);passed=abs(cpu_loss.item()-mps_loss.item())<.1 and worst[1]<.02
 records.append(dict(length=maxlen,cpu_loss=cpu_loss.item(),mps_loss=mps_loss.item(),worst_gradient=worst,global_gradient_relative_l2=(total_error_sq/max(total_cpu_sq,1e-12))**.5,global_gradient_cosine=dot/max((total_cpu_sq*total_mps_sq)**.5,1e-12),passed=passed))
 print(records[-1],flush=True)
write_json(a.output,{'torch':torch.__version__,'model':a.model,'results':records,'passed':all(r['passed'] for r in records)})
if not all(r['passed'] for r in records):raise SystemExit(1)
