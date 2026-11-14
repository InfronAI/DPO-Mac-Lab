"""Reproduce the nonfinite-gradient issue on the same frozen sentiment pairs."""
import json,random,sys
import torch
from dpo_lab.models import load_model,load_tokenizer
from dpo_lab.engine import encode_rows,batch_for,logps
from dpo_lab.objectives import preference_loss
from dpo_lab.runtime import read_jsonl,seed_everything
cfg=json.load(open('experiments/mac.json'));device=sys.argv[1] if len(sys.argv)>1 else 'mps'
seed_everything(0)
tok=load_tokenizer('artifacts/failed-public-torch26/seed_0/sentiment/sft/model')
model=load_model('artifacts/failed-public-torch26/seed_0/sentiment/sft/model',torch.device(device))
items=encode_rows(read_jsonl('artifacts/failed-public-torch26/seed_0/sentiment/preferences/train/pairs.jsonl'),tok,cfg,'sentiment')
refs=json.load(open('artifacts/failed-public-torch26/seed_0/sentiment/train_reference.json'))['values']
opt=torch.optim.RMSprop(model.parameters(),lr=cfg['preference_lr']);model.train()
order=list(range(len(items)));random.Random(0).shuffle(order)
with torch.autograd.detect_anomaly():
 for step in range(12):
  opt.zero_grad(set_to_none=True)
  for group in opt.param_groups:group['lr']=cfg['preference_lr']*(step+1)/4
  for i in order[step*4:step*4+4]:
   b=batch_for([items[i]],tok,device)
   c,r=logps(model,b),logps(model,b,'rejected')
   rc,rr=(torch.tensor([refs[i][k]],device=device) for k in ('chosen','rejected'))
   loss,*_=preference_loss(c,r,rc,rr,.1)
   print('step',step,'row',i,'loss',loss.item(),flush=True)
   (loss.mean()/4).backward()
  bad=[(n,int((~torch.isfinite(p.grad)).sum())) for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
  print('gradient norm',norm.item(),'bad',bad[:5],flush=True)
  opt.step()
