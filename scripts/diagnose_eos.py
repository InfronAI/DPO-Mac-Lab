"""Document EOS likelihood and verify causal masking on a supplied checkpoint."""
import argparse,json
import torch
from dpo_lab.models import load_model,load_tokenizer
from dpo_lab.runtime import write_json,seed_everything
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--revision');p.add_argument('--output',required=True);a=p.parse_args()
seed_everything(0);m=load_model(a.model,torch.device('cpu'),a.revision).eval();t=load_tokenizer(a.model,a.revision)
text='The movie was good and I really enjoyed it. I thought it was interesting.'
ids=t(text,return_tensors='pt')['input_ids'];with_eos=torch.cat([ids,torch.tensor([[t.eos_token_id]])],1)
changed=with_eos.clone();changed[0,-1]=100
with torch.no_grad():
 output=m(with_eos,attention_mask=torch.ones_like(with_eos)).logits
 altered=m(changed,attention_mask=torch.ones_like(changed)).logits
 logps=output[0,:-1].log_softmax(-1).gather(-1,with_eos[0,1:,None]).squeeze(-1)
 result={'model':a.model,'revision':a.revision,'text':text,'eos_token_id':t.eos_token_id,
         'eos_log_probability':logps[-1].item(),'mean_non_eos_log_probability':logps[:-1].mean().item(),
         'future_token_max_logit_difference':(output[:,:-1]-altered[:,:-1]).abs().max().item(),
         'scope':'Single fixed diagnostic sentence; not a corpus quality benchmark','torch':torch.__version__}
write_json(a.output,result);print(json.dumps(result,indent=2))
