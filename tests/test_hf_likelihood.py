import torch
from transformers import GPT2Config,GPT2LMHeadModel
from dpo_lab.objectives import _get_batch_logps


def test_response_likelihood_matches_hugging_face_loss():
    torch.manual_seed(0)
    model=GPT2LMHeadModel(GPT2Config(vocab_size=19,n_positions=16,n_embd=16,n_head=2,n_layer=1,
                                    resid_pdrop=0,attn_pdrop=0,embd_pdrop=0)).eval()
    ids=torch.tensor([[1,2,3,4,5,0],[1,3,2,4,6,7]])
    mask=torch.tensor([[1,1,1,1,1,0],[1,1,1,1,1,1]])
    labels=ids.clone();labels[:,:2]=-100;labels[mask==0]=-100
    output=model(ids,attention_mask=mask,labels=labels)
    logps=_get_batch_logps(output.logits,labels)
    n=(labels[:,1:]!=-100).sum()
    torch.testing.assert_close(-logps.sum()/n,output.loss)
