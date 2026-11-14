import json
import pytest
import torch
from dpo_lab.models import build_fixture_model,load_model,load_tokenizer
from dpo_lab.data import fixture_rows,assert_disjoint,validate_rows
from dpo_lab.engine import encode_rows,batch_for,cache_reference,logps
from dpo_lab.objectives import preference_loss
from dpo_lab.runtime import write_jsonl
from preference_datasets import tokenize_batch_element,get_batch_iterator
from utils import slice_and_move_batch_for_device
from dpo_lab.evaluation import score_judgments
from dpo_lab.baselines import gae

@pytest.fixture
def tiny(tmp_path):
    rows=fixture_rows('dialogue',5)
    build_fixture_model(rows,tmp_path/'model')
    return rows,load_model(tmp_path/'model',torch.device('cpu')),load_tokenizer(tmp_path/'model')


def test_truncation_labels_eos_and_generation_bound(tiny):
    rows,_,tok=tiny
    item=tokenize_batch_element('Human: '*20,'Start '*20,'Never '*20,'keep_end',tok,16,7)
    assert len(item['prompt_input_ids'])==7
    for side in ('chosen','rejected'):
        assert len(item[f'{side}_input_ids'])==16
        assert item[f'{side}_input_ids'][-1]==tok.eos_token_id
        assert item[f'{side}_labels'][:7]==[-100]*7
    short=tokenize_batch_element('Human:','Start '*20,'Never '*20,'keep_start',tok,16,7)
    assert len(short['chosen_input_ids'])==16  # Uses actual prompt length, not max_prompt_length.


def test_tail_batch_limits_and_deterministic_seed(tmp_path,tiny):
    rows,_,tok=tiny
    write_jsonl(tmp_path/'train.jsonl',rows)
    kw=dict(names=[str(tmp_path)],tokenizer=tok,n_epochs=1,batch_size=4,max_length=64,max_prompt_length=40)
    batches=list(get_batch_iterator(**kw))
    assert [len(b['prompt']) for b in batches]==[4,1]
    assert sum(len(b['prompt']) for b in get_batch_iterator(**(kw|{'n_examples':3})))==3
    one=list(get_batch_iterator(**(kw|{'batch_size':20,'n_epochs':None,'n_examples':2})))
    assert len(one[0]['prompt'])==2
    b={'x':torch.arange(5)}
    slices=[slice_and_move_batch_for_device(b,i,2,'cpu')['x'] for i in range(2)]
    torch.testing.assert_close(torch.cat(slices),b['x'])


def test_cache_matches_live_gradients_and_rejects_stale(tmp_path,tiny):
    rows,model,tok=tiny
    cfg={'max_length':64,'max_prompt_length':40}
    items=encode_rows(rows[:2],tok,cfg,'dialogue')
    refs=cache_reference(model,items,tok,torch.device('cpu'),tmp_path/'ref.json',{'sft':'one'})
    model.eval()
    def backward(cached):
        model.zero_grad()
        b=batch_for(items,tok,'cpu')
        chosen,rejected=logps(model,b),logps(model,b,'rejected')
        if cached:
            rc,rr=[torch.tensor([r[s] for r in refs]) for s in ('chosen','rejected')]
        else:
            with torch.no_grad(): rc,rr=logps(model,b),logps(model,b,'rejected')
        loss,*_=preference_loss(chosen,rejected,rc,rr,.1)
        loss.mean().backward()
        return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    torch.testing.assert_close(backward(True),backward(False),atol=2e-6,rtol=2e-5)
    with pytest.raises(ValueError,match='Stale'): cache_reference(model,items,tok,'cpu',tmp_path/'ref.json',{'sft':'two'})


def test_microbatch_accumulation_equals_batch(tiny):
    rows,model,tok=tiny
    items=encode_rows(rows[:3],tok,{'max_length':64,'max_prompt_length':40},'dialogue')
    model.eval(); model.zero_grad()
    (-logps(model,batch_for(items,tok,'cpu')).mean()).backward()
    expected={k:p.grad.clone() for k,p in model.named_parameters()}
    model.zero_grad()
    for item in items: (-logps(model,batch_for([item],tok,'cpu')).sum()/3).backward()
    for k,p in model.named_parameters(): torch.testing.assert_close(p.grad,expected[k],atol=2e-6,rtol=3e-5)


def test_leakage_empty_data_and_judgment_provenance():
    with pytest.raises(ValueError): assert_disjoint([{'prompt':'a'}],[{'prompt':'a'}])
    with pytest.raises(ValueError): validate_rows([])
    key=[{'id':'0','candidate':'A'}]
    with pytest.raises(ValueError): score_judgments([{'id':'0','winner':None}],key)
    result=score_judgments([{'id':'0','winner':'A','evaluator':'human:test'}],key)
    assert result['win_rate']['mean']==1


def test_gae_terminal_and_returns():
    adv,ret=gae(torch.tensor([0.,0.,1.]),torch.zeros(3),1.)
    torch.testing.assert_close(adv,torch.ones(3)); torch.testing.assert_close(ret,torch.ones(3))
