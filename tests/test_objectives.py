import math
import pytest
import torch
import torch.nn.functional as F
from dpo_lab.objectives import preference_loss, _get_batch_logps


def test_dpo_equation_and_gradient_direction():
    chosen=torch.tensor([-.8,-1.2],requires_grad=True)
    rejected=torch.tensor([-1.5,-.7],requires_grad=True)
    rc,rr=torch.tensor([-1.,-1.]),torch.tensor([-1.1,-1.1])
    loss,cw,rw=preference_loss(chosen,rejected,rc,rr,.2)
    expected=-F.logsigmoid(.2*((chosen-rc)-(rejected-rr)))
    torch.testing.assert_close(loss,expected)
    loss.sum().backward()
    assert (chosen.grad<0).all() and (rejected.grad>0).all()
    assert not cw.requires_grad and not rw.requires_grad


def test_identity_is_log_two_and_cdpo_symmetry():
    p=torch.tensor([-3.,-4.]); q=torch.tensor([-2.,-5.])
    loss,*_=preference_loss(p,q,p,q,.1)
    torch.testing.assert_close(loss,torch.full_like(loss,math.log(2)))
    a,*_=preference_loss(p,q,torch.zeros(2),torch.zeros(2),.1,label_smoothing=.5)
    b,*_=preference_loss(q,p,torch.zeros(2),torch.zeros(2),.1,label_smoothing=.5)
    torch.testing.assert_close(a,b)


def test_ipo_and_reference_free_rewards():
    p,q=torch.tensor([-1.]),torch.tensor([-3.])
    loss,*_=preference_loss(p,q,p,q,.2,ipo=True)
    assert loss.item()==pytest.approx(6.25)
    loss,cw,rw=preference_loss(p,q,torch.tensor([-9.]),torch.tensor([-8.]),.2,reference_free=True)
    torch.testing.assert_close(cw-rw,.2*(p-q))


def test_mask_shift_padding_and_ipo_mean():
    torch.manual_seed(0)
    logits=torch.randn(2,5,9,requires_grad=True)
    labels=torch.tensor([[-100,-100,3,4,-100],[-100,2,3,4,5]])
    sums=_get_batch_logps(logits,labels)
    means=_get_batch_logps(logits,labels,True)
    torch.testing.assert_close(sums,means*torch.tensor([2,4]))
    expected=logits[0,1].log_softmax(-1)[3]+logits[0,2].log_softmax(-1)[4]
    torch.testing.assert_close(sums[0],expected)
    sums.sum().backward()
    assert logits.grad[0,0].abs().sum()==0
    assert logits.grad[0,3:].abs().sum()==0
    with pytest.raises(ValueError): _get_batch_logps(logits,torch.full_like(labels,-100))


@pytest.mark.parametrize('beta,smoothing',[(0,0),(-1,0),(.1,-.1),(.1,.8)])
def test_invalid_objective(beta,smoothing):
    z=torch.zeros(1)
    with pytest.raises(ValueError): preference_loss(z,z,z,z,beta,smoothing)
