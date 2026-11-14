import pytest
from dpo_lab.judging import agreement


def rows(winners,evaluator):
    return [dict(id=str(i),winner=w,evaluator=evaluator,prompt='q',answer_a='a',answer_b='b') for i,w in enumerate(winners)]


def test_agreement_and_alignment():
    result=agreement(rows(['A','B'],'human:1'),rows(['A','B'],'human:2'))
    assert result['agreement']['mean']==1 and result['cohen_kappa']==1
    result=agreement(rows(['A','B'],'human:1'),rows(['B','A'],'judge:version'))
    assert result['agreement']['mean']==0 and result['cohen_kappa']==-1
    other=rows(['A','B'],'human:2');other[0]['answer_a']='different'
    with pytest.raises(ValueError):agreement(rows(['A','B'],'human:1'),other)
