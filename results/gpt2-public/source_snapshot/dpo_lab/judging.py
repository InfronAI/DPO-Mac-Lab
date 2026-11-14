"""Agreement statistics for genuine, aligned external judgments."""
from collections import defaultdict
from dpo_lab.engine import mean_ci


def agreement(left, right):
    def index(rows):
        result={}
        for r in rows:
            if r.get('winner') not in {'A','B','tie'} or not r.get('evaluator'):
                raise ValueError('Actual winner and evaluator provenance are required')
            if r['id'] in result: raise ValueError('Duplicate IDs; pass one judgment per item per evaluator')
            result[r['id']]=r
        return result
    a,b=index(left),index(right)
    if not a or set(a)!=set(b): raise ValueError('Judgment sets must have identical, nonempty IDs')
    scores=[]
    for key in a:
        for field in ('prompt','answer_a','answer_b'):
            if a[key].get(field)!=b[key].get(field): raise ValueError('Judges did not see the same A/B presentation')
        scores.append(float(a[key]['winner']==b[key]['winner']))
    observed=sum(scores)/len(scores)
    labels=('A','B','tie')
    expected=sum(sum(r['winner']==label for r in a.values())*sum(r['winner']==label for r in b.values()) for label in labels)/len(scores)**2
    return {'agreement':mean_ci(scores),'cohen_kappa':(observed-expected)/(1-expected) if expected<1 else None,
            'left_evaluators':sorted({r['evaluator'] for r in a.values()}),
            'right_evaluators':sorted({r['evaluator'] for r in b.values()})}
