"""Audit completed runs, complete receipts, numerical artifacts and train/test splits."""
import argparse,json,math
from pathlib import Path
from dpo_lab.runtime import read_jsonl,file_hash,write_json
from dpo_lab.pipeline import is_complete,variants
from dpo_lab.data import assert_disjoint
p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();root=Path(a.directory)
meta=json.loads((root/'manifest.json').read_text());cfg=meta['config']
if meta['status']!='completed':raise ValueError('Experiment is not complete')
expected=sum(len(variants(cfg,t))+len(cfg['best_of']) for t in cfg['tasks'])*len(cfg['seeds'])
summaries=json.loads((root/'summary.json').read_text())
if len(summaries)!=expected:raise ValueError(f'Expected {expected} variants, got {len(summaries)}')
keys={(r['seed'],r['task'],r['method']) for r in summaries}
if len(keys)!=expected:raise ValueError('Duplicate evaluation records')
receipts=0
for path in root.rglob('complete.json'):
 is_complete(path.parent);receipts+=1
for task in cfg['tasks']:
 assert_disjoint(read_jsonl(root/'data'/task/'train.jsonl'),read_jsonl(root/'data'/task/'test.jsonl'))
def finite(value):
 if isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite recorded scalar')
 if isinstance(value,dict):
  for v in value.values():finite(v)
 if isinstance(value,list):
  for v in value:finite(v)
files={}
for path in sorted(root.rglob('*')):
 if not path.is_file() or path.name=='validation.json':continue
 if path.suffix in {'.json','.jsonl'}:
  if path.suffix=='.json':finite(json.loads(path.read_text()))
  else:
   for row in read_jsonl(path):finite(row)
 # Completion receipts separately verify large weights. This index binds analysis/data files too.
 if path.suffix not in {'.pt','.safetensors'}:files[str(path.relative_to(root))]=file_hash(path)
result={'passed':True,'expected_variants':expected,'actual_variants':len(summaries),'verified_receipts':receipts,'files':files}
write_json(root/'validation.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
