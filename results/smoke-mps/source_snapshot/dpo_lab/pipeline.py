"""Sequential experiment orchestration with verified stage-level restart."""
from __future__ import annotations
import json
import shutil
import time
from pathlib import Path
from huggingface_hub import HfApi
import torch
from dpo_lab.data import prepare, assert_disjoint
from dpo_lab.engine import encode_rows, cache_reference, train_stage, evaluate_pairs
from dpo_lab.models import build_fixture_model, load_model, load_tokenizer
from dpo_lab.evaluation import (SentimentOracle, fixture_oracle, sentiment_pairs,
                                evaluate_generations, export_judgments)
from dpo_lab.baselines import train_reward, reward_scorer, reward_accuracy, train_ppo, RewardModel
from dpo_lab.runtime import (select_device, seed_everything, release_memory, environment,
                             canonical_hash, file_hash, write_json, read_jsonl, write_jsonl)


def validate_config(cfg):
    for key in ('train_examples','eval_examples','batch_size','microbatch_size','sft_epochs','preference_epochs',
                'max_new_tokens','generation_examples','pair_candidates','ppo_steps','reward_epochs'):
        if cfg[key] < 1: raise ValueError(f'{key} must be positive')
    if not 0 < cfg['max_prompt_length'] < cfg['max_length']: raise ValueError('Invalid token lengths')
    if cfg['max_prompt_length'] + cfg['max_new_tokens'] > cfg['max_length']: raise ValueError('Generation exceeds max_length')
    if not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']): raise ValueError('Seeds must be nonempty and unique')
    allowed = {'sentiment','summarization','dialogue'}
    if not cfg['tasks'] or set(cfg['tasks']) - allowed: raise ValueError('Unknown or empty tasks')
    if any(b <= 0 for values in cfg['betas'].values() for b in values): raise ValueError('beta must be positive')
    if any(t < 0 for t in cfg['temperatures']): raise ValueError('temperature must be nonnegative')


def mark_complete(path):
    path = Path(path)
    files = {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob('*')) if p.is_file() and p.name != 'complete.json'}
    write_json(path / 'complete.json', {'files': files})


def is_complete(path):
    path = Path(path)
    if not (path / 'complete.json').exists(): return False
    record = json.loads((path / 'complete.json').read_text())
    for name, digest in record['files'].items():
        p = path / name
        if not p.exists() or file_hash(p) != digest: raise ValueError(f'Corrupt completed artifact: {p}')
    return True


def variants(cfg, task):
    result = [('base','base',None), ('sft','sft',None)]
    result += [(f'dpo_beta_{beta:g}','dpo',beta) for beta in cfg['betas'][task]]
    result += [(m,m,cfg['betas'][task][0] if task != 'sentiment' else .1) for m in cfg['extra_methods']]
    if task == 'sentiment': result += [('unlikelihood','unlikelihood',.1), ('ppo_gt','ppo_gt',.1)]
    return result


def run(config_path, output, device_name='auto', data_path=None, resume=False):
    cfg = json.loads(Path(config_path).read_text()); validate_config(cfg)
    device = select_device(device_name)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    request_hash = canonical_hash(cfg)
    code_hash = canonical_hash({str(p): file_hash(p) for p in sorted(p for p in Path('dpo_lab').glob('*.py') if p.stem not in {'report','cli','__main__','judging'})} |
                               {str(p): file_hash(p) for p in (Path('preference_datasets.py'), Path('utils.py'))})
    meta_path = output / 'manifest.json'
    if meta_path.exists():
        if not resume: raise ValueError('Output exists: choose another path or use --resume')
        meta = json.loads(meta_path.read_text())
        if meta['request_hash'] != request_hash or meta['code_hash'] != code_hash:
            raise ValueError('Config or source changed; use a new output directory')
        current = environment(device)
        if meta['environment']['packages'] != current['packages'] or meta['environment']['device'] != str(device):
            raise ValueError('Package versions or device changed; use a new output directory')
        cfg = meta['config']
    else:
        if cfg['data_mode'] == 'public':
            if not cfg['model_revision']: cfg['model_revision'] = HfApi().model_info(cfg['model']).sha
            if not cfg['sentiment_oracle_revision']: cfg['sentiment_oracle_revision'] = HfApi().model_info(cfg['sentiment_oracle']).sha
        meta = {'request_hash':request_hash,'code_hash':code_hash,'config':cfg,'environment':environment(device),
                'status':'running','scope':'scaled public-data experiment' if cfg['data_mode']=='public' else 'synthetic engineering smoke test',
                'paper_scale_reproduction':False,'gpt4_evaluation':'not run','human_evaluation':'not run'}
        write_json(meta_path, meta)
        snapshot = output / 'source_snapshot'
        (snapshot/'dpo_lab').mkdir(parents=True,exist_ok=True)
        for p in Path('dpo_lab').glob('*.py'): shutil.copy2(p,snapshot/'dpo_lab'/p.name)
        for name in ('preference_datasets.py','utils.py'): shutil.copy2(name,snapshot/name)
    data = output / 'data'
    if not (data / 'manifest.json').exists():
        if data_path:
            source = Path(data_path)
            manifest = json.loads((source/'manifest.json').read_text())
            if manifest['mode'] != cfg['data_mode']: raise ValueError('Data mode does not match config')
            shutil.copytree(source, data, dirs_exist_ok=True)
        else: prepare(cfg, data)
    data_manifest = json.loads((data/'manifest.json').read_text())
    for name, record in data_manifest['files'].items():
        if file_hash(data/name) != record['sha256']: raise ValueError(f'Corrupt dataset {name}')
    base_path = cfg['model']
    if base_path == 'fixture':
        base_path = output/'fixture_model'
        if not base_path.exists():
            rows = [r for task in cfg['tasks'] for split in ('train','test') for r in read_jsonl(data/task/f'{split}.jsonl')]
            build_fixture_model(rows, base_path)
    tokenizer = load_tokenizer(base_path, cfg['model_revision'])
    summaries = []
    for seed in cfg['seeds']:
        for task in cfg['tasks']:
            root = output / f'seed_{seed}' / task; root.mkdir(parents=True, exist_ok=True)
            print(f'\n=== {task} / seed {seed} / {device} ===', flush=True)
            sft_path = root/'sft'
            if not is_complete(sft_path):
                model = load_model(base_path, device, cfg['model_revision'], cfg['gradient_checkpointing'])
                sft = encode_rows(read_jsonl(data/task/'sft.jsonl'),tokenizer,cfg,task)
                train_stage(model,tokenizer,sft,None,cfg,'sft',sft_path,device,seed)
                # SFT training completion is recorded separately from its later evaluation.
                mark_complete(sft_path)
                del model; release_memory()
            sft_model = sft_path/'model'
            reference = load_model(sft_model,device).eval().requires_grad_(False)
            train_rows,test_rows = [read_jsonl(data/task/f'{s}.jsonl') for s in ('train','test')]
            oracle = None
            if task == 'sentiment':
                oracle = SentimentOracle(cfg['sentiment_oracle'],cfg['sentiment_oracle_revision'],device) if cfg['data_mode']=='public' else fixture_oracle
                if cfg['data_mode']=='public':
                    generated = root/'preferences'
                    for split,rows in [('train',train_rows),('test',test_rows)]:
                        if not (generated/split/'pairs.jsonl').exists():
                            sentiment_pairs(reference,tokenizer,rows,cfg,device,oracle,generated/split,seed + (10000 if split=='test' else 0))
                    train_rows,test_rows = [read_jsonl(generated/s/'pairs.jsonl') for s in ('train','test')]
            assert_disjoint(train_rows,test_rows)
            # Retain one generation prompt per source; preference pairs may share a prompt.
            generation_rows = list({r['prompt']:r for r in test_rows}.values())
            train_items,test_items = [encode_rows(rows,tokenizer,cfg,task) for rows in (train_rows,test_rows)]
            filtered = {}
            for name,items,rows in [('train',train_items,train_rows),('test',test_items,test_rows)]:
                kept = [(i,r) for i,r in zip(items,rows) if not i.get('indistinguishable')]
                filtered[name] = {'before':len(items),'after':len(kept)}
                items[:] = [i for i,r in kept]; rows[:] = [r for i,r in kept]
                if not items: raise ValueError(f'No distinguishable {task}/{name} pairs after truncation')
            write_json(root/'pair_counts.json',filtered)
            identity = {p.name:file_hash(p) for p in sft_model.glob('*.safetensors')}
            train_refs = cache_reference(reference,train_items,tokenizer,device,root/'train_reference.json',identity)
            test_refs = cache_reference(reference,test_items,tokenizer,device,root/'test_reference.json',identity)
            reward_path = root/'reward_model'
            if is_complete(reward_path):
                reward = RewardModel(load_model(sft_model,device))
                reward.load_state_dict(torch.load(reward_path/'reward.pt',map_location=device,weights_only=True))
                reward.requires_grad_(False).eval()
            else:
                reward = train_reward(sft_model,tokenizer,train_items,cfg,device,reward_path,seed)
                write_json(reward_path/'evaluation.json',{'heldout_accuracy':reward_accuracy(reward,tokenizer,test_items,device),
                                                          'selection':'final epoch; no validation tuning in scaled profile'})
                mark_complete(reward_path)
            learned_scorer = reward_scorer(reward,tokenizer,cfg,device,task)
            for name,method,beta in variants(cfg,task):
                path = root/name
                # SFT model's completion receipt covers training; evaluations have separate receipts.
                evaluation = path/'evaluation'
                if is_complete(evaluation):
                    summaries.append(json.loads((evaluation/'summary.json').read_text())); continue
                print(f'--- {name} ---', flush=True)
                source = base_path if method=='base' else sft_model
                revision = cfg['model_revision'] if method=='base' else None
                if method not in {'base','sft'} and is_complete(path):
                    model = load_model(path/'model',device)
                else:
                    model = load_model(source,device,revision,cfg['gradient_checkpointing'])
                    if method in {'ppo','ppo_gt'}:
                        scorer = oracle if method=='ppo_gt' else learned_scorer
                        train_ppo(model,reference,tokenizer,train_rows,scorer,cfg,device,task,path,seed)
                        mark_complete(path)
                    elif method not in {'base','sft'}:
                        reference.to('cpu')
                        release_memory()
                        train_stage(model,tokenizer,train_items,train_refs,cfg,method,path,device,seed,beta,
                                    smoothing=.1 if method=='cdpo' else 0.)
                        mark_complete(path)
                reference.to(device)
                evaluation.mkdir(parents=True,exist_ok=True)
                pair_metrics = evaluate_pairs(model,tokenizer,test_items,test_refs,cfg,device,beta or cfg['betas'][task][0],evaluation/'pairs.jsonl')
                generation_metrics = evaluate_generations(model,reference,tokenizer,generation_rows,cfg,device,task,
                                                         evaluation/'samples.jsonl',seed,oracle if task=='sentiment' else learned_scorer)
                if task != 'sentiment':
                    export_judgments(read_jsonl(evaluation/'samples.jsonl'),task,evaluation/'judgments.jsonl',seed)
                if task=='summarization':
                    ood = evaluate_generations(model,reference,tokenizer,read_jsonl(data/'ood.jsonl'),cfg,device,task,
                                               evaluation/'ood_samples.jsonl',seed,learned_scorer)
                    export_judgments(read_jsonl(evaluation/'ood_samples.jsonl'),task,evaluation/'ood_judgments.jsonl',seed)
                else: ood = None
                summary = dict(task=task,seed=seed,method=name,beta=beta,pairs=pair_metrics,generation=generation_metrics,ood=ood,
                               score_kind='sentiment oracle probability' if task=='sentiment' else 'learned reward diagnostic, not a win rate')
                write_json(evaluation/'summary.json',summary); mark_complete(evaluation)
                summaries.append(summary)
                del model; release_memory()
            for n in cfg['best_of']:
                path=root/f'best_of_{n}'/'evaluation'
                if is_complete(path):
                    summaries.append(json.loads((path/'summary.json').read_text())); continue
                metrics=evaluate_generations(reference,reference,tokenizer,generation_rows,cfg,device,task,path/'samples.jsonl',seed,learned_scorer,best_of=n)
                summary=dict(task=task,seed=seed,method=f'best_of_{n}',beta=None,pairs=None,generation=metrics,ood=None,
                             score_kind='learned reward used for selection; not an independent quality evaluator')
                if task=='sentiment':
                    samples=read_jsonl(path/'samples.jsonl')
                    for row in samples: row['oracle_score']=oracle(row['prompt'],row['completion'])
                    write_jsonl(path/'samples.jsonl',samples)
                else: export_judgments(read_jsonl(path/'samples.jsonl'),task,path/'judgments.jsonl',seed)
                write_json(path/'summary.json',summary); mark_complete(path); summaries.append(summary)
            del reference,reward,learned_scorer,oracle
            release_memory()
            write_json(output/'summary.json',summaries)
    meta['status']='completed'; meta['completed_runs']=len(summaries)
    write_json(meta_path,meta)
    print(f'Completed {len(summaries)} evaluated variants. Results: {output}',flush=True)
    return summaries
