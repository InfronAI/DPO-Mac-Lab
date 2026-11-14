"""Bounded public data acquisition. No synthetic fallback for a failed download."""
from __future__ import annotations

import itertools
import random
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from dpo_lab.runtime import canonical_hash, file_hash, read_jsonl, write_json, write_jsonl


def validate_rows(rows, name="dataset"):
    if not rows:
        raise ValueError(f"{name} is empty")
    for i, row in enumerate(rows):
        for key in ("prompt", "chosen", "rejected"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{name}[{i}].{key} must be nonempty text")
        if row['chosen'] == row['rejected'] and not row.get('sft_only'):
            raise ValueError(f"{name}[{i}] has identical preference responses")


def assert_disjoint(train, test):
    overlap = {r['prompt'] for r in train} & {r['prompt'] for r in test}
    if overlap:
        raise ValueError(f"Train/test leakage: {len(overlap)} shared prompts")


def fixture_rows(task, n, offset=0):
    """Explicitly artificial fixtures for fast, network-free engineering tests."""
    rows = []
    for i in range(offset, offset + n):
        if task == 'sentiment':
            prompt, chosen, rejected = f'Review {i}: This movie', ' is delightful and enjoyable.', ' is dull and disappointing.'
        elif task == 'summarization':
            prompt = f'POST: On day {i}, Alex repaired a bicycle and rode to school.\nTL;DR:'
            chosen, rejected = ' Alex fixed a bicycle and rode to school.', ' Alex bought a car and stayed home.'
        else:
            prompt = f'\n\nHuman: In exercise {i}, how can I learn to cook?\n\nAssistant:'
            chosen, rejected = ' Start with a simple recipe and practice.', ' Never practice and ignore all recipes.'
        rows.append(dict(id=f'fixture-{task}-{i}', prompt=prompt, chosen=chosen, rejected=rejected,
                         source='synthetic engineering fixture'))
    return rows


def stream_rows(repo, config, split, limit, seed, revision):
    if repo == 'openai/summarize_from_feedback':
        url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{config}/{split}/0000.parquet'
        stream = load_dataset('parquet', data_files={split: url}, split=split, streaming=True)
    else:
        stream = load_dataset(repo, name=config, split=split, revision=revision, streaming=True)
    # Bounded shuffle, not a claim of uniform sampling over the full corpus.
    return itertools.islice(stream.shuffle(seed=seed, buffer_size=256), limit)


def prepare(config, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'mode': config['data_mode'], 'sampling': 'seeded streaming buffer shuffle (256); bounded scan',
                'seed': config['data_seed'], 'sources': {}, 'files': {}}
    n, m = config['train_examples'], config['eval_examples']
    if config['data_mode'] == 'fixture':
        for task in config['tasks']:
            for split, size, offset in [('train', n, 0), ('test', m, 10000), ('sft', n, 0)]:
                write_jsonl(out / task / f'{split}.jsonl', fixture_rows(task, size, offset))
        write_jsonl(out / 'ood.jsonl', fixture_rows('summarization', m, 20000))
    elif config['data_mode'] == 'public':
        api = HfApi()
        def source(repo, name, split, limit):
            revision = config.get('dataset_revisions', {}).get(repo) or api.dataset_info(repo, revision='refs/convert/parquet' if repo == 'openai/summarize_from_feedback' else 'main').sha
            manifest['sources'][repo] = revision
            return stream_rows(repo, name, split, limit, config['data_seed'], revision)
        for task in config['tasks']:
            for split, size in [('train', n), ('test', m)]:
                rows = []
                seen = {r['prompt'] for r in read_jsonl(out/task/'train.jsonl')} if split == 'test' else set()
                if task == 'dialogue':
                    raw = source('Anthropic/hh-rlhf', None, split, size * 50)
                    for row in raw:
                        pos = row['chosen'].rfind('\n\nAssistant:') + len('\n\nAssistant:')
                        prompt = row['chosen'][:pos]
                        # Paper evaluates single-turn dialogue; use it for both bounded splits.
                        if prompt.count('\n\nHuman:') != 1 or not row['rejected'].startswith(prompt):
                            continue
                        item = dict(prompt=prompt, chosen=row['chosen'][pos:], rejected=row['rejected'][pos:])
                        if prompt in seen or not item['chosen'].strip() or item['chosen'] == item['rejected']:
                            continue
                        seen.add(prompt); rows.append(item)
                        if len(rows) == size: break
                elif task == 'summarization':
                    raw = source('openai/summarize_from_feedback', 'comparisons', 'train' if split == 'train' else 'validation', size * 30)
                    for row in raw:
                        info = row['info']
                        prompt = f"SUBREDDIT: {info['subreddit']}\nTITLE: {info['title']}\nPOST: {info['post']}\nTL;DR:"
                        chosen = row['choice']
                        item = dict(prompt=prompt, chosen=' ' + row['summaries'][chosen]['text'],
                                    rejected=' ' + row['summaries'][1 - chosen]['text'])
                        if prompt in seen or item['chosen'] == item['rejected']: continue
                        seen.add(prompt); rows.append(item)
                        if len(rows) == size: break
                elif task == 'sentiment':
                    raw = source('stanfordnlp/imdb', None, split, size * 2)
                    for row in raw:
                        # Exact 2–8 tokenizer tokens are selected after SFT by sentiment_pairs().
                        words = row['text'].replace('<br />', ' ').split()
                        prompt = ' '.join(words[:4])
                        if prompt in seen or len(words) < 12: continue
                        seen.add(prompt)
                        rows.append(dict(prompt=prompt, chosen=' ' + ' '.join(words[4:]),
                                         rejected=' ' + ' '.join(words[4:]), sft_only=True,
                                         review=' '.join(words), label=row['label']))
                        if len(rows) == size: break
                else:
                    raise ValueError(f'Unknown task: {task}')
                if len(rows) != size:
                    raise RuntimeError(f'{task}/{split}: found {len(rows)} of {size}; increase bounded scan or lower size')
                for i, row in enumerate(rows):
                    row['id'] = canonical_hash([task, split, row['prompt']])[:20]
                    row['source'] = task
                validate_rows(rows)
                write_jsonl(out / task / f'{split}.jsonl', rows)
            if task == 'summarization':
                raw = source('CarperAI/openai_summarize_tldr', None, 'train', n * 20)
                test_prompts = {r['prompt'] for r in read_jsonl(out / task / 'test.jsonl')}
                sft = []
                for row in raw:
                    if row['prompt'] in test_prompts: continue
                    sft.append(dict(prompt=row['prompt'], chosen=row['label'], rejected=row['label'], sft_only=True))
                    if len(sft) == n: break
                if len(sft) != n: raise RuntimeError('Not enough summary SFT examples')
                write_jsonl(out / task / 'sft.jsonl', sft)
            else:
                write_jsonl(out / task / 'sft.jsonl', read_jsonl(out / task / 'train.jsonl'))
        ood = []
        for row in source('abisee/cnn_dailymail', '3.0.0', 'test', m):
            ood.append(dict(prompt='ARTICLE: ' + row['article'] + '\nTL;DR:', chosen=row['highlights'],
                            rejected=row['highlights'], sft_only=True, source='CNN/DailyMail test'))
        write_jsonl(out / 'ood.jsonl', ood)
    else:
        raise ValueError('data_mode must be fixture or public')
    for task in config['tasks']:
        train, test = [read_jsonl(out / task / f'{split}.jsonl') for split in ('train', 'test')]
        assert_disjoint(train, test)
        assert_disjoint(read_jsonl(out / task / 'sft.jsonl'), test)
    for path in sorted(out.rglob('*.jsonl')):
        manifest['files'][str(path.relative_to(out))] = {'sha256': file_hash(path), 'rows': len(read_jsonl(path))}
    write_json(out / 'manifest.json', manifest)
    return manifest
