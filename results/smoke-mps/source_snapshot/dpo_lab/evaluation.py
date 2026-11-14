"""Generation, exact conditional KL on sampled prefixes, and offline judging."""
from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dpo_lab.runtime import seed_everything, write_json, write_jsonl
from dpo_lab.engine import mean_ci


def prompt_ids(tokenizer, prompt, max_prompt_length, task):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = ids[-max_prompt_length:] if task == 'dialogue' else ids[:max_prompt_length]
    return ids or [tokenizer.bos_token_id or tokenizer.eos_token_id]


@torch.no_grad()
def generate(model, tokenizer, prompt, cfg, device, task, temperature=1.0, n=1):
    model.eval()
    ids = torch.tensor([prompt_ids(tokenizer, prompt, cfg['max_prompt_length'], task)], device=device)
    kwargs = dict(max_new_tokens=cfg['max_new_tokens'], do_sample=temperature > 0,
                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                  use_cache=True, num_return_sequences=n)
    if temperature > 0: kwargs.update(temperature=temperature, top_k=0, top_p=1.0)
    output = model.generate(ids, attention_mask=torch.ones_like(ids), **kwargs)
    return [(seq, tokenizer.decode(seq[ids.shape[1]:], skip_special_tokens=True), ids.shape[1]) for seq in output]


@torch.no_grad()
def conditional_kl(policy, reference, sequence, prompt_length):
    """Sum KL at visited prefixes. Unbiased MC sequence KL only for untempered policy samples."""
    ids = sequence.unsqueeze(0)
    kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    p = policy(**kwargs).logits[0, prompt_length - 1:-1].float().log_softmax(-1)
    q = reference(**kwargs).logits[0, prompt_length - 1:-1].float().log_softmax(-1)
    return float((p.exp() * (p - q)).sum())


class SentimentOracle:
    def __init__(self, name, revision, device):
        self.name, self.device = name, device
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, revision=revision).to(device).eval()
        self.model.requires_grad_(False)
        labels = self.model.config.id2label
        matches = [int(k) for k, v in labels.items() if 'POS' in v.upper()]
        if len(matches) != 1: raise ValueError(f'Cannot identify positive label: {labels}')
        self.positive = matches[0]

    @torch.no_grad()
    def __call__(self, prompt, completion):
        batch = self.tokenizer(prompt + completion, return_tensors='pt', truncation=True, max_length=512).to(self.device)
        return float(self.model(**batch).logits.softmax(-1)[0, self.positive])


def fixture_oracle(prompt, completion):
    # This is deliberately named as a fixture, never a sentiment benchmark.
    return float('delightful' in completion or 'enjoyable' in completion)


def sentiment_pairs(model, tokenizer, rows, cfg, device, oracle, output, seed):
    seed_everything(seed)
    candidates, pairs = [], []
    for i, row in enumerate(rows[:cfg['sentiment_prefixes']]):
        review = row.get('review', row['prompt'] + row['chosen'])
        length = random.Random(seed + i).randint(2, 8)
        prompt = tokenizer.decode(tokenizer.encode(review, add_special_tokens=False)[:length])
        samples = generate(model, tokenizer, prompt, cfg, device, 'sentiment', 1.0, cfg['pair_candidates'])
        scored = [(text, oracle(prompt, text)) for _, text, _ in samples]
        candidates.append(dict(prompt=prompt, candidates=[dict(text=t, score=s) for t, s in scored]))
        for a in range(len(scored)):
            for b in range(a + 1, len(scored)):
                x, y = scored[a], scored[b]
                if x[0] == y[0] or abs(x[1] - y[1]) < 1e-8 or not x[0].strip() or not y[0].strip(): continue
                win, lose = (x, y) if x[1] > y[1] else (y, x)
                pairs.append(dict(prompt=prompt, chosen=win[0], rejected=lose[0], chosen_score=win[1],
                                  rejected_score=lose[1], source='SFT samples ranked by sentiment oracle'))
    write_jsonl(Path(output) / 'candidates.jsonl', candidates)
    write_jsonl(Path(output) / 'pairs.jsonl', pairs)
    if not pairs: raise ValueError('No non-tied sentiment pairs; increase prefixes/candidates')
    return pairs


@torch.no_grad()
def evaluate_generations(model, reference, tokenizer, rows, cfg, device, task, output, seed, scorer=None, best_of=1):
    records = []
    for temperature in cfg['temperatures']:
        if best_of > 1 and temperature == 0: continue
        for i, row in enumerate(rows[:cfg['generation_examples']]):
            seed_everything(seed + i)  # Common random numbers across compared methods.
            samples = generate(model, tokenizer, row['prompt'], cfg, device, task, temperature, best_of)
            ranked = [(seq, text, prefix, scorer(row['prompt'], text) if scorer else None) for seq, text, prefix in samples]
            seq, text, prefix, score = max(ranked, key=lambda x: x[3]) if scorer and best_of > 1 else ranked[0]
            record = dict(index=i, prompt=row['prompt'], completion=text, baseline=row['chosen'], temperature=temperature,
                          generated_tokens=len(seq) - prefix, best_of=best_of, score=score)
            if temperature == 1.0 and best_of == 1:
                record['sequence_kl_mc'] = conditional_kl(model, reference, seq, prefix)
            if task == 'summarization':
                # Lexical diagnostic only; not the paper's GPT-4 or human win rate.
                a, b = set(text.lower().split()), set(row['chosen'].lower().split())
                record['token_overlap_f1'] = 2 * len(a & b) / max(1, len(a) + len(b))
            records.append(record)
    write_jsonl(output, records)
    summary = {}
    for temperature in cfg['temperatures']:
        subset = [r for r in records if r['temperature'] == temperature]
        summary[str(temperature)] = {key: mean_ci([r[key] for r in subset if r.get(key) is not None])
                                    for key in ('score', 'sequence_kl_mc', 'token_overlap_f1', 'generated_tokens')}
    return summary


def export_judgments(records, task, path, seed=0):
    """Blinded forms; answers are never generated or inferred by this function."""
    rng = random.Random(seed); questions, key = [], []
    for i, row in enumerate(records):
        reverse = rng.choice([True, False])
        answers = [row['completion'], row['baseline']]
        if reverse: answers.reverse()
        questions.append(dict(id=str(i), task=task, prompt=row['prompt'], answer_a=answers[0], answer_b=answers[1],
                              criterion='Choose the more helpful response.' if task == 'dialogue' else 'Choose the more accurate and concise summary.',
                              winner=None, evaluator=None))
        key.append(dict(id=str(i), candidate='B' if reverse else 'A'))
    write_jsonl(path, questions)
    write_jsonl(str(path) + '.key.jsonl', key)


def score_judgments(questions, key):
    mapping = {r['id']: r['candidate'] for r in key}
    seen, scores = set(), []
    for row in questions:
        if row['id'] in seen or row['id'] not in mapping: raise ValueError('Duplicate or unknown judgment ID')
        seen.add(row['id'])
        if row.get('winner') not in {'A', 'B', 'tie'} or not row.get('evaluator'):
            raise ValueError('Each judgment requires winner A/B/tie and evaluator provenance')
        scores.append(0.5 if row['winner'] == 'tie' else float(row['winner'] == mapping[row['id']]))
    if set(mapping) != seen: raise ValueError('Missing judgments')
    return {'win_rate': mean_ci(scores), 'evaluators': sorted({r['evaluator'] for r in questions})}
