"""Memory-bounded training with cached frozen reference likelihoods."""
from __future__ import annotations
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from preference_datasets import tokenize_batch_element, get_collate_fn
from dpo_lab.objectives import _get_batch_logps, preference_loss
from dpo_lab.runtime import canonical_hash, seed_everything, synchronize, write_json, write_jsonl, read_jsonl
from dpo_lab.models import save_model


def encode_rows(rows, tokenizer, cfg, task):
    mode = 'keep_end' if task == 'dialogue' else 'keep_start'
    result = [tokenize_batch_element(r['prompt'], r['chosen'], r['rejected'], mode,
                                    tokenizer, cfg['max_length'], cfg['max_prompt_length']) for r in rows]
    # Preference truncation may erase the difference. SFT-only rows intentionally match.
    for item, row in zip(result, rows):
        if not row.get('sft_only') and item['chosen_input_ids'] == item['rejected_input_ids']:
            item['indistinguishable'] = True
    return result


def batch_for(items, tokenizer, device):
    # Metadata differs between rows and is not needed by the model.
    clean = [{k: v for k, v in x.items() if k.endswith(('_input_ids', '_attention_mask', '_labels'))} for x in items]
    batch = get_collate_fn(tokenizer)(clean)
    return {k: v.to(device) for k, v in batch.items()}


def logps(model, batch, side='chosen', average=False):
    logits = model(input_ids=batch[f'{side}_input_ids'], attention_mask=batch[f'{side}_attention_mask'], use_cache=False).logits.float()
    return _get_batch_logps(logits, batch[f'{side}_labels'], average)


@torch.no_grad()
def cache_reference(model, items, tokenizer, device, path, identity):
    """Compute once on identical tokens; both sum (DPO) and mean (IPO) retained."""
    fingerprint = canonical_hash({'identity': identity, 'tokens': [
        {k: v for k, v in x.items() if k.endswith(('_input_ids', '_labels'))} for x in items]})
    path = Path(path)
    if path.exists():
        cache = json.loads(path.read_text())
        if cache['fingerprint'] != fingerprint: raise ValueError(f'Stale reference cache: {path}')
        return cache['values']
    model.eval()
    values = []
    for item in items:
        batch = batch_for([item], tokenizer, device)
        row = {}
        for side in ('chosen', 'rejected'):
            total = logps(model, batch, side).item()
            count = int((batch[f'{side}_labels'][:, 1:] != -100).sum())
            row[side] = total
            row[side + '_mean'] = total / count
        values.append(row)
    write_json(path, {'fingerprint': fingerprint, 'identity': identity, 'values': values})
    return values


def train_stage(model, tokenizer, items, refs, cfg, method, output, device, seed, beta=0.1, smoothing=0.0):
    """A stage is atomic at completion; interrupted stages can be rerun from their input."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    lr = cfg['sft_lr'] if method in {'sft', 'preferred_ft'} else cfg['preference_lr']
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    epochs = cfg['sft_epochs'] if method == 'sft' else cfg['preference_epochs']
    logs, examples, step = [], 0, 0
    model.train(); started = time.monotonic()
    for epoch in range(epochs):
        order = list(range(len(items))); random.Random(seed + epoch).shuffle(order)
        for start in range(0, len(order), cfg['batch_size']):
            indices = order[start:start + cfg['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            scale = min(1.0, (step + 1) / max(1, cfg['warmup_steps']))
            for group in optimizer.param_groups: group['lr'] = lr * scale
            losses, margins = [], []
            for mb in range(0, len(indices), cfg['microbatch_size']):
                ix = indices[mb:mb + cfg['microbatch_size']]
                batch = batch_for([items[i] for i in ix], tokenizer, device)
                chosen = logps(model, batch, average=method == 'ipo')
                if method in {'sft', 'preferred_ft'}:
                    loss_vector = -chosen
                else:
                    rejected = logps(model, batch, 'rejected', average=method == 'ipo')
                    if method == 'unlikelihood':
                        # Paper Appendix C.3: log p(y_w) - alpha log p(y_l).
                        loss_vector = -chosen + cfg['unlikelihood_alpha'] * rejected
                    else:
                        suffix = '_mean' if method == 'ipo' else ''
                        rc = torch.tensor([refs[i]['chosen' + suffix] for i in ix], device=device)
                        rr = torch.tensor([refs[i]['rejected' + suffix] for i in ix], device=device)
                        loss_vector, cw, rw = preference_loss(chosen, rejected, rc, rr, beta,
                                                             label_smoothing=smoothing, ipo=method == 'ipo',
                                                             reference_free=method == 'reference_free')
                        margins.extend((cw - rw).tolist())
                if not torch.isfinite(loss_vector).all(): raise FloatingPointError('Nonfinite training loss')
                (loss_vector.sum() / len(indices)).backward()
                losses.extend(loss_vector.detach().tolist())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
            if not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradient')
            optimizer.step(); synchronize(device)
            examples += len(indices); step += 1
            record = dict(step=step, epoch=epoch, examples=examples, loss=float(np.mean(losses)),
                          grad_norm=float(norm), lr=lr * scale, elapsed_seconds=time.monotonic() - started)
            if margins: record['reward_margin'] = float(np.mean(margins))
            logs.append(record)
            # Every step is flushed so interrupted work remains inspectable.
            with (output / 'metrics.jsonl').open('a' if step > 1 else 'w') as f:
                f.write(json.dumps(record) + '\n')
            if step == 1 or step % 8 == 0: print(f'{method} step={step} loss={record["loss"]:.4f}', flush=True)
    save_model(model, tokenizer, output / 'model')
    write_json(output / 'training.json', {'method': method, 'seed': seed, 'beta': beta, 'label_smoothing': smoothing,
               'updates': step, 'examples': examples, 'elapsed_seconds': time.monotonic() - started,
               'reference_cached': refs is not None, 'optimizer': 'RMSprop', 'learning_rate': lr})
    return logs


def mean_ci(values, seed=0):
    """Percentile bootstrap over examples; descriptive with small held-out sets."""
    x = np.asarray(values, dtype=float)
    if len(x) == 0: return {'mean': None, 'ci95': [None, None], 'n': 0}
    rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(x, size=(1000, len(x)), replace=True), axis=1)
    return {'mean': float(x.mean()), 'ci95': np.quantile(boot, [0.025, 0.975]).tolist(), 'n': len(x)}


@torch.no_grad()
def evaluate_pairs(model, tokenizer, items, refs, cfg, device, beta, output):
    model.eval(); rows = []
    for i, item in enumerate(items):
        b = batch_for([item], tokenizer, device)
        c, r = logps(model, b), logps(model, b, 'rejected')
        rc, rr = (torch.tensor([refs[i][s]], device=device) for s in ('chosen', 'rejected'))
        loss, cw, rw = preference_loss(c, r, rc, rr, beta)
        margin = (cw - rw).item()
        count = int((b['chosen_labels'][:, 1:] != -100).sum())
        rows.append(dict(index=i, loss=loss.item(), reward_margin=margin,
                         preference_accuracy=1.0 if margin > 1e-5 else 0.0 if margin < -1e-5 else 0.5,
                         chosen_nll_per_token=-c.item() / count,
                         policy_logp_chosen=c.item(), policy_logp_rejected=r.item(),
                         reference_logp_chosen=rc.item(), reference_logp_rejected=rr.item()))
    write_jsonl(output, rows)
    return {k: mean_ci([r[k] for r in rows]) for k in ('loss', 'reward_margin', 'preference_accuracy', 'chosen_nll_per_token')}
