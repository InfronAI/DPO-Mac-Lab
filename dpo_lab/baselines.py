"""Bradley–Terry reward model and token-level clipped PPO comparison baselines.

These are independent baselines. The DPO training path never constructs either.
"""
from __future__ import annotations
import json
import random
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from dpo_lab.models import load_model, save_model
from dpo_lab.engine import batch_for
from dpo_lab.evaluation import generate, prompt_ids
from dpo_lab.runtime import seed_everything, write_json, write_jsonl


class RewardModel(nn.Module):
    def __init__(self, causal):
        super().__init__()
        self.backbone = causal.base_model
        self.head = nn.Linear(causal.config.hidden_size, 1, bias=False).to(next(causal.parameters()).device)

    def forward(self, ids, mask):
        hidden = self.backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        # Training batches are right padded; generation scoring uses one unpadded row.
        index = mask.sum(1) - 1
        return self.head(hidden[torch.arange(len(ids), device=ids.device), index]).squeeze(-1)


def train_reward(model_path, tokenizer, items, cfg, device, output, seed):
    seed_everything(seed)
    causal = load_model(model_path, device)
    model = RewardModel(causal)
    del causal
    optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg['reward_lr'])
    logs = []; model.train()
    for epoch in range(cfg['reward_epochs']):
        order = list(range(len(items))); random.Random(seed + epoch).shuffle(order)
        for start in range(0, len(order), cfg['batch_size']):
            indices = order[start:start + cfg['batch_size']]
            optimizer.zero_grad(set_to_none=True); losses = []
            for i in indices:
                batch = batch_for([items[i]], tokenizer, device)
                c = model(batch['chosen_input_ids'], batch['chosen_attention_mask'])
                r = model(batch['rejected_input_ids'], batch['rejected_attention_mask'])
                loss = -F.logsigmoid(c - r).mean()
                (loss / len(indices)).backward(); losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
            optimizer.step()
            logs.append(dict(epoch=epoch, step=len(logs)+1, loss=sum(losses)/len(losses)))
    Path(output).mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, Path(output) / 'reward.pt')
    write_jsonl(Path(output) / 'metrics.jsonl', logs)
    model.requires_grad_(False).eval()
    return model


def reward_scorer(model, tokenizer, cfg, device, task):
    @torch.no_grad()
    def score(prompt, completion):
        ids = prompt_ids(tokenizer, prompt, cfg['max_prompt_length'], task)
        response = tokenizer.encode(completion, add_special_tokens=False)
        ids = (ids + response + [tokenizer.eos_token_id])[:cfg['max_length']]
        ids = torch.tensor([ids], device=device)
        return model(ids, torch.ones_like(ids)).item()
    return score


@torch.no_grad()
def reward_accuracy(model, tokenizer, items, device):
    scores = []
    for item in items:
        b = batch_for([item], tokenizer, device)
        margin = (model(b['chosen_input_ids'], b['chosen_attention_mask']) - model(b['rejected_input_ids'], b['rejected_attention_mask'])).item()
        scores.append(float(margin > 0) if abs(margin) > 1e-7 else 0.5)
    return sum(scores)/len(scores)


def token_logps(output, sequence, start):
    logits = output.logits[0, start - 1:-1].float()
    targets = sequence[start:]
    return logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def gae(rewards, values, lam):
    """Undiscounted episodic GAE; terminal value is exactly zero."""
    advantage = torch.zeros_like(rewards); running = torch.zeros((), device=rewards.device)
    for t in reversed(range(len(rewards))):
        following = values[t+1] if t + 1 < len(values) else 0
        delta = rewards[t] + following - values[t]
        running = delta + lam * running
        advantage[t] = running
    return advantage, advantage + values


def train_ppo(policy, reference, tokenizer, rows, scorer, cfg, device, task, output, seed):
    seed_everything(seed)
    head = nn.Linear(policy.config.hidden_size, 1).to(device)
    nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
    optimizer = torch.optim.RMSprop(list(policy.parameters()) + list(head.parameters()), lr=cfg['preference_lr'])
    logs = []; started = time.monotonic()
    reference.eval().requires_grad_(False)
    for step in range(cfg['ppo_steps']):
        rollouts = []
        for j in range(cfg['ppo_rollout_batch']):
            row = rows[(step * cfg['ppo_rollout_batch'] + j) % len(rows)]
            seq, completion, start = generate(policy, tokenizer, row['prompt'], cfg, device, task, 1.)[0]
            with torch.no_grad():
                old = policy(seq[None], output_hidden_states=True, use_cache=False)
                old_logps = token_logps(old, seq, start)
                old_values = head(old.hidden_states[-1][0, start-1:-1]).squeeze(-1)
                ref = reference(seq[None], use_cache=False)
                ref_logps = token_logps(ref, seq, start)
                reward = scorer(row['prompt'], completion)
                rewards = -cfg['ppo_kl_coef'] * (old_logps - ref_logps)
                rewards[-1] += reward
                adv, returns = gae(rewards, old_values, cfg['ppo_lambda'])
            rollouts.append(dict(seq=seq, start=start, old_logps=old_logps.detach(), old_values=old_values.detach(),
                                 advantages=adv.detach(), returns=returns.detach(), reward=reward))
        all_adv = torch.cat([r['advantages'] for r in rollouts])
        std = all_adv.std(unbiased=False).clamp_min(1e-8)
        for rollout in rollouts: rollout['advantages'] = (rollout['advantages']-all_adv.mean()) / std
        policy.train()
        for epoch in range(cfg['ppo_epochs']):
            optimizer.zero_grad(set_to_none=True); actor_losses, critic_losses = [], []
            for rollout in rollouts:
                current = policy(rollout['seq'][None], output_hidden_states=True, use_cache=False)
                lp = token_logps(current, rollout['seq'], rollout['start'])
                values = head(current.hidden_states[-1][0, rollout['start']-1:-1]).squeeze(-1)
                ratio = (lp - rollout['old_logps']).exp()
                clipped = ratio.clamp(1-cfg['ppo_clip'], 1+cfg['ppo_clip'])
                actor = -torch.minimum(ratio * rollout['advantages'], clipped * rollout['advantages']).mean()
                value_clipped = rollout['old_values'] + (values-rollout['old_values']).clamp(-cfg['ppo_clip'], cfg['ppo_clip'])
                critic = .5 * torch.maximum((values-rollout['returns'])**2, (value_clipped-rollout['returns'])**2).mean()
                loss = (actor + cfg['ppo_value_coef'] * critic) / len(rollouts)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite PPO loss')
                loss.backward(); actor_losses.append(actor.item()); critic_losses.append(critic.item())
            grad = torch.nn.utils.clip_grad_norm_(list(policy.parameters())+list(head.parameters()), cfg['max_grad_norm'])
            if not torch.isfinite(grad): raise FloatingPointError('Nonfinite PPO gradients')
            optimizer.step()
            logs.append(dict(step=step+1, epoch=epoch, actor_loss=sum(actor_losses)/len(actor_losses),
                             value_loss=sum(critic_losses)/len(critic_losses), reward=sum(r['reward'] for r in rollouts)/len(rollouts)))
        print(f'ppo rollout={step+1} reward={logs[-1]["reward"]:.4f}', flush=True)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    save_model(policy, tokenizer, output / 'model')
    torch.save(head.state_dict(), output / 'value_head.pt')
    write_jsonl(output / 'metrics.jsonl', logs)
    write_json(output / 'training.json', dict(method='ppo', updates=len(logs), rollouts=cfg['ppo_steps']*cfg['ppo_rollout_batch'],
                                             elapsed_seconds=time.monotonic()-started, seed=seed, kl_coef=cfg['ppo_kl_coef']))
    return logs
