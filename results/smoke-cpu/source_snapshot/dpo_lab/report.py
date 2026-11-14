"""Standalone plots and evidence tables; never invent missing evaluations."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dpo_lab.runtime import read_jsonl,write_json


def report(directory):
    root=Path(directory)
    manifest=json.loads((root/'manifest.json').read_text())
    summaries=json.loads((root/'summary.json').read_text())
    records=[]
    for s in summaries:
        if s['pairs'] is None: continue
        records.append(dict(task=s['task'],seed=s['seed'],method=s['method'],beta=s['beta'],
                            accuracy=s['pairs']['preference_accuracy']['mean'],
                            margin=s['pairs']['reward_margin']['mean'],
                            nll=s['pairs']['chosen_nll_per_token']['mean'],
                            loss=s['pairs']['loss']['mean'],pairs=s['pairs']['preference_accuracy']['n']))
    with (root/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    groups=defaultdict(list)
    for row in records: groups[(row['task'],row['method'])].append(row)
    aggregate=[]
    for (task,method),rows in groups.items():
        aggregate.append(dict(task=task,method=method,seeds=len(rows),accuracy_mean=float(np.mean([r['accuracy'] for r in rows])),
                              accuracy_sd=float(np.std([r['accuracy'] for r in rows],ddof=1)) if len(rows)>1 else 0.,
                              margin_mean=float(np.mean([r['margin'] for r in rows])),
                              nll_mean=float(np.mean([r['nll'] for r in rows])),pairs=[r['pairs'] for r in rows]))
    write_json(root/'aggregate.json',aggregate)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f8fafc','axes.facecolor':'#f8fafc'})
    fig,axes=plt.subplots(1,len(manifest['config']['tasks']),figsize=(17,5.5),squeeze=False)
    for ax,task in zip(axes[0],manifest['config']['tasks']):
        rows=[r for r in aggregate if r['task']==task]
        names=[r['method'] for r in rows]; values=[r['accuracy_mean'] for r in rows]
        ax.barh(names,values,xerr=[r['accuracy_sd'] for r in rows],color=['#0d9488' if m.startswith('dpo') else '#64748b' for m in names],capsize=3)
        ax.axvline(.5,color='#9ca3af',linestyle='--',linewidth=1)
        ax.set_xlim(0,1); ax.set_title(task.title()); ax.set_xlabel('Held-out implicit reward accuracy (ties = 0.5)'); ax.invert_yaxis()
    fig.suptitle('Scaled experiments · mean ± sample SD across training seeds',fontsize=15)
    fig.tight_layout(); fig.savefig(root/'preference_accuracy.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for s in summaries:
        if s['task']!='sentiment' or s['pairs'] is None: continue
        gen=s['generation'].get('1.0',{})
        kl=gen.get('sequence_kl_mc',{}).get('mean'); reward=gen.get('score',{}).get('mean')
        if kl is not None and reward is not None:
            axes[0].scatter(kl,reward,label=s['method'] if s['seed']==manifest['config']['seeds'][0] else None,s=30)
        path=root/f'seed_{s["seed"]}'/s['task']/s['method']/'metrics.jsonl'
        if s['method'].startswith('dpo') and path.exists():
            logs=read_jsonl(path)
            axes[1].plot([r['step'] for r in logs],[r['loss'] for r in logs],alpha=.7,label=s['method'] if s['seed']==manifest['config']['seeds'][0] else None)
    axes[0].set(xlabel='MC sequence KL at temperature 1',ylabel='Sentiment oracle probability',title='Reward / KL diagnostic (few samples)')
    axes[1].set(xlabel='Optimizer updates',ylabel='Training DPO loss',title='Sentiment optimization traces')
    for ax in axes: ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(root/'reward_kl_and_loss.png',dpi=160); plt.close(fig)
    lines=['# Experiment report','',f"Scope: **{manifest['scope']}**. Status: **{manifest['status']}**.",'',
           f"Device: `{manifest['environment']['device']}`; model: `{manifest['config']['model']}`; seeds: `{manifest['config']['seeds']}`.",'',
           'These measurements do not reproduce the paper-scale results. GPT-4 and human judgments have not been collected.',
           'Held-out preference accuracy measures the implicit reward ordering of provided pairs, not generated-answer quality.',
           'Ties receive 0.5. The SFT reference therefore scores exactly 0.5 by construction; that is not a generation win rate.','',
           '![Preference accuracy](preference_accuracy.png)','',
           '| Task | Method | Seeds | Accuracy mean ± SD | Reward margin | Chosen NLL/token |',
           '|---|---|---:|---:|---:|---:|']
    for r in aggregate:
        lines.append(f"| {r['task']} | {r['method']} | {r['seeds']} | {r['accuracy_mean']:.3f} ± {r['accuracy_sd']:.3f} | {r['margin_mean']:.3f} | {r['nll_mean']:.3f} |")
    lines += ['', '![Sentiment diagnostics](reward_kl_and_loss.png)','', '## Interpretation','',
              'Compare DPO with SFT and Preferred-FT within each task. Inspect per-seed dispersion and the per-example bootstrap intervals in each `evaluation/summary.json` before drawing conclusions. Preference pairs sharing a prompt are correlated, so the example bootstrap can underestimate uncertainty. With this small corpus and few generations, these are workflow and optimization diagnostics, not statistically powered algorithm rankings.',
              '', 'The sentiment reward/KL plot uses untempered policy samples (temperature 1, no top-k/top-p filtering). KL is the sum of exact conditional vocabulary KL values along sampled prefixes; averaging these is a Monte Carlo sequence-KL estimate. Other temperatures and Best-of-N are not mislabeled as policy KL.',
              '', 'For summaries and dialogue, learned reward scores and lexical overlap are diagnostics. The same learned reward selects Best-of-N and scores its candidates, so those scores are not independent evidence of quality. Blinded forms are exported for external evaluation; unfilled forms are not counted as judgments.',
              '', '## Reproducibility','',
              '`manifest.json` records the configuration, source hash, package versions, model revisions, and scope. `data/manifest.json` records source revisions and exact selected-file SHA-256 hashes. Per-stage completion receipts hash checkpoints and evaluation artifacts. Reference caches bind token IDs and labels to the frozen SFT checkpoint hash.',
              '', 'Run `python -m dpo_lab run --config <same config> --output <this directory> --resume` to validate and reuse completed stages. An interrupted stage restarts from its input checkpoint; mid-step optimizer/RNG restoration is not implemented.',
              '', '## Remaining paper-scale work','',
              'The original GPT-2-large / GPT-J / Pythia-2.8B models, full data volumes and convergence sweeps, original RoBERTa-large sentiment oracle, TRL PPO-GT comparison and tuned PPO settings, GPT-4-0314 judgments, and human agreement study have not been reproduced in this local run. See `../../docs/PAPER_REPRODUCTION.md` for the coverage matrix.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(f'Wrote {root}/REPORT.md, metrics.csv, aggregate.json and two plots')
