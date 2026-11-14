# Experiment report

Scope: **scaled public-data experiment**. Status: **completed**.

Device: `mps`; model: `openai-community/gpt2`; seeds: `[0]`.

These measurements do not reproduce the paper-scale results. GPT-4 and human judgments have not been collected.
Held-out preference accuracy measures the implicit reward ordering of provided pairs, not generated-answer quality.
Ties receive 0.5. The SFT reference therefore scores exactly 0.5 by construction; that is not a generation win rate.

![Preference accuracy](preference_accuracy.png)

| Task | Method | Seeds | Accuracy mean ± SD | Reward margin | Chosen NLL/token |
|---|---|---:|---:|---:|---:|
| sentiment | base | 1 | 0.427 ± not estimated | -0.000 | 4.627 |
| sentiment | sft | 1 | 0.500 ± not estimated | 0.000 | 4.627 |
| sentiment | dpo_beta_0.1 | 1 | 0.500 ± not estimated | 0.000 | 4.627 |
| sentiment | preferred_ft | 1 | 0.453 ± not estimated | -0.000 | 4.625 |
| sentiment | unlikelihood | 1 | 0.438 ± not estimated | -0.000 | 4.625 |
| sentiment | ppo_gt | 1 | 0.458 ± not estimated | -0.005 | 4.632 |
| summarization | base | 1 | 0.562 ± not estimated | -0.007 | 3.801 |
| summarization | sft | 1 | 0.500 ± not estimated | 0.000 | 3.793 |
| summarization | dpo_beta_0.5 | 1 | 0.438 ± not estimated | -0.005 | 3.792 |
| summarization | preferred_ft | 1 | 0.438 ± not estimated | -0.008 | 3.782 |
| dialogue | base | 1 | 0.438 ± not estimated | -0.000 | 3.171 |
| dialogue | sft | 1 | 0.500 ± not estimated | 0.000 | 3.163 |
| dialogue | dpo_beta_0.1 | 1 | 0.500 ± not estimated | 0.003 | 3.163 |
| dialogue | preferred_ft | 1 | 0.562 ± not estimated | 0.000 | 3.156 |

## Observed default-DPO comparisons

- **sentiment**: default DPO pair accuracy 0.500 (seed SD not estimated), versus the SFT identity baseline 0.500 and Preferred-FT 0.453. Chosen NLL/token changes from 4.627 to 4.627. These are held-out likelihood diagnostics, not evaluator win rates.
- **summarization**: default DPO pair accuracy 0.438 (seed SD not estimated), versus the SFT identity baseline 0.500 and Preferred-FT 0.438. Chosen NLL/token changes from 3.793 to 3.792. These are held-out likelihood diagnostics, not evaluator win rates.
- **dialogue**: default DPO pair accuracy 0.500 (seed SD not estimated), versus the SFT identity baseline 0.500 and Preferred-FT 0.562. Chosen NLL/token changes from 3.163 to 3.163. These are held-out likelihood diagnostics, not evaluator win rates.

![Sentiment diagnostics](reward_kl_and_loss.png)

## Interpretation

Compare DPO with SFT and Preferred-FT within each task. Inspect per-seed dispersion and the per-example bootstrap intervals in each `evaluation/summary.json` before drawing conclusions. Preference pairs sharing a prompt are correlated, so the example bootstrap can underestimate uncertainty. With this small corpus and few generations, these are workflow and optimization diagnostics, not statistically powered algorithm rankings.

The sentiment reward/KL plot uses untempered policy samples (temperature 1, no top-k/top-p filtering). KL is the sum of exact conditional vocabulary KL values along sampled prefixes; averaging these is a Monte Carlo sequence-KL estimate. Other temperatures and Best-of-N are not mislabeled as policy KL.

For summaries and dialogue, learned reward scores and lexical overlap are diagnostics. The same learned reward selects Best-of-N and scores its candidates, so those scores are not independent evidence of quality. Blinded forms are exported for external evaluation; unfilled forms are not counted as judgments.

## Reproducibility

`manifest.json` records the configuration, source hash, package versions, model revisions, and scope. `data/manifest.json` records source revisions and exact selected-file SHA-256 hashes. Per-stage completion receipts hash checkpoints and evaluation artifacts. Reference caches bind token IDs and labels to the frozen SFT checkpoint hash.

Run `python -m dpo_lab run --config <same config> --output <this directory> --resume` to validate and reuse completed stages. An interrupted stage restarts from its input checkpoint; mid-step optimizer/RNG restoration is not implemented.

## Remaining paper-scale work

The original GPT-2-large / GPT-J / Pythia-2.8B models, full data volumes and convergence sweeps, original RoBERTa-large sentiment oracle, TRL PPO-GT comparison and tuned PPO settings, GPT-4-0314 judgments, and human agreement study have not been reproduced in this local run. See `../../docs/PAPER_REPRODUCTION.md` for the coverage matrix.
