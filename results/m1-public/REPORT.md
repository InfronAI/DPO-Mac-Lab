# Experiment report

Scope: **scaled public-data experiment**. Status: **completed**.

Device: `mps`; model: `EleutherAI/pythia-70m-deduped`; seeds: `[0, 1, 2]`.

These measurements do not reproduce the paper-scale results. GPT-4 and human judgments have not been collected.
Held-out preference accuracy measures the implicit reward ordering of provided pairs, not generated-answer quality.
Ties receive 0.5. The SFT reference therefore scores exactly 0.5 by construction; that is not a generation win rate.

![Preference accuracy](preference_accuracy.png)

| Task | Method | Seeds | Accuracy mean ± SD | Reward margin | Chosen NLL/token |
|---|---|---:|---:|---:|---:|
| sentiment | base | 3 | 0.521 ± 0.048 | 0.051 | 53.593 |
| sentiment | sft | 3 | 0.500 ± 0.000 | 0.000 | 51.408 |
| sentiment | dpo_beta_0.05 | 3 | 0.524 ± 0.032 | 0.000 | 51.429 |
| sentiment | dpo_beta_0.1 | 3 | 0.514 ± 0.049 | 0.010 | 51.433 |
| sentiment | dpo_beta_1 | 3 | 0.528 ± 0.012 | 0.076 | 51.433 |
| sentiment | dpo_beta_5 | 3 | 0.521 ± 0.089 | 0.247 | 51.425 |
| sentiment | preferred_ft | 3 | 0.500 ± 0.091 | 0.609 | 12.802 |
| sentiment | cdpo | 3 | 0.545 ± 0.032 | 0.012 | 51.434 |
| sentiment | ipo | 3 | 0.521 ± 0.068 | -0.019 | 51.425 |
| sentiment | reference_free | 3 | 0.500 ± 0.031 | -0.001 | 51.453 |
| sentiment | ppo | 3 | 0.479 ± 0.068 | -0.004 | 51.430 |
| sentiment | unlikelihood | 3 | 0.479 ± 0.055 | -0.042 | 51.091 |
| sentiment | ppo_gt | 3 | 0.538 ± 0.067 | 0.008 | 51.442 |
| summarization | base | 3 | 0.500 ± 0.062 | -21.384 | 38.116 |
| summarization | sft | 3 | 0.500 ± 0.000 | 0.000 | 11.919 |
| summarization | dpo_beta_0.5 | 3 | 0.458 ± 0.191 | -1.228 | 12.041 |
| summarization | preferred_ft | 3 | 0.458 ± 0.095 | -17.203 | 9.787 |
| summarization | cdpo | 3 | 0.438 ± 0.062 | 0.959 | 12.057 |
| summarization | ipo | 3 | 0.542 ± 0.072 | 2.998 | 11.812 |
| summarization | reference_free | 3 | 0.438 ± 0.165 | -5.431 | 11.007 |
| summarization | ppo | 3 | 0.542 ± 0.095 | -0.487 | 11.965 |
| dialogue | base | 3 | 0.646 ± 0.072 | 17.585 | 42.024 |
| dialogue | sft | 3 | 0.500 ± 0.000 | 0.000 | 11.015 |
| dialogue | dpo_beta_0.1 | 3 | 0.438 ± 0.165 | 0.155 | 10.987 |
| dialogue | preferred_ft | 3 | 0.521 ± 0.095 | 5.110 | 6.965 |
| dialogue | cdpo | 3 | 0.604 ± 0.130 | 0.704 | 11.010 |
| dialogue | ipo | 3 | 0.521 ± 0.180 | 0.150 | 11.171 |
| dialogue | reference_free | 3 | 0.562 ± 0.165 | -0.105 | 10.514 |
| dialogue | ppo | 3 | 0.583 ± 0.130 | 0.138 | 11.311 |

## Observed default-DPO comparisons

- **sentiment**: default DPO pair accuracy 0.514 (seed SD 0.049), versus the SFT identity baseline 0.500 and Preferred-FT 0.500. Chosen NLL/token changes from 51.408 to 51.433. These are held-out likelihood diagnostics, not evaluator win rates.
- **summarization**: default DPO pair accuracy 0.458 (seed SD 0.191), versus the SFT identity baseline 0.500 and Preferred-FT 0.458. Chosen NLL/token changes from 11.919 to 12.041. These are held-out likelihood diagnostics, not evaluator win rates.
- **dialogue**: default DPO pair accuracy 0.438 (seed SD 0.165), versus the SFT identity baseline 0.500 and Preferred-FT 0.521. Chosen NLL/token changes from 11.015 to 10.987. These are held-out likelihood diagnostics, not evaluator win rates.

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

## EOS and generation diagnostic

A separate fixed-sentence CPU diagnostic measured an EOS log-probability of approximately −1232.6 while ordinary tokens averaged approximately −2.93. The causal-mask check passed. This unusually large EOS term can dominate the short-sequence SFT loss and makes NLL reductions difficult to interpret as fluency improvements. EOS was retained to preserve the original objective. Some saved generations repeat phrases. See `../../docs/VALIDATION.md` and the separate GPT-2 experiment; the model and optimizer configuration both change there, so it is not a single-factor comparison.
