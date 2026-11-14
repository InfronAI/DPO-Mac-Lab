# Saved experiment evidence

**122 public-data evaluation variants** completed on this M1 Mac, plus separate 32-variant CPU and MPS synthetic smoke workflows. “Variant” includes untrained baselines and Best-of-N evaluations; it is not a count of independent training jobs.

| Run | Model | Seeds | Evaluated variants | Report |
|---|---|---:|---:|---|
| m1-public | EleutherAI/pythia-70m-deduped | 3 | 105 | [Report](m1-public/REPORT.md) |
| gpt2-public | openai-community/gpt2 | 1 | 17 | [Report](gpt2-public/REPORT.md) |

## Default DPO measurements

These are held-out implicit reward accuracies, not GPT-4 or human win rates. The SFT reference receives 0.5 for tied rewards by definition.

| Model | Task | DPO accuracy | Seed SD | Preferred-FT accuracy |
|---|---|---:|---:|---:|
| m1-public | sentiment | 0.514 | 0.049 | 0.500 |
| m1-public | summarization | 0.458 | 0.191 | 0.458 |
| m1-public | dialogue | 0.438 | 0.165 | 0.521 |
| gpt2-public | sentiment | 0.500 | N/A (one seed) | 0.453 |
| gpt2-public | summarization | 0.438 | N/A (one seed) | 0.438 |
| gpt2-public | dialogue | 0.500 | N/A (one seed) | 0.562 |

## What the data supports

The Pythia matrix does not establish a consistent DPO improvement in this small setting. Saved generations include repetitive text. A numerical investigation identified an unusually low EOS probability in that checkpoint; EOS remains in the objective. The separate GPT-2 profile changes both the model and optimizer settings, so its results are a practical diagnostic and cannot isolate the cause of any difference.

Only 64 training source examples and a short context are used. PPO rollout budgets, DPO epochs, and Best-of-N inference costs differ. The generation sample is very small, and the sentiment seeds also change preference sampling. These runs demonstrate execution and expose limitations; they cannot verify the paper’s algorithm rankings.

## Files and verification

- Each run contains source/model/data revisions, raw selected examples, per-step training JSONL, per-example likelihood measurements, generation samples, blinded judgment forms and SHA-256 completion receipts.
- Large model checkpoints and reward/value weights remain locally available, but are excluded from Git. Fresh clones should run a new experiment directory.
- [CPU smoke](smoke-cpu/REPORT.md), [MPS smoke](smoke-mps/REPORT.md), [test output](validation/pytest.txt).
- [Numerical validation and timing](../docs/VALIDATION.md): Pythia passed the CPU/MPS thresholds; GPT-2 failed the strict 2% per-parameter gradient threshold in two cases (maximum 2.121%, whole-model error 0.056%). These failures are retained. The reference-cache microbenchmark measured 1.25× forward/backward throughput, excluding cache construction and optimizer updates.
- [Original Hydra validation on CPU](validation/legacy-cpu.json) and [MPS](validation/legacy-mps.json).
- [Paper coverage and unexecuted research](../docs/PAPER_REPRODUCTION.md). Full-scale models, original tuned controls, historical GPT-4 and human evaluation remain uncompleted.
