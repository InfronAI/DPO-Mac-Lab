# Paper coverage and reproduction limits

Primary source: [Rafailov et al., DPO, arXiv:2305.18290v3](https://arxiv.org/html/2305.18290v3), Sections 4 and 6 and Appendices B–D. Original repository baseline: commit `f8b8c0f49dc92a430bae41585f9d467d3618fe2f`. This audit distinguishes implementing a workflow, executing a reduced experiment, and reproducing a published result.

**This project does not claim 100% numerical or experimental reproduction of the paper.** A 16 GB M1 cannot hold the original GPT-J full-parameter optimizer plus reference model. External historical GPT-4 judgments and a human study cannot be replaced by local proxy metrics.

| Component | Original repository | Added implementation | Local execution / limitation |
|---|---|---|---|
| SFT → frozen SFT reference → DPO | Present; CUDA assumptions | MPS/CPU selection, portable weights, checked cached reference probabilities | Real full-parameter 70M and synthetic tiny-model runs |
| Original DPO Eq. 7 | Present | Shared tested implementation, positive beta validation | Executed; sequence log-probability sums preserved |
| IMDb controlled sentiment | Missing | Review SFT, 2–8 token prefixes, four sampled completions, classifier ranking, all non-tied pairs | Reduced corpus and DistilBERT SST-2 oracle; not original GPT-2-large/RoBERTa-large |
| TL;DR summarization | Missing adapter/workflow | Human-summary SFT and OpenAI comparison data | Small-model subset; independent SFT and preference samples |
| HH single-turn dialogue | HH training available | Single-turn extraction, disjoint held-out prompts, generation evaluation | Bounded HH subset; 70M replaces Pythia-2.8B |
| Preferred-FT | Possible manually | Matched SFT initialization and explicit baseline | Executed |
| Unlikelihood, Appendix C.3 | Missing | Chosen log-likelihood minus weighted rejected log-likelihood | Executed on sentiment; can produce degeneration |
| Explicit reward model | Missing | Transformer + scalar head, Bradley–Terry loss | Final epoch, no model-selection validation sweep in local profile |
| PPO / PPO-GT | Missing | Token-level GAE, clipped actor and value objectives, frozen-reference KL, learned or oracle reward | Bounded local implementation; not the paper's separate TRL implementation or tuned large-rollout study |
| Best-of-N | Missing | Sample candidates and select with separately trained reward model | N=1,4 locally; config accepts larger N |
| Reward/KL frontier | Missing | Oracle reward and conditional full-vocabulary KL along untempered sampled prefixes | Final checkpoints and four prompts; not convergence-frontier evidence |
| Beta sweep | Manual commands only | Sentiment β ∈ {0.05,0.1,1,5} | Three training seeds; shared selected data |
| Temperature sweep | Sampling only | Greedy, 0.25 and 1.0 generation | Other temperatures configurable; small sample sizes |
| CNN/DailyMail distribution shift | Missing | Held-out article generation from summarization checkpoints | Local diagnostics and blinded evaluation exports |
| GPT-4 quality judging | Missing | Blinded comparison export and validated result import | **Not executed**; no fabricated or proxy “GPT-4 win rates” |
| Human judgment validation | Missing | Blinded labels plus aligned agreement / Cohen kappa import | **Not executed**; no recruitment or agreement result claimed |
| Base prompting baselines | Not packaged | Base-model task prompt generation | Original GPT-J and Pythia two-shot baselines not reproduced |
| cDPO and IPO extensions | Loss code; missing IPO config | Runnable configs, IPO average response log-probability, cDPO smoothing | Executed; these postdate the original DPO paper |
| Multi-GPU training | FSDP / tensor parallel | Preserved, CUDA guard, optional tensor-parallel import | **Not hardware-tested** on this Mac |

## What “complete local run” means

A local run finishes data acquisition, SFT, preference construction/loading, frozen-reference caching, optimization, held-out likelihood evaluation, sampling, baseline comparisons, out-of-distribution generation, and artifact export. It is a scaled workflow validation. A successful run does not imply matching the paper's response quality, training convergence, sample complexity, or numerical tables.

The additional `gpt2-mac.json` profile uses GPT-2 and the paper’s learning-rate/warmup settings for a one-seed diagnostic with fewer baselines. It is recorded separately; it does not replace the full small-model matrix or establish a causal attribution for model-quality differences.

The `mac.json` profile intentionally uses 64 SFT/preference source examples, 16 held-out source examples, short contexts, Pythia-70M, and three seeds. Sentiment converts 16 source prefixes per split into up to 96 preference pairs. Empty/tied/truncated-identical pairs are excluded and counts are recorded. The test prefix set is independent of the training prefixes. The corpus is a reproducible bounded streaming sample, not an unbiased sample of every dataset row.

## Evaluation interpretation

- **Implicit reward accuracy** ranks existing held-out chosen/rejected answers using β(log π − log π_ref). It is not a generated-answer win rate. SFT ties receive 0.5. Reference-free and IPO trained models are also assessed with this common DPO diagnostic; their own training objectives differ.
- **Sequence KL** uses full-vocabulary conditional KL at prefixes sampled from the untempered policy. Only temperature 1, top-k=0, top-p=1, N=1 qualifies. Padding is excluded. The finite generation horizon is recorded.
- **Sentiment reward** is the positive-class probability of the configured oracle on prompt plus generated continuation. The local classifier differs from the paper's oracle.
- **Learned reward** and **token-overlap F1** on summarization/dialogue are diagnostics. Neither substitutes for GPT-4/human evaluation. Best-of-N selection scores are particularly susceptible to reward overoptimization.
- Bootstrap intervals in per-run JSON resample examples. Several sentiment pairs share one prompt, so these intervals can be optimistic. Across-seed standard deviations use three training runs on the same bounded corpus, not three independent datasets. In sentiment, each seed also constructs new preference samples from its own SFT model, so variation includes preference-generation randomness. The small PPO profile uses only 16 online rollouts; optimization budgets are not matched across algorithms.
- No test-set score selects training hyperparameters or reward checkpoints. The small profile selects the last epoch explicitly.

## Completing the remaining research

Full-scale reproduction requires separate compute and evaluation resources. Increase examples, context, optimization duration and generations; use the original models and oracle; restore original reward-model validation selection and convergence checkpoints; add the paper's independently tuned TRL/PPO and prompting controls. The training engine supports arbitrary causal Hugging Face models, but original model sizes need substantially more memory than this Mac provides. Original supervised summarization initialization must be obtained and verified separately.

Exported `judgments.jsonl` files contain randomized A/B comparisons and no scores. Fill `winner` (`A`, `B`, or `tie`) and `evaluator` with real provenance, then run:

```bash
python -m dpo_lab judge \
  --answers path/to/completed-judgments.jsonl \
  --key path/to/judgments.jsonl.key.jsonl \
  --output results/judgments-summary.json
```

Keep answer keys hidden from raters. Collect independent human judgments and retain their sample alignment before calculating agreement. Run `python -m dpo_lab agreement --left human.jsonl --right judge.jsonl --output agreement.json` on completed, aligned A/B forms. This computes observed agreement and Cohen’s kappa; it does not create judgments. The paper's `gpt-4-0314` results are historical; any replacement judge is a new evaluation condition and must be named as such.

## Sources and datasets

- [Paper and appendices](https://arxiv.org/html/2305.18290v3)
- [Original implementation](https://github.com/eric-mitchell/direct-preference-optimization)
- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb)
- [OpenAI summarization comparisons](https://huggingface.co/datasets/openai/summarize_from_feedback) — pinned Hub Parquet conversion; no remote dataset Python execution
- [Human-summary SFT data](https://huggingface.co/datasets/CarperAI/openai_summarize_tldr)
- [Anthropic HH](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail)
- [Local causal model](https://huggingface.co/EleutherAI/pythia-70m-deduped)
- [Local sentiment oracle](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)

Dataset text and model weights retain their upstream licenses and usage conditions. Exact source revisions and selected-data hashes are in each run's manifests.
