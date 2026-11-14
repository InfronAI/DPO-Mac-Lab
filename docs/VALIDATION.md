# Validation record

Validation is performed on an M1 Pro with 16 GB RAM. Evidence files live in `results/validation/`; the public experiment has its own environment and source snapshot.

## Scientific and implementation checks

- Original DPO equation and gradient direction.
- Identity policy/reference loss equals log(2).
- cDPO symmetry at 50% smoothing; valid beta/smoothing ranges.
- IPO formula and token-average likelihood; reference-free diagnostic consistency.
- Causal token shift, masked prompt/padding gradients and empty-target rejection.
- EOS preservation and actual prompt-length truncation budget.
- Small datasets, partial batches, finite sample limits and seeded iteration.
- Gradient accumulation matches an equivalent logical batch.
- Cached reference gradients match live frozen-reference gradients; stale cache is rejected.
- Dataset split leakage and empty-data guards.
- Completed-artifact hashes reject corrupted checkpoints.
- PPO terminal GAE/returns.
- Judge provenance, A/B alignment and agreement calculations.

Tests do not assert that every training run improves a metric; doing so would bias the experiment toward favorable results.

## PyTorch 2.6 failure and environment correction

The first public-data attempt used PyTorch 2.6. It stopped at a nonfinite MPS gradient in the sentiment DPO β=0.1 stage. The same dataset and SFT weights produced inconsistent MPS gradient norms in a separate diagnostic, while CPU gradients remained finite and much smaller. The older run was excluded from all reported public results; its local artifacts remain under `artifacts/failed-public-torch26`.

The environment was upgraded to PyTorch 2.15 and the public experiment restarted from the pretrained model. This is an observed environment correction, not a claim to have isolated a particular upstream kernel bug. `scripts/check_device_parity.py` compares CPU/MPS forward and backward computations on the actual pretrained Pythia model at three sequence lengths. The retained JSON records losses and the worst per-parameter relative gradient error. Acceptance tolerances are 0.1 absolute sequence-loss error and 2% worst relative gradient error; these are numerical tolerances, not bitwise equality.

The historical diagnostic logs are retained separately. Do not use their optimizer traces as experiment results.

## CPU/MPS numerical comparison: mixed results

The Pythia comparison passed the stated tolerances at lengths 48, 80 and 128 ([record](../results/validation/device-parity.json)). GPT-2 did **not** pass every strict per-parameter check on the same three-example computation:

| Context limit | Absolute sequence-loss difference | Worst parameter gradient relative L2 error | Whole-model gradient relative L2 error | Strict result |
|---|---:|---:|---:|---|
| 48 | 0.000832 | 1.706% | 0.0377% | Pass |
| 80 | 0.001488 | 2.001% | 0.0551% | Fail |
| 128 | 0.001587 | 2.121% | 0.0560% | Fail |

The worst parameter was `transformer.h.10.ln_2.weight`; the minimum whole-model gradient cosine similarity was 0.99999967. These measurements show a small aggregate discrepancy, but do not erase the failed 2% parameter threshold or guarantee identical training trajectories. The [full GPT-2 record](../results/validation/gpt2-device-parity.json) retains `passed: false`; selecting Metal matmul explicitly also failed that threshold ([separate record](../results/validation/gpt2-device-parity-metal.json)). No threshold was relaxed to obtain a pass. Use `--device cpu` for a separate CPU experiment when this distinction matters.

All 18 unit tests and both CPU/MPS end-to-end smoke workflows completed. Original Hydra SFT → DPO and IPO completed on both devices. Those are execution and mathematical checks, not a claim that every numerical comparison passed. CUDA remains untested.

## Measured reference-cache benefit

On the saved public-data GPT-2 SFT checkpoint, ten measured single-pair DPO forward/backward steps after two warmups averaged **0.1236 s with a live reference** and **0.0989 s with cached reference probabilities**: **1.25× step throughput**. Reported MPS allocated tensor memory fell from 1,520,854,272 to 1,009,709,312 bytes after moving the reference off MPS. See the [raw timing samples](../results/validation/reference-benchmark.json) and [benchmark script](../scripts/benchmark_reference.py).

This is one sequential microbenchmark with one pair, not an end-to-end training speedup or a peak system-memory measurement. Cache construction, optimizer updates and generation are excluded. The mathematical equivalence of live and cached reference gradients is tested separately.

## Hardware and evaluation limits

MPS and CPU are exercised locally. CUDA/FSDP/TensorParallel are preserved but cannot be validated on this hardware. No human study or GPT-4 evaluation was performed. The synthetic demo's apparent success measures fixture learnability only.

The `resources.csv` file samples process resident memory from shortly after run launch; it is not a full accounting of all macOS unified GPU allocations. The reference timing benchmark explicitly excludes cache construction and does not measure end-to-end task speedup.

## EOS diagnostic and additional GPT-2 run

A separate CPU check on the exact pretrained Pythia-70M-deduped revision found an EOS log-probability of about −1232.6 on one fixed sentence, compared with a mean ordinary-token log-probability of about −2.93. Changing the last input token changed no preceding logits (maximum difference 0), so this case does not indicate a causal-mask leak. The complete record is `results/validation/pythia-eos.json`.

This exceptionally large EOS term can dominate short SFT sequences. The original EOS-inclusive objective is retained; removing EOS to make the numbers look better would change the training target. A second, explicitly separate `experiments/gpt2-mac.json` profile uses GPT-2 with the paper's 1e-6 learning rate and 150-step warmup. It is a model-and-configuration diagnostic, not a controlled single-factor causal comparison and not a replacement for the original failed or weak results. The companion fixed-sentence measurement is `results/validation/gpt2-eos.json`.
