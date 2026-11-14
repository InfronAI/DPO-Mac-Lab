"""Build a bilingual evidence index from completed, generated reports."""
import json
from pathlib import Path
roots=[Path('results/m1-public'),Path('results/gpt2-public')]
meta=[json.loads((r/'manifest.json').read_text()) for r in roots]
if any(m['status']!='completed' for m in meta):raise ValueError('Both public matrices must finish first')
aggregates=[json.loads((r/'aggregate.json').read_text()) for r in roots]
count=sum(m['completed_runs'] for m in meta)
en=['# Saved experiment evidence','',f'**{count} public-data evaluation variants** completed on this M1 Mac, plus separate 32-variant CPU and MPS synthetic smoke workflows. “Variant” includes untrained baselines and Best-of-N evaluations; it is not a count of independent training jobs.','',
    '| Run | Model | Seeds | Evaluated variants | Report |','|---|---|---:|---:|---|']
zh=['# 实验数据与分析概览','',f'本机完成 **{count} 个公开数据评估变体**，另有 CPU 与 MPS 各 32 个合成数据评估变体。“评估变体”包括未训练的基线与 Best-of-N，不等于独立训练次数。','',
    '| 实验 | 模型 | 随机种子数 | 评估变体数 | 详细报告 |','|---|---|---:|---:|---|']
for root,m in zip(roots,meta):
 row=f"| {root.name} | {m['config']['model']} | {len(m['config']['seeds'])} | {m['completed_runs']} | [Report]({root.name}/REPORT.md) |"
 en.append(row);zh.append(row)
en+=['','## Default DPO measurements','','These are held-out implicit reward accuracies, not GPT-4 or human win rates. The SFT reference receives 0.5 for tied rewards by definition.','',
     '| Model | Task | DPO accuracy | Seed SD | Preferred-FT accuracy |','|---|---|---:|---:|---:|']
zh+=['','## 默认 DPO 设置的实际测量','','这些值是独立偏好对上的隐式奖励准确率，并非 GPT-4 或人类胜率。SFT 参考模型的隐式奖励相等，按平局规则固定计 0.5。','',
     '| 模型 | 任务 | DPO 准确率 | 跨种子标准差 | Preferred-FT 准确率 |','|---|---|---:|---:|---:|']
for root,aggregate in zip(roots,aggregates):
 for task,beta in [('sentiment','0.1'),('summarization','0.5'),('dialogue','0.1')]:
  d=next(r for r in aggregate if r['task']==task and r['method']==f'dpo_beta_{beta}')
  f=next(r for r in aggregate if r['task']==task and r['method']=='preferred_ft')
  sd=f"{d['accuracy_sd']:.3f}" if d['accuracy_sd'] is not None else 'N/A (one seed)'
  line=f"| {root.name} | {task} | {d['accuracy_mean']:.3f} | {sd} | {f['accuracy_mean']:.3f} |"
  en.append(line);zh.append(line)
en+=['','## What the data supports','','The Pythia matrix does not establish a consistent DPO improvement in this small setting. Saved generations include repetitive text. A numerical investigation identified an unusually low EOS probability in that checkpoint; EOS remains in the objective. The separate GPT-2 profile changes both the model and optimizer settings, so its results are a practical diagnostic and cannot isolate the cause of any difference.',
     '', 'Only 64 training source examples and a short context are used. PPO rollout budgets, DPO epochs, and Best-of-N inference costs differ. The generation sample is very small, and the sentiment seeds also change preference sampling. These runs demonstrate execution and expose limitations; they cannot verify the paper’s algorithm rankings.',
     '', '## Files and verification','',
     '- Each run contains source/model/data revisions, raw selected examples, per-step training JSONL, per-example likelihood measurements, generation samples, blinded judgment forms and SHA-256 completion receipts.',
     '- Large model checkpoints and reward/value weights remain locally available, but are excluded from Git. Fresh clones should run a new experiment directory.',
     '- [CPU smoke](smoke-cpu/REPORT.md), [MPS smoke](smoke-mps/REPORT.md), [test output](validation/pytest.txt).',
     '- [Numerical validation and timing](../docs/VALIDATION.md): Pythia passed the CPU/MPS thresholds; GPT-2 failed the strict 2% per-parameter gradient threshold in two cases (maximum 2.121%, whole-model error 0.056%). These failures are retained. The reference-cache microbenchmark measured 1.25× forward/backward throughput, excluding cache construction and optimizer updates.',
     '- [Original Hydra validation on CPU](validation/legacy-cpu.json) and [MPS](validation/legacy-mps.json).',
     '- [Paper coverage and unexecuted research](../docs/PAPER_REPRODUCTION.md). Full-scale models, original tuned controls, historical GPT-4 and human evaluation remain uncompleted.']
zh+=['','## 数据支持的结论','','Pythia 的完整矩阵没有证明小规模设置下 DPO 能稳定优于对照，保存的生成中存在重复文本。数值调查发现该检查点的 EOS 概率极低；为保留原始目标，没有删除 EOS。额外的 GPT-2 配置同时改变了模型和优化设置，因此仅作为实用诊断，不能用于单因素因果归因。',
     '', '训练来源样本仅 64 条，上下文很短；PPO 在线样本预算、DPO 轮数与 Best-of-N 推理成本也不相同。生成评估样本很少，情感任务的随机种子还会改变偏好采样。这些结果证明流程可以执行，并揭示其局限，不能验证论文中的算法排名。',
     '', '## 产物与验证','',
     '- 每次运行保存源代码、模型、数据版本，原始选定样本，逐步训练指标，逐条概率评估，生成文本，盲评表与 SHA-256 完成凭据。',
     '- 大型检查点和奖励／价值模型权重保留在本地，Git 不跟踪；新克隆请使用新的实验目录重新运行。',
     '- [CPU 完整测试](smoke-cpu/REPORT.md)、[MPS 完整测试](smoke-mps/REPORT.md)、[单元测试输出](validation/pytest.txt)。',
     '- [数值验证与计时](../docs/VALIDATION.md)：Pythia 通过 CPU/MPS 阈值；GPT-2 有两组超过单参数梯度误差 2% 的严格阈值（最大 2.121%，整体误差 0.056%），失败记录保留。参考缓存微基准的前向／反向吞吐量为原来的 1.25 倍，不含缓存构造和优化器更新。',
     '- 原 Hydra 入口的 [CPU 验证](validation/legacy-cpu.json)与 [MPS 验证](validation/legacy-mps.json)。',
     '- [论文覆盖与未完成部分](../docs/PAPER_REPRODUCTION.md)：原始模型规模、完整调参对照、历史 GPT-4 及人工评估仍未完成，不能称为 100% 论文复现。']
Path('results/README.md').write_text('\n'.join(en)+'\n');Path('results/README.zh-CN.md').write_text('\n'.join(zh)+'\n')
