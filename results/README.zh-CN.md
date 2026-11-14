# 实验数据与分析概览

本机完成 **122 个公开数据评估变体**，另有 CPU 与 MPS 各 32 个合成数据评估变体。“评估变体”包括未训练的基线与 Best-of-N，不等于独立训练次数。

| 实验 | 模型 | 随机种子数 | 评估变体数 | 详细报告 |
|---|---|---:|---:|---|
| m1-public | EleutherAI/pythia-70m-deduped | 3 | 105 | [Report](m1-public/REPORT.md) |
| gpt2-public | openai-community/gpt2 | 1 | 17 | [Report](gpt2-public/REPORT.md) |

## 默认 DPO 设置的实际测量

这些值是独立偏好对上的隐式奖励准确率，并非 GPT-4 或人类胜率。SFT 参考模型的隐式奖励相等，按平局规则固定计 0.5。

| 模型 | 任务 | DPO 准确率 | 跨种子标准差 | Preferred-FT 准确率 |
|---|---|---:|---:|---:|
| m1-public | sentiment | 0.514 | 0.049 | 0.500 |
| m1-public | summarization | 0.458 | 0.191 | 0.458 |
| m1-public | dialogue | 0.438 | 0.165 | 0.521 |
| gpt2-public | sentiment | 0.500 | N/A (one seed) | 0.453 |
| gpt2-public | summarization | 0.438 | N/A (one seed) | 0.438 |
| gpt2-public | dialogue | 0.500 | N/A (one seed) | 0.562 |

## 数据支持的结论

Pythia 的完整矩阵没有证明小规模设置下 DPO 能稳定优于对照，保存的生成中存在重复文本。数值调查发现该检查点的 EOS 概率极低；为保留原始目标，没有删除 EOS。额外的 GPT-2 配置同时改变了模型和优化设置，因此仅作为实用诊断，不能用于单因素因果归因。

训练来源样本仅 64 条，上下文很短；PPO 在线样本预算、DPO 轮数与 Best-of-N 推理成本也不相同。生成评估样本很少，情感任务的随机种子还会改变偏好采样。这些结果证明流程可以执行，并揭示其局限，不能验证论文中的算法排名。

## 产物与验证

- 每次运行保存源代码、模型、数据版本，原始选定样本，逐步训练指标，逐条概率评估，生成文本，盲评表与 SHA-256 完成凭据。
- 大型检查点和奖励／价值模型权重保留在本地，Git 不跟踪；新克隆请使用新的实验目录重新运行。
- [CPU 完整测试](smoke-cpu/REPORT.md)、[MPS 完整测试](smoke-mps/REPORT.md)、[单元测试输出](validation/pytest.txt)。
- [数值验证与计时](../docs/VALIDATION.md)：Pythia 通过 CPU/MPS 阈值；GPT-2 有两组超过单参数梯度误差 2% 的严格阈值（最大 2.121%，整体误差 0.056%），失败记录保留。参考缓存微基准的前向／反向吞吐量为原来的 1.25 倍，不含缓存构造和优化器更新。
- 原 Hydra 入口的 [CPU 验证](validation/legacy-cpu.json)与 [MPS 验证](validation/legacy-mps.json)。
- [论文覆盖与未完成部分](../docs/PAPER_REPRODUCTION.md)：原始模型规模、完整调参对照、历史 GPT-4 及人工评估仍未完成，不能称为 100% 论文复现。
