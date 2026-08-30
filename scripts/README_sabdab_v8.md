# SAbDab v8 正式入口

正式入口是 `scripts/evo_sabdab_v8.py`，可通过下列命令从项目根目录或任意目录启动：

```sh
/home/dataset-local/projects_dir/MERF/scripts/run_evo_sabdab_v8.sh --dataset_idx 0 --total_epochs 60
```

v8 专属参数会在入口层消费：`--actions-per-triple`（默认 3）、`--inner-passes`
（默认 2）以及 `--enable-early-stop`、`--early-stop-min-epochs`
（默认 40）、`--early-stop-patience`（默认 12）、`--early-stop-min-delta`
（默认 0.03）；其余参数与 `evo_sabdab_v3.py` 相同。

v8 的候选动作、action-gather GRPO、coverage-balanced update、历史最佳权重恢复和
真实 early stop 都直接定义在 `scripts/evo_sabdab_v8.py` 中。入口不再引用
`analysis/0803analysis_1/`，也不会动态加载或 monkey-patch v3；该目录仅保留阶段性
实验、日志和结果。

共享支持文件已经位于主项目，不重复复制：

- 模型：`model/MERF_v6.py`
- SAbDab RepairPDB 数据集：`dataset/dataset_evo_sabdab_v3.py`
- FoldX 结构构建：`protein/mutate_scripts_foldx_sabdab_v3.py`
- 原始数据与默认输入：`data/sabdab/sabdab_evo.csv`、`data/sabdab/PDBs/`
- 结构预测评测入口：`scripts/calculate_colabdesign_v3.py`

默认运行目录由 v8 入口统一指定，均可用同名参数覆盖：

- 日志：`logs/evo_sabdab_v8/`
- RepairPDB WT 缓存：`data/sabdab/PDBs_fixed_v8/`
- FoldX 突变体缓存：`data/sabdab/PDBs_evo_v8/`
- PLM embedding 缓存：`data/sabdab/PLM_embeddings_sabdab_v8.pkl`

首次运行会按现有 v3 数据管线生成缺失的 RepairPDB、FoldX 和 PLM 缓存。模型
checkpoint、FoldX、ColabDesign Python 环境等仍可通过原有命令行参数覆盖。

## v8 相对 v3 的实现落点

| v8 改进 | 在独立 v8 代码中的位置 | 保留的 v3 行为 |
| --- | --- | --- |
| 三位点完整非 WT 动作 | `generate_mutate_info_list`：屏蔽 native action；训练期逐位点独立采样 3 个完整动作，监测/最终期逐位点 greedy argmax | CDR 三位点组合、FoldX 建模和 reward 批处理评分 |
| 正确的 reward-to-action 归因 | `make_policy_samples`、`non_wt_log_probs`、`selected_joint_log_probs` | v3 的全候选池 advantage、ratio 上界 20 和预训练 reference KL 锚点 |
| coverage-balanced inner update | `balanced_inner_batches` 与 `train_one_epoch`：每个候选每个 pass 精确出现一次；默认 batch=32、passes=2 | Adam、学习率、权重衰减和 reward transform |
| deterministic greedy 监测 | `run_deterministic_greedy_monitor`：每次更新后按与最终提名相同的 non-WT greedy 规则重评分 top-8 | top-8 acquisition 指标及其候选排序 |
| 历史最佳权重恢复 | `capture_best_policy`、`restore_best_policy`：任何严格更优的监测值都保存 CPU 权重，不与 `min_delta` 混用 | 训练期 reward 模型、候选评分和最终 top-8 结构评测接口 |
| 有效早停 | `update_early_stop_state` 与 `run_one_dataset`：触发后先恢复最佳策略，完成当轮监测后直接 `break`，不再像旧 wrapper 一样冻结后空转 | 最终只从恢复后的单一策略重新提名 8 个候选，不使用历史候选池 |
| v8 可追溯性 | 训练 CSV/TensorBoard 写入采样数、pass 数、coverage、监测 acquisition、最佳 epoch、早停状态和每轮耗时；外部评测子目录使用 `_v8` 后缀 | Rosetta ddG 与 ColabDesign 的 SAbDab 适配调用方式 |

因此，`dataset_evo_sabdab_v3.py`、`MERF_v6.py`、FoldX 支持和
`calculate_colabdesign_v3.py` 的名字仍保留 v3/v6，是因为 v8 没有修改这些共享的
数据、模型或评测算法；v8 的算法差异已全部直接写入正式入口，而不是运行时替换它们。
