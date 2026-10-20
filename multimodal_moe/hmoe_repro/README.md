# hmoe_repro：用 Soft MoE 的划分 / seed 复现原版 HierarchicalMoE

本目录**不改**原版 `src/` 训练算法，只做两件事：

1. 用 `multimodal_moe` 的协议生成 / 复用 `train_30` / `test_10`（`--seed 64..68`）
2. 把该名单注入原版 HMoE 的 `MACHSUITE_KERNEL` / `poly_KERNEL` / `random_seed`，再跑预训练或微调

产物元数据：`multimodal_moe/hmoe_repro_outputs/seed_N/`  
模型日志仍在：`HierarchicalMoE-master/src/logs/`（原版 saver）

---

## 依赖

- 已按仓库根 `requirements.txt` 装好原版 HMoE 环境
- 有 `dse_database` / programl 图数据（原版 HMoE 数据路径）
  - 当前仓库常见缺库：`machsuite/.../spmv-crs_*.db`（缺则该核被跳过）
  - `aes` 若 `final valid: 0` 表示该库几乎无有效点，不影响其它核继续 regen
- 协议划分用的 CDFG 计数来自 `two_tower_datasets*/graph`（与 Soft MoE 一致）
  - 若 `graph/` 为空，会与 Soft MoE 相同地 fallback 丢掉 `md` / `seidel-2d`，划分仍由 `--seed` 决定
  - 已有 `ablation_outputs[/seed_N]/kernel_split.json` 且 seed 一致时会直接复用

在仓库根执行（入口会自动 `chdir` 到 `src/`，以匹配原版 `../dse_database` 相对路径）：

```bash
cd HierarchicalMoE-master
```

---

## 一键多 seed（65–68）

在 `HierarchicalMoE-master` 下：

```bash
chmod +x multimodal_moe/hmoe_repro/run_seeds.sh
# 默认跑 65 66 67 68；每个 seed：split → pretrain(1000) → finetune(500)
nohup bash multimodal_moe/hmoe_repro/run_seeds.sh > multimodal_moe/hmoe_repro_outputs/pipeline_65_68.log 2>&1 &

# 或指定 seed：
# bash multimodal_moe/hmoe_repro/run_seeds.sh 65 66 67 68
```

汇总表：`multimodal_moe/hmoe_repro_outputs/pipeline_summary.tsv`  
各 seed 日志：`hmoe_repro_outputs/seed_N/{split_only,pretrain,finetune}.log`  
**最终结果看每行 `finetune` 的 `best_val` / `final_test` / `best_test`（未见核 k-shot 设定下）。**

**磁盘清理（默认开启）**：每个 stage 训完后删 `save/.../data_*.pt`；预训练会把 `encoders.klepto` 备份到 `hmoe_repro_outputs/seed_N/`；微调结束后再删掉该 seed 的 pretrain+finetune 整套 cache。权重只留在 `src/logs/**/*.pth`。若要保留 cache：加 `--no_cleanup_save`。

---

## 推荐流程（每个 seed 一遍）

### 1) 划分（与 Soft MoE 对齐）

若已跑过 ablation 同 seed 的 `split_only`，会自动复用  
`ablation_outputs[_not_0]/seed_N/kernel_split.json`。

```bash
python -m multimodal_moe.hmoe_repro.run_hmoe --stage split_only --seed 67
# not_0 数据：
# python -m multimodal_moe.hmoe_repro.run_hmoe --stage split_only --seed 67 --dataset_mode not_0
```

### 2) 预训练（核 = train_30）

首次务必 `--force_regen`（核名单随 seed 变，不能沿用旧 cache）：

```bash
python -m multimodal_moe.hmoe_repro.run_hmoe \
  --stage pretrain --seed 67 --force_regen
```

结束后在 `src/logs/` 里找路径含 `mm-proto-seed_67-pt` 的  
`*val_model_state_dict.pth`，记下来。

### 3) 微调（核 = test_10，k=50，kmeans）

```bash
python -m multimodal_moe.hmoe_repro.run_hmoe \
  --stage finetune --seed 67 --force_regen \
  --model_path /绝对路径/xxx_val_model_state_dict.pth
```

入口会自动加载预训练目录下的 `encoders.klepto`（与模型输入维一致）；**不要**在微调阶段重新 fit encoder。  
看 `*train_model_state_dict.pth`（tag 含 `mm-proto-seed_67-ft`）。

### 4) 与 Soft MoE 对照

同 seed 跑你的门控：

```bash
python -m multimodal_moe.moe_system.moe_cv10_main --subtask train --seed 67
python -m multimodal_moe.moe_system.moe_cv10_main --subtask finetune --seed 67
```

只在**同一 `test_10` + 同一 seed** 上比指标。

---

## 常用参数

| 参数 | 含义 |
|------|------|
| `--seed` | 64..68，决定 `train_30/test_10` 与 `random_seed` |
| `--dataset_mode` | `filtered` / `not_0`（协议 CDFG 路径与默认 split 复用） |
| `--split_json` | 显式指定 `kernel_split.json` |
| `--rebuild_split` | 强制重写划分 |
| `--force_regen` | 按当前核名单重建 HMoE programl 数据集 |
| `--moe_layers` | 默认 `hierarchy-weighted-hidden` |
| `--transfer_k_shot` | 默认 50 |
| `--coreset` | `kmeans` / `random` |

---

## 目录结构

```
multimodal_moe/hmoe_repro/
  __init__.py
  protocol_io.py      # 读/写协议划分
  run_hmoe.py         # 入口
  README.md

multimodal_moe/hmoe_repro_outputs/seed_N/
  kernel_split.json   # 若未复用 ablation 的
  run_meta.json
  pretrain_last_run.json / finetune_last_run.json
```
