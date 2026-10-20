# multimodal_moe/src — 消融（默认恢复最初 only-* 方案）

## 最初方案（当前默认）

| 项 | 设置 |
|----|------|
| 划分 | kernel **24 / 6 / 10**（`--split_by kernel`） |
| 模型 | `cdfg_only` / `ast_only` / `LM`：单模态 → D → MLP（**无融合**） |
| 早停 | 关闭（`--patience 0`） |
| 默认 mode | `cdfg_only` |

```bash
cd HierarchicalMoE-master

python -m multimodal_moe.src.main_ablation --subtask split_only

python -m multimodal_moe.src.main_ablation --subtask train \
  --ablation_mode cdfg_only --split_by kernel --no_resume

python -m multimodal_moe.src.main_ablation --subtask train \
  --ablation_mode ast_only --split_by kernel --no_resume

python -m multimodal_moe.src.main_ablation --subtask train \
  --ablation_mode LM --split_by kernel --no_resume
```

产物（随 `--dataset_mode` 隔离）：
- `filtered` → `ablation_outputs/{mode}/best.pt`
- `not_0` → `ablation_outputs_not_0/{mode}/best.pt`

可选：`--split_by config`（30 核内配置 8:2）；多模态 `all/cdfg_text/...` 仍可用融合栈。

## 对比实验 SOTA（`comp_model`）

已迁移 `src/comp_model/`（IronMan / HGP / PNA / GNN-DSE / GCN / GAT / GIN / GraphSAGE）。

| 开关 | 含义 |
|------|------|
| `--comparative_if false`（默认） | 正常：`gnn_type=gcn\|transformer` |
| `--comparative_if true` | 对比：CDFG 骨干换成 `--comparative_model` |
| `--comparative_model ironman` | IronMan-Pro GCN 骨干 |

```bash
# 1) 源域训练（产物 …/seed_N/comp_ironman/cdfg_only/best.pt）
python -m multimodal_moe.src.main_ablation --subtask train \
  --ablation_mode cdfg_only --seed 64 \
  --comparative_if true --comparative_model ironman --no_resume

# 2) 小样本适应（test_10，每核 k=50；与 HMoE/Soft MoE 协议对齐）
python -m multimodal_moe.src.main_ablation --subtask finetune \
  --ablation_mode cdfg_only --seed 64 \
  --comparative_if true --comparative_model ironman \
  --transfer_k_shot 50 --coreset kmeans

# gnn-dse 同理：把 comparative_model 换成 gnn-dse
```

微调产物：

```text
…/seed_N/comp_ironman/cdfg_only/finetune_k50_kmeans/
  train_model.pt          # 按 train loss 最优（对齐 HMoE）
  best_query.pt           # query mse_sum5 最优
  finetune_result.json    # BEFORE/AFTER mse·rmse
```

**零样本（你已跑过）**：只看源域 `best.pt` 在 test 上的 `inference` / train 日志即可；  
**加小样本**：必须先有源域 `best.pt`，再 `--subtask finetune`。
