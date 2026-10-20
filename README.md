# Soft MoE for HLS QoR Prediction

**Soft Mixture of Experts (Soft MoE)** predicts HLS quality-of-results (QoR) from three complementary program views—**CDFG**, **AST**, and **source-code Text**—and mixes frozen modality experts with a **sample- and metric-adaptive soft gate**.

This directory (`multimodal_moe/`) is self-contained relative to the upstream HierarchicalMoE repo: data generation, single-modality experts, CV10 folds, Soft MoE training/finetune, and paper assets live here without modifying HierarchicalMoE’s original `src/`.

Paper draft: [`soft_moe_hls_qor.tex`](soft_moe_hls_qor.tex)

---

## Method in one page

| Component | Role |
|-----------|------|
| **CDFG expert** | GNN on the control/data-flow graph → \(\mathbf{h}_{\mathrm{CDFG}}\), \(\hat{\mathbf{y}}_{\mathrm{CDFG}}\) |
| **AST expert** | tree-sitter AST → CodeBERT node features (+ HW feats) → mean-pool 784-d → MLP → \(\mathbf{h}_{\mathrm{AST}}\), \(\hat{\mathbf{y}}_{\mathrm{AST}}\) |
| **Text expert** | CodeBERT on pragma-augmented source → 768-d → MLP → \(\mathbf{h}_{\mathrm{Text}}\), \(\hat{\mathbf{y}}_{\mathrm{Text}}\) |
| **Soft gate** | Continuous weights \(w_{t,m}\) per QoR metric \(t\) and expert \(m\); fuse **predictions**, not early features |
| **Few-shot** | Freeze all three experts; update **gate only** on \(k\) labeled configs per unseen kernel |

**Mix ablations** (`--mix_mode`):

| Mode | Behavior |
|------|----------|
| `soft` | Learnable soft weights (main Soft MoE) |
| `uniform` | Fixed \(1/3\) each expert (no gate training) |
| `hard_select` | Per-metric argmax; only one expert used (STE in training) |

---

## Repository layout

```text
multimodal_moe/
├── dse_gen/                    # HLSyn → aligned Graph / Text / .c
├── two_tower_datasets*/        # CDFG + Text features
├── two_tower_datasets_c*/code/ # AST: tree-sitter → CodeBERT → .ast.pt
├── src/                        # Single-modal / fusion ablations + SOTA backbones
│   ├── main_ablation.py
│   ├── cv10_single_modal.py
│   ├── model_ablation.py
│   └── comp_model/             # IronMan, GNN-DSE, CoGNN/sotampm, …
├── moe_system/                 # Soft MoE (frozen experts + gate)
│   ├── moe_cv10_main.py        # CLI entry
│   ├── moe_cv10_model.py
│   ├── moe_cv10_train.py
│   └── cv10_expert_bank.py
├── hmoe_repro/                 # HierarchicalMoE reproduction helpers
├── soft_moe_hls_qor.tex        # Paper
└── README.md                   # This file
```

More detail:

- Experts / ablation: [`src/ABLATION_README.md`](src/ABLATION_README.md)
- 10-fold experts: [`CV10_README.md`](CV10_README.md)
- MoE gate training: [`moe_system/MOE_CV10_README.md`](moe_system/MOE_CV10_README.md)
- Selector gate variant: [`moe_system/SELECTOR_MOE.md`](moe_system/SELECTOR_MOE.md)

---

## Environment

```bash
cd HierarchicalMoE-master
# Use the same env as HierarchicalMoE (see repo requirements.txt)
# Typical: PyTorch, torch-geometric, transformers (CodeBERT), tree-sitter-languages
```

CodeBERT weights: set `CODEBERT_DIR`, or place under `HierarchicalMoE-master/codebert/`  
(needs `config.json` + `pytorch_model.bin` / `model.safetensors`).

HLSyn inputs (configs, DBs, programl graphs) are under `new_kernel/dse_database/` (or your local equivalent).

---

## End-to-end pipeline

Run everything from **`HierarchicalMoE-master`** unless noted.

### 1) Build multimodal features

```bash
cd multimodal_moe/dse_gen
python count_hlsyn_configs.py
python gen_dataset_c.py          # .c + CDFG → two_tower_datasets_c*
python gen_dataset.py            # Text (CodeBERT) → two_tower_datasets*

cd ../two_tower_datasets_c_not_0/code   # or two_tower_datasets_c/code
python convert_hls_to_ast.py     # tree-sitter → .ast.json
python ast_dataset_upgrade.py    # CodeBERT node feats → .ast.pt
python merge_ast_dataset.py      # attach QoR labels
```

`dataset_mode`:

| Mode | Meaning | Typical roots |
|------|---------|---------------|
| `filtered` | Filtered configs | `two_tower_datasets/`, `ablation_outputs/`, `moe_cv10_outputs/` |
| `not_0` | Non-zero / alternate filter | `*_not_0/` counterparts |

### 2) Train single-modality experts (ablation)

Kernel split: 42 → drop 2 → **train_30 / test_10**; train_30 → **24 / 6** val.

```bash
cd HierarchicalMoE-master

python -m multimodal_moe.src.main_ablation --subtask split_only

python -m multimodal_moe.src.main_ablation --subtask train \
  --ablation_mode cdfg_only --seed 64 --split_by kernel --no_resume

python -m multimodal_moe.src.main_ablation --subtask train \
  --ablation_mode ast_only --seed 64 --split_by kernel --no_resume

python -m multimodal_moe.src.main_ablation --subtask train \
  --ablation_mode LM --seed 64 --split_by kernel --no_resume
```

Optional fixed fusion / SOTA CDFG backbones: see [`src/ABLATION_README.md`](src/ABLATION_README.md).

Outputs: `ablation_outputs/seed_{seed}/{mode}/best.pt`

### 3) (Recommended) CV10 fold experts for MoE

Avoid using ablation weights that already saw all of train_30 when routing on those kernels:

```bash
python -m multimodal_moe.src.cv10_single_modal --cv_subtask run_all
```

Outputs: `cv10_outputs/seed_{seed}/fold_{i}/{mode}/best.pt`

### 4) Soft MoE — train the gate (experts frozen)

```bash
# Main Soft MoE
python -m multimodal_moe.moe_system.moe_cv10_main --subtask train \
  --mix_mode soft --seed 64

# Mix ablations (separate save dirs under mix_*)
python -m multimodal_moe.moe_system.moe_cv10_main --subtask train \
  --mix_mode uniform --seed 64
python -m multimodal_moe.moe_system.moe_cv10_main --subtask train \
  --mix_mode hard_select --seed 64
```

Useful flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--mix_mode` | (see CLI) | `soft` / `uniform` / `hard_select` |
| `--gate_strategy` | `legacy` | `legacy` or `selector` |
| `--expert_source` | `cv10` | `cv10` or `ablation` |
| `--patience` | `0` | Early stop; `0` = run all epochs |
| `--no_resume` | off | Ignore existing `best.pth` |

Default save root (example, soft):

```text
moe_cv10_outputs/seed_64/          # soft, legacy
moe_cv10_outputs/seed_64/selector/ # soft + selector
moe_cv10_outputs/seed_64/mix_hard_select/
moe_cv10_outputs/seed_64/selector/mix_uniform/
```

### 5) Few-shot finetune (gate only)

```bash
python -m multimodal_moe.moe_system.moe_cv10_main --subtask finetune \
  --mix_mode soft --seed 64 \
  --transfer_k_shot 50 --coreset kmeans
```

Keep the **same** `--mix_mode` / `--gate_strategy` / `--seed` as source train.  
Loads `{save_dir}/best.pth`; writes `{save_dir}/finetune_k50_kmeans/train_model.pth`.

### 6) Inference

```bash
# Source-domain gate (best.pth under save_dir)
python -m multimodal_moe.moe_system.moe_cv10_main --subtask inference \
  --mix_mode soft --seed 64

# After finetune — must point to train_model.pth
python -m multimodal_moe.moe_system.moe_cv10_main --subtask inference \
  --mix_mode soft --seed 64 \
  --model_path moe_cv10_outputs/seed_64/finetune_k50_kmeans/train_model.pth
```

Finetune logs already report **query** MSE; that is the few-shot result. Default inference without `--model_path` is **pre-finetune**.

---

## Metrics and fairness

Primary metric: **MSE** per target and **`mse_sum5`** (sum over perf + util-LUT/FF/DSP/BRAM).

At Soft MoE train start, same-split baselines are printed:

- `cdfg_only` / `ast_only` / `LM`
- `uniform` (fixed 1/3)
- `oracle` (per-sample best expert; upper bound)

**Only compare MoE to these baselines on the same val/test.**  
Do not compare CV10 holdout MSE directly to MoE test MSE.

`head2head_report.txt` / `moe_effective=YES` means MoE `sum5` beats the strongest single expert on that split.

---

## Checkpoint cheat sheet

| File | Meaning |
|------|---------|
| `best.pth` | Gate selected on val (default `beat_first`) |
| `last.pth` | Last training epoch |
| `best_perf.pth`, `best_util_*.pth` | Per-target snapshots |
| `finetune_k*/train_model.pth` | Few-shot gate (lowest train loss) |
| `result.json` | Metrics + paths |

---

## Modality feature summary

```text
Source (.c + pragmas)
├── CDFG  → ProGraML / HLSyn graph → GNN expert
├── Text  → CodeBERT → 768-d → MLP expert
└── AST   → tree-sitter → CodeBERT(+HW) nodes → mean-pool 784-d → MLP expert
```

Soft MoE freezes the three experts and learns continuous mixture weights per QoR metric.

---

## Citation / related work

- Soft MoE (this project): see `soft_moe_hls_qor.tex`
- Hierarchical MoE: Li et al., AAAI 2025 (upstream `HierarchicalMoE-master/`)
- Related predictors: GNN-DSE, IronMan-Pro, HGBO, ProgSG, MPM
