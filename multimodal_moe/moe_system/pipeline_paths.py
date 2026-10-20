"""
MoE 流水线产物根目录（新建，不写入 save_models_and_data）。

布局:
  moe_pipeline_outputs/
    hierarchical_moe/      # 典型 Hierarchical MoE 权重与推理结果
    fold_experts/          # （可选）旧 label_gen 折专家
    labels/                # （可选）旧算法选择标签
    selector/              # （可选）旧选择器
    logs/
"""
from __future__ import annotations

import os
from os.path import abspath, dirname, join

_MOE_DIR = dirname(abspath(__file__))
_ROOT = dirname(_MOE_DIR)

PIPELINE_ROOT = join(_ROOT, "moe_pipeline_outputs")

MOE_DIR = join(PIPELINE_ROOT, "hierarchical_moe")
FOLD_EXPERT_DIR = join(PIPELINE_ROOT, "fold_experts")
LABEL_DIR = join(PIPELINE_ROOT, "labels")
SELECTOR_DIR = join(PIPELINE_ROOT, "selector")
LOG_DIR = join(PIPELINE_ROOT, "logs")

STATIC_EXPERT_DIR = join(_ROOT, "model_select_weight_del_0_and_random")

SCHEME_NAMES = [
    "all",
    "ast_cdfg",
    "ast_text",
    "cdfg_text",
    "cdfg_only",
    "ast_only",
    "LM",
]

SCHEME_DIM = {
    "all": 153,
    "ast_cdfg": 153,
    "ast_text": 153,
    "cdfg_text": 153,
    "cdfg_only": 153,
    "ast_only": 784,
    "LM": 153,
}

TARGET_NAMES = ["perf", "util-LUT", "util-FF", "util-DSP", "util-BRAM"]

STATIC_FILES = {
    "all": "all_regression_model_state_dict.pth",
    "ast_cdfg": "ast_and_cdfg_regression_model_state_dict.pth",
    "ast_text": "ast_and_text_regression_model_state_dict.pth",
    "cdfg_text": "cdfg_and_text_regression_model_state_dict.pth",
    "cdfg_only": "cdfg_regression_model_state_dict.pth",
    "ast_only": "ast_regression_model_state_dict.pth",
    "LM": "text_regression_model_state_dict.pth",
}

N_FOLDS = 5
SELECTOR_TRAIN_RATIO = 0.7


def ensure_pipeline_dirs():
    for d in (PIPELINE_ROOT, MOE_DIR, FOLD_EXPERT_DIR, LABEL_DIR, SELECTOR_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)


def fold_expert_path(mode: str, fold_idx: int) -> str:
    return join(FOLD_EXPERT_DIR, f"expert_{mode}_fold_{fold_idx}.pth")


def labels_json_path() -> str:
    return join(LABEL_DIR, "oof_scheme_rmse_labels.json")


def labels_csv_path() -> str:
    return join(LABEL_DIR, "oof_scheme_rmse_labels.csv")


def selector_best_path() -> str:
    return join(SELECTOR_DIR, "moe_selector_best.pth")


def selector_final_path() -> str:
    return join(SELECTOR_DIR, "moe_selector_final.pth")


def moe_best_path() -> str:
    return join(MOE_DIR, "moe_hierarchical_best.pth")


def moe_final_path() -> str:
    return join(MOE_DIR, "moe_hierarchical_final.pth")
