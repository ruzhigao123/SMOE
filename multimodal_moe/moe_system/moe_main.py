"""
典型 Hierarchical MoE 入口。

  train / train_moe   15 kernel 上端到端回归（kernel 7:3），存 best by val RMSE
  inference           7 未见 kernel 上 MoE 直接预测（不加载 model_select_weight）
  all                 train → inference
  label_gen           （可选）旧算法选择标签流水线，与典型 MoE 无关

运行:
  python moe_system/moe_main.py --subtask train
  python moe_system/moe_main.py --subtask inference
  python moe_system/moe_main.py --subtask all
"""
from __future__ import annotations

import os
import sys

import torch
from torch_geometric.data import Dataset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)  # multimodal_moe
_HMOE_ROOT = os.path.dirname(_ROOT)
_HMOE_SRC = os.path.join(_HMOE_ROOT, "src")
for _p in (_HMOE_SRC, _HMOE_ROOT, _ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.config import FLAGS
from utils_new import build_aligned_list, default_data_paths
from pipeline_paths import PIPELINE_ROOT, MOE_DIR, ensure_pipeline_dirs
from moe_train import train_moe, inference_heldout


class SimpleDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def load_aligned_cdfg_dataset():
    paths = default_data_paths()
    for name, root in (
        ("CDFG", paths["cdfg_root"]),
        ("Text", paths["text_root"]),
        ("AST", paths["ast_pt_root"]),
    ):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"{name} 根目录不存在: {root}")

    aligned = build_aligned_list(
        paths["cdfg_root"], paths["text_root"], paths["ast_pt_root"]
    )
    print(
        f"[MoE] 三模态对齐交集: {len(aligned)} 条 "
        f"(CDFG=data_i, Text=i+1, AST=i.ast.pt)"
    )
    if len(aligned) == 0:
        raise FileNotFoundError(
            "三模态对齐交集为空。请检查:\n"
            f"  CDFG={paths['cdfg_root']}\n"
            f"  Text={paths['text_root']}\n"
            f"  AST ={paths['ast_pt_root']}"
        )

    data_list = []
    for info in aligned:
        suite, kernel, idx = info["key"]
        data = torch.load(info["cdfg"], map_location="cpu")
        if not hasattr(data, "x"):
            raise RuntimeError(f"CDFG 不是图对象: {info['cdfg']}")
        data.file = os.path.basename(info["cdfg"])
        data.kernel = kernel
        data.benchmark = suite
        data.moe_cdfg_path = info["cdfg"]
        data.moe_text_path = info["text"]
        data.moe_ast_path = info["ast"]
        data.moe_graph_index = int(idx)
        data_list.append(data)

    print(f"[MoE] 已加载 CDFG 主样本 {len(data_list)} 条")
    return data_list


if __name__ == "__main__":
    ensure_pipeline_dirs()
    print(f"[MoE] 产物根目录: {PIPELINE_ROOT}")
    print(f"[MoE] Hierarchical MoE 目录: {MOE_DIR}")
    print("[MoE] 典型 Hierarchical MoE：可学习三专家 + soft route，不加载静态 .pth")

    samples = load_aligned_cdfg_dataset()
    dataset = SimpleDataset(samples)
    subtask = 'train'
    # subtask = str(getattr(FLAGS, "subtask", "train")).strip().lower()
    # if not any(a == "--subtask" or a.startswith("--subtask=") for a in sys.argv):
    #     # 未显式传参时默认 train（典型 MoE）
    #     if subtask in ("label_gen", "train_selector", "all"):
    #         pass
    #     else:
    #         subtask = "train"

    alias = {
        "train": "train",
        "train_moe": "train",
        "train_selector": "train",
        "selector": "train",
        "infer": "inference",
        "export_oof": "label_gen",
        "oof": "label_gen",
        "label": "label_gen",
        "full15": "train",
    }
    subtask = alias.get(subtask, subtask)
    print(f"[MoE] subtask={subtask}")

    if subtask == "label_gen":
        from label_gen import run_label_gen

        run_label_gen(dataset)
    elif subtask == "train":
        train_moe(dataset)
    elif subtask == "inference":
        inference_heldout(dataset)
    elif subtask == "all":
        train_moe(dataset)
        inference_heldout(dataset)
    else:
        raise ValueError(
            f"未知 subtask={subtask!r}，可选: train | inference | all | label_gen"
        )
