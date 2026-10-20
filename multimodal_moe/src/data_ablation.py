"""
三模态对齐样本加载（不做融合）。

路径约定与 moe_system/utils_new 一致，由 dataset_mode 切换：
  filtered: two_tower_datasets + two_tower_datasets_c
  not_0:    two_tower_datasets_not_0 + two_tower_datasets_c_not_0
"""
from __future__ import annotations

import os
import sys
from os.path import dirname, abspath, join
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

_SRC = dirname(abspath(__file__))
_MM = dirname(_SRC)
_MOE = join(_MM, "moe_system")
for _p in (_MM, _SRC, _MOE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils_new import (  # noqa: E402
    build_aligned_list,
    default_data_paths,
    is_all_zero_labels,
    normalize_dataset_mode,
    set_dataset_mode,
)

TARGET_ATTRS = ["perf", "util_LUT", "util_FF", "util_DSP", "util_BRAM"]


def _row_vec(feat, dim: int, what: str) -> torch.Tensor:
    if hasattr(feat, "x") and not isinstance(feat, torch.Tensor):
        feat = feat.x
    if not isinstance(feat, torch.Tensor):
        feat = torch.as_tensor(feat, dtype=torch.float32)
    t = feat.float().detach()
    if t.dim() > 1:
        t = t.mean(dim=0)
    t = t.view(-1)
    if t.numel() != dim:
        raise RuntimeError(f"{what} 期望 {dim} 维，实际 {t.numel()}")
    return t


def _ast_vec(obj, path: str) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return _row_vec(obj, 784, f"AST({path})")
    if hasattr(obj, "x") and isinstance(obj.x, torch.Tensor):
        x = obj.x.float()
        if x.dim() == 2 and x.size(1) >= 784:
            return x[:, :784].mean(dim=0)
        if x.dim() == 2 and x.size(1) == 768:
            # 仅 CodeBERT 节点特征时补 16 维零
            pad = torch.zeros(x.size(0), 16, dtype=x.dtype)
            return torch.cat([x, pad], dim=1).mean(dim=0)
    raise RuntimeError(f"无法从 AST 提取 784 维: {path}")


def load_sample(rec: dict) -> Optional[Data]:
    cdfg = torch.load(rec["cdfg"], map_location="cpu")
    if not isinstance(cdfg, Data):
        raise RuntimeError(f"CDFG 不是 PyG Data: {rec['cdfg']}")
    text = torch.load(rec["text"], map_location="cpu")
    text_vec = _row_vec(text, 768, f"Text({rec['text']})")
    ast_obj = torch.load(rec["ast"], map_location="cpu")
    ast_vec = _ast_vec(ast_obj, rec["ast"])

    suite, kernel, idx = rec["key"]
    cdfg.kernel = kernel
    cdfg.suite = suite
    cdfg.cfg_id = int(idx)
    cdfg.file = f"data_{idx}.pt"
    cdfg.text_feat = text_vec
    cdfg.ast_feat = ast_vec
    return cdfg


def build_dataset(
    target_list=None,
    skip_all_zero: bool = True,
    kernels: Optional[List[str]] = None,
    dataset_mode: Optional[str] = None,
) -> List[Data]:
    """
    dataset_mode: filtered | not_0（None 则用全局 set_dataset_mode / 默认 filtered）
    """
    mode = normalize_dataset_mode(dataset_mode) if dataset_mode is not None else None
    if mode is not None:
        set_dataset_mode(mode)
    paths = default_data_paths(mode)
    target_list = target_list or TARGET_ATTRS
    aligned = build_aligned_list(dataset_mode=mode)
    if kernels is not None:
        allow = set(kernels)
        aligned = [r for r in aligned if r["key"][1] in allow]

    data_list: List[Data] = []
    skipped = 0
    for rec in aligned:
        try:
            d = load_sample(rec)
        except Exception:
            skipped += 1
            continue
        if skip_all_zero and is_all_zero_labels(d, target_list):
            skipped += 1
            continue
        data_list.append(d)
    print(
        f"[data_ablation] mode={paths['dataset_mode']} "
        f"tt={paths['tt_root']} "
        f"loaded={len(data_list)} skipped={skipped} aligned_raw={len(aligned)}",
        flush=True,
    )
    return data_list


class AblationDataset(Dataset):
    def __init__(self, data_list: List[Data]):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_ablation(batch: List[Data]):
    texts = torch.stack([b.text_feat for b in batch], dim=0)
    asts = torch.stack([b.ast_feat for b in batch], dim=0)
    has_pragma = all(hasattr(b, "pragma_feat") and b.pragma_feat is not None for b in batch)
    pragmas = (
        torch.stack([b.pragma_feat.float().view(-1) for b in batch], dim=0)
        if has_pragma
        else None
    )
    # 必须 clone 后再删：原地 del 会破坏 Dataset 里的样本，第 2 个 epoch 会缺 text_feat
    graphs = []
    for b in batch:
        d = b.clone()
        if hasattr(d, "text_feat"):
            del d.text_feat
        if hasattr(d, "ast_feat"):
            del d.ast_feat
        if hasattr(d, "pragma_feat"):
            del d.pragma_feat
        graphs.append(d)
    g = Batch.from_data_list(graphs)
    g.text_feat = texts
    g.ast_feat = asts
    if pragmas is not None:
        g.pragma_feat = pragmas
    return g


def default_paths(dataset_mode: Optional[str] = None) -> Dict[str, str]:
    return default_data_paths(dataset_mode)
