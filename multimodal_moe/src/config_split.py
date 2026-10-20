"""
配置级划分工具。

默认协议（与用户要求一致）:
  1) kernel 级：30 训练核 / 10 测试核（不变）
  2) 仅在 30 训练核的配置池内，按配置随机 8:2 → train / val
"""
from __future__ import annotations

import json
from os.path import join
from typing import Dict, List, Optional, Sequence, Tuple

from numpy.random import RandomState

from .get_root_path import get_root_path


def sample_key(d) -> Tuple[str, str, str]:
    suite = str(getattr(d, "suite", "") or "")
    kernel = str(getattr(d, "kernel", "") or "")
    file = str(getattr(d, "file", "") or getattr(d, "cfg_id", "") or "")
    return (suite, kernel, file)


def filter_by_keys(data_list: Sequence, keys: Sequence) -> List:
    want = {tuple(k) for k in keys}
    return [d for d in data_list if sample_key(d) in want]


def filter_by_kernels(data_list: Sequence, kernels: Sequence[str]) -> List:
    allow = set(kernels)
    return [d for d in data_list if getattr(d, "kernel", None) in allow]


def split_pool_train_val(
    pool: Sequence,
    seed: int = 64,
    train_ratio: float = 0.8,
) -> Dict:
    """在给定样本池内按配置随机划分 train/val（默认 8:2）。"""
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio 应在 (0,1)，得到 {train_ratio}")
    n = len(pool)
    if n < 2:
        raise ValueError(f"train_val 池过小：n={n}")

    rng = RandomState(seed)
    idx = rng.permutation(n)
    n_train = max(1, int(round(n * train_ratio)))
    if n_train >= n:
        n_train = n - 1
    train_i = idx[:n_train].tolist()
    val_i = idx[n_train:].tolist()
    train_list = [pool[i] for i in train_i]
    val_list = [pool[i] for i in val_i]

    def _keys(items):
        return [list(sample_key(d)) for d in items]

    return {
        "train_list": train_list,
        "val_list": val_list,
        "train_keys": _keys(train_list),
        "val_keys": _keys(val_list),
        "n_train": len(train_list),
        "n_val": len(val_list),
        "train_ratio": float(train_ratio),
        "seed": int(seed),
    }


def save_config_split(proto: dict, path: Optional[str] = None) -> str:
    path = path or join(get_root_path(), "ablation_outputs", "config_split.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(proto, f, indent=2)
    return path


def load_config_split(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
