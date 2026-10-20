"""
42 kernel 协议划分（无特征融合消融实验）：

1) 按 CDFG 配置数丢掉最少的 2 个 → 40 kernel
2) seed 固定：30 训练 / 10 测试（按 kernel）
3) 30 个训练 kernel 再按 8:2 → 24 训练 / 6 验证（按 kernel）
"""
from __future__ import annotations

import json
from os.path import join
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.random import RandomState

from .get_root_path import get_root_path

MACHSUITE_KERNEL = [
    "aes",
    "gemm-blocked",
    "gemm-ncubed",
    "spmv-crs",
    "spmv-ellpack",
    "stencil",
    "nw",
    "md",
    "stencil-3d",
]
POLY_KERNEL = [
    "2mm",
    "3mm",
    "adi",
    "atax",
    "bicg",
    "bicg-large",
    "covariance",
    "doitgen",
    "doitgen-red",
    "fdtd-2d",
    "fdtd-2d-large",
    "gemm-p",
    "gemm-p-large",
    "gemver",
    "gesummv",
    "heat-3d",
    "jacobi-1d",
    "jacobi-2d",
    "mvt",
    "seidel-2d",
    "symm",
    "symm-opt",
    "syrk",
    "syr2k",
    "trmm",
    "trmm-opt",
    "mvt-medium",
    "correlation",
    "atax-medium",
    "bicg-medium",
    "gesummv-medium",
    "symm-opt-medium",
    "gemver-medium",
]
ALL_KERNELS_42 = MACHSUITE_KERNEL + POLY_KERNEL

SEED = 50
N_DROP = 2
N_TRAIN = 30
N_TEST = 10
TRAIN_VAL_RATIO = 0.8  # 30 → 24 / 6


def count_cdfg_per_kernel(cdfg_root: Optional[str] = None) -> Dict[str, int]:
    root = Path(cdfg_root or join(get_root_path(), "two_tower_datasets", "graph"))
    counts: Dict[str, int] = {}
    for suite in ("machsuite", "poly"):
        suite_dir = root / suite
        if not suite_dir.is_dir():
            continue
        for kdir in suite_dir.iterdir():
            if kdir.is_dir():
                counts[kdir.name] = len(list(kdir.glob("data_*.pt")))
    return counts


def drop_two_fewest(
    all_kernels: Sequence[str] = ALL_KERNELS_42,
    counts: Optional[Dict[str, int]] = None,
    n_drop: int = N_DROP,
) -> Tuple[List[str], List[str]]:
    """返回 (kept_40, dropped_2)。并列时按名字排序保证可复现。"""
    counts = counts or count_cdfg_per_kernel()
    n_pos = sum(1 for k in all_kernels if counts.get(k, 0) > 0)
    if n_pos == 0:
        # 本地尚未生成 graph：用协议默认丢弃（历史上配置偏少），并告警
        print(
            "[kernel_protocol] WARNING: all CDFG counts are 0; "
            "temporarily drop ['md', 'seidel-2d']; re-run split_only after data is ready"
        )
        dropped = ["md", "seidel-2d"]
        kept = [k for k in all_kernels if k not in set(dropped)]
        return kept, dropped

    ranked = sorted(list(all_kernels), key=lambda k: (counts.get(k, 0), k))
    dropped = ranked[:n_drop]
    kept = [k for k in all_kernels if k not in set(dropped)]
    return kept, dropped


def split_30_10(
    kernels_40: Sequence[str],
    seed: int = SEED,
    n_train: int = N_TRAIN,
) -> Tuple[List[str], List[str]]:
    rs = RandomState(seed)
    order = rs.permutation(len(kernels_40))
    names = [kernels_40[i] for i in order]
    return names[:n_train], names[n_train:]


def split_train_val_8_2(
    train_kernels_30: Sequence[str],
    seed: int = SEED,
    ratio: float = TRAIN_VAL_RATIO,
) -> Tuple[List[str], List[str]]:
    rs = RandomState(seed + 1)
    names = list(train_kernels_30)
    order = rs.permutation(len(names))
    n_tr = int(round(len(names) * ratio))
    n_tr = max(1, min(n_tr, len(names) - 1))
    shuffled = [names[i] for i in order]
    return shuffled[:n_tr], shuffled[n_tr:]


def build_protocol(
    cdfg_root: Optional[str] = None,
    seed: int = SEED,
    save_json: Optional[str] = None,
) -> dict:
    counts = count_cdfg_per_kernel(cdfg_root)
    kept40, dropped = drop_two_fewest(ALL_KERNELS_42, counts)
    if len(kept40) != 40:
        # 若本地尚未生成完整 graph，仍按名单凑满逻辑，但标出缺失
        missing = [k for k in ALL_KERNELS_42 if counts.get(k, 0) == 0]
    else:
        missing = []
    train30, test10 = split_30_10(kept40, seed=seed)
    train24, val6 = split_train_val_8_2(train30, seed=seed)
    proto = {
        "seed": seed,
        "all_42": list(ALL_KERNELS_42),
        "counts": {k: int(counts.get(k, 0)) for k in ALL_KERNELS_42},
        "dropped_2": dropped,
        "kept_40": kept40,
        "train_30": train30,
        "test_10": test10,
        "train_24": train24,
        "val_6": val6,
        "missing_cdfg": missing,
    }
    if save_json:
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(proto, f, indent=2, ensure_ascii=False)
    return proto


def filter_by_kernels(samples: Sequence, kernel_names: Sequence[str]):
    allow = set(kernel_names)
    out = []
    for s in samples:
        k = s.get("kernel") if isinstance(s, dict) else getattr(s, "kernel", None)
        if k in allow:
            out.append(s)
    return out


if __name__ == "__main__":
    p = build_protocol(save_json=join(get_root_path(), "ablation_outputs", "kernel_split.json"))
    print("dropped_2", p["dropped_2"])
    print("train_30", len(p["train_30"]), p["train_30"])
    print("test_10", len(p["test_10"]), p["test_10"])
    print("train_24/val_6", len(p["train_24"]), len(p["val_6"]))
    print("saved", join(get_root_path(), "ablation_outputs", "kernel_split.json"))
