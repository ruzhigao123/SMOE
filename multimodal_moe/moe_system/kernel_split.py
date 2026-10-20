"""
Kernel 划分：seed=64 → 15 训练 / 7 未见；15 上再均分 5 折（每折留出 3 个）。
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from numpy.random import permutation

MACHSUITE_KERNEL = ["aes", "gemm-blocked", "gemm-ncubed", "spmv-crs", "spmv-ellpack", "stencil", "nw"]
poly_KERNEL = [
    "2mm", "3mm", "adi", "atax", "bicg", "doitgen",
    "mvt", "fdtd-2d", "gemver", "gemm-p", "gesummv",
    "heat-3d", "jacobi-1d", "jacobi-2d", "seidel-2d",
]
ALL_KERNELS = MACHSUITE_KERNEL + poly_KERNEL

SEED = 64
N_TRAIN_KERNELS = 15
N_FOLDS = 5


def split_15_7(seed: int = SEED):
    """返回 (train_kernel_names, heldout_kernel_names)。"""
    np.random.seed(seed)
    shuffled = permutation(len(ALL_KERNELS))
    train_names = [ALL_KERNELS[i] for i in shuffled[:N_TRAIN_KERNELS]]
    heldout_names = [ALL_KERNELS[i] for i in shuffled[N_TRAIN_KERNELS:]]
    return train_names, heldout_names


def make_kernel_folds(train_kernel_names: Sequence[str], n_folds: int = N_FOLDS):
    """
    把 15 个训练 kernel 均分成 n_folds 组。
    返回 list[(train_names, val_names)]，val 每折 3 个。
    """
    names = list(train_kernel_names)
    if len(names) % n_folds != 0:
        raise ValueError(f"训练 kernel 数 {len(names)} 无法整除折数 {n_folds}")
    fold_size = len(names) // n_folds
    folds = []
    for i in range(n_folds):
        val_names = names[i * fold_size : (i + 1) * fold_size]
        tr_names = [k for k in names if k not in val_names]
        folds.append((tr_names, val_names))
    return folds


def filter_by_kernels(data_list, kernel_names: Sequence[str]):
    allow = set(kernel_names)
    return [d for d in data_list if getattr(d, "kernel", getattr(d, "gname", "N/A")) in allow]


def split_kernels_7_3(kernel_names: Sequence[str], ratio: float = 0.7, seed: int = SEED):
    """
    按 kernel 做 7:3 划分（不是按样本）。
    默认对 15 个训 kernel：约 10 训 / 5 验。
    """
    names = list(dict.fromkeys(kernel_names))  # 保序去重
    if len(names) < 2:
        raise ValueError(f"kernel 数过少，无法 7:3：{names}")
    np.random.seed(seed)
    order = permutation(len(names))
    n_tr = int(len(names) * ratio)
    n_tr = max(1, min(n_tr, len(names) - 1))
    tr_names = [names[i] for i in order[:n_tr]]
    va_names = [names[i] for i in order[n_tr:]]
    return tr_names, va_names


def partition_dataset(dataset):
    """对 Dataset.data_list 做 15/7 划分，返回 train_pool, heldout_list, train_names, heldout_names。"""
    train_names, heldout_names = split_15_7()
    train_pool = filter_by_kernels(dataset.data_list, train_names)
    heldout = filter_by_kernels(dataset.data_list, heldout_names)
    return train_pool, heldout, train_names, heldout_names
