"""加载 / 生成与 Soft MoE 一致的 kernel 协议划分。"""
from __future__ import annotations

import json
import os
from os.path import abspath, dirname, join
from typing import Dict, List, Optional, Tuple

_THIS = dirname(abspath(__file__))
_MM = dirname(_THIS)
_HMOE = dirname(_MM)


def project_roots() -> Dict[str, str]:
    return {
        "hmoe_root": _HMOE,
        "mm_root": _MM,
        "src_root": join(_HMOE, "src"),
        "out_root": join(_MM, "hmoe_repro_outputs"),
    }


def seed_tag(seed: int) -> str:
    return f"seed_{int(seed)}"


def _split_candidates(seed: int, dataset_mode: str = "filtered") -> List[str]:
    """按优先级列出可能的 kernel_split.json 路径。"""
    from multimodal_moe.moe_system.utils_new import default_output_dirs

    outs = default_output_dirs(dataset_mode, seed=int(seed))
    outs_nos = default_output_dirs(dataset_mode, seed=None)
    return [
        join(outs["ablation_root"], "kernel_split.json"),  # …/ablation_outputs[_not_0]/seed_N/
        join(outs_nos["ablation_root"], "kernel_split.json"),  # …/ablation_outputs[_not_0]/（旧布局）
        join(project_roots()["out_root"], seed_tag(seed), "kernel_split.json"),
    ]


def default_split_path(seed: int, dataset_mode: str = "filtered") -> str:
    """优先复用 ablation 同 seed 的划分；否则落到 hmoe_repro_outputs/seed_N/。"""
    for cand in _split_candidates(seed, dataset_mode):
        if not os.path.isfile(cand):
            continue
        try:
            with open(cand, encoding="utf-8") as f:
                proto = json.load(f)
            if int(proto.get("seed", -1)) == int(seed):
                return cand
        except Exception:
            continue
    return join(project_roots()["out_root"], seed_tag(seed), "kernel_split.json")


def load_or_build_protocol(
    seed: int,
    dataset_mode: str = "filtered",
    split_json: Optional[str] = None,
    rebuild: bool = False,
) -> Tuple[dict, str]:
    path = split_json or default_split_path(seed, dataset_mode)
    if (not rebuild) and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            proto = json.load(f)
        if int(proto.get("seed", -1)) != int(seed):
            raise ValueError(
                f"{path} 的 seed={proto.get('seed')} 与请求 seed={seed} 不一致；"
                "请换路径或加 --rebuild_split"
            )
        return proto, path

    from multimodal_moe.moe_system.utils_new import (
        default_data_paths,
        normalize_dataset_mode,
        set_dataset_mode,
    )
    from multimodal_moe.src.kernel_protocol import build_protocol

    ds = normalize_dataset_mode(dataset_mode)
    set_dataset_mode(ds)
    cdfg_root = default_data_paths(ds)["cdfg_root"]
    os.makedirs(dirname(path) or ".", exist_ok=True)
    proto = build_protocol(cdfg_root=cdfg_root, seed=int(seed), save_json=path)
    return proto, path


def split_machsuite_poly(kernel_names: List[str]) -> Tuple[List[str], List[str]]:
    from multimodal_moe.src.kernel_protocol import MACHSUITE_KERNEL

    ms_set = set(MACHSUITE_KERNEL)
    machsuite, poly = [], []
    for k in kernel_names:
        if k in ms_set:
            machsuite.append(k)
        else:
            poly.append(k)
    return machsuite, poly
