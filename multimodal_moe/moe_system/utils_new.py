"""
MoE 辅助工具：全零过滤、三模态路径对齐。

# 权威约定（HMoE 迁移版）:
#   filtered（默认，过滤零指标）:
#     CDFG/Text = two_tower_datasets/{graph,code}/...
#     .c / AST  = two_tower_datasets_c/code/...
#   not_0（保留零指标）:
#     CDFG/Text = two_tower_datasets_not_0/{graph,code}/...
#     .c / AST  = two_tower_datasets_c_not_0/code/...
"""
from __future__ import annotations

import glob
import os
import re
from os.path import abspath, basename, dirname, join
from typing import Dict, List, Optional, Tuple

# 项目根目录: .../HierarchicalMoE-master/multimodal_moe
_MOE_DIR = dirname(abspath(__file__))
_ROOT = dirname(_MOE_DIR)

# filtered = 过滤零指标（原版）；not_0 = 保留零指标
DATASET_MODES = ("filtered", "not_0")
_DEFAULT_DATASET_MODE = "filtered"
_ACTIVE_DATASET_MODE = _DEFAULT_DATASET_MODE


def get_project_root() -> str:
    return _ROOT


def normalize_dataset_mode(mode: Optional[str] = None) -> str:
    """统一别名：del_0/filtered → filtered；keep_zero/all/not_0 → not_0。"""
    m = str(mode or _ACTIVE_DATASET_MODE).strip().lower()
    if m in ("filtered", "del_0", "filter", "zero_filtered", "drop_zero"):
        return "filtered"
    if m in ("not_0", "not0", "keep_zero", "keep0", "all", "unfiltered"):
        return "not_0"
    raise ValueError(
        f"未知 dataset_mode={mode!r}，可选: filtered|not_0 "
        f"（别名 del_0 / keep_zero）"
    )


def set_dataset_mode(mode: str) -> str:
    """设置全局默认数据集模式（后续 default_data_paths / build_aligned_list 生效）。"""
    global _ACTIVE_DATASET_MODE
    _ACTIVE_DATASET_MODE = normalize_dataset_mode(mode)
    return _ACTIVE_DATASET_MODE


def get_dataset_mode() -> str:
    return _ACTIVE_DATASET_MODE


def dataset_root_names(mode: Optional[str] = None) -> Dict[str, str]:
    """返回 (tt_name, ttc_name)。"""
    m = normalize_dataset_mode(mode)
    if m == "not_0":
        return {
            "tt": "two_tower_datasets_not_0",
            "ttc": "two_tower_datasets_c_not_0",
        }
    return {
        "tt": "two_tower_datasets",
        "ttc": "two_tower_datasets_c",
    }


def seed_tag(seed: int) -> str:
    """多随机种子实验目录名，如 seed_64。"""
    return f"seed_{int(seed)}"


def ensure_seed_outdir(path: str, seed: int) -> str:
    """
    将产物目录落到 …/seed_{seed}/，避免多 seed 互相覆盖。
    若 path 已以 seed_{seed} 结尾则原样返回。
    """
    if not path:
        return path
    tag = seed_tag(seed)
    base = basename(path.rstrip("\\/"))
    if base == tag:
        return path
    return join(path, tag)


def default_output_dirs(
    dataset_mode: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, str]:
    """
    训练产物目录（按 dataset_mode + seed 隔离，避免互相覆盖）。

    filtered → ablation_outputs[/seed_N] / cv10_outputs[/seed_N] / moe_cv10_outputs[/seed_N]
    not_0    → ablation_outputs_not_0[/seed_N] / …
    传入 seed 时自动追加 seed_{seed} 子目录（对齐 HMoE 多 seed 实验）。
    """
    m = normalize_dataset_mode(dataset_mode)
    suffix = "_not_0" if m == "not_0" else ""
    out = {
        "dataset_mode": m,
        "ablation_root": join(_ROOT, f"ablation_outputs{suffix}"),
        "cv10_root": join(_ROOT, f"cv10_outputs{suffix}"),
        "moe_cv10_root": join(_ROOT, f"moe_cv10_outputs{suffix}"),
        "pipeline_root": join(_ROOT, f"moe_pipeline_outputs{suffix}"),
    }
    if seed is not None:
        tag = seed_tag(seed)
        out["seed"] = int(seed)
        out["seed_tag"] = tag
        for k in ("ablation_root", "cv10_root", "moe_cv10_root", "pipeline_root"):
            out[k] = join(out[k], tag)
    return out


def default_data_paths(dataset_mode: Optional[str] = None):
    """
    相对 multimodal_moe 根目录的默认三模态路径。

    dataset_mode:
      filtered — two_tower_datasets + two_tower_datasets_c
      not_0    — two_tower_datasets_not_0 + two_tower_datasets_c_not_0
    """
    m = normalize_dataset_mode(dataset_mode)
    names = dataset_root_names(m)
    outs = default_output_dirs(m)
    tt = join(_ROOT, names["tt"])
    ttc = join(_ROOT, names["ttc"])
    return {
        "dataset_mode": m,
        "cdfg_root": join(tt, "graph"),
        "text_root": join(tt, "code"),
        "points_root": join(tt, "points"),
        "ast_pt_root": join(ttc, "code", "ast_final_dataset_1"),
        "c_src_root": join(ttc, "code"),
        "c_graph_root": join(ttc, "graph"),
        "c_points_root": join(ttc, "points"),
        "save_dir": outs["pipeline_root"],
        "ablation_root": outs["ablation_root"],
        "cv10_root": outs["cv10_root"],
        "moe_cv10_root": outs["moe_cv10_root"],
        "tt_root": tt,
        "ttc_root": ttc,
    }

def extract_id_from_filename(name) -> Optional[int]:
    m = re.search(r"\d+", str(name))
    return int(m.group()) if m else None


def is_all_zero_labels(data, target_list, eps: float = 1e-12) -> bool:
    """与 train_del_0 / select_train_del_0 一致：五指标真值全为 0 则视为无效配置。"""
    for name in target_list:
        attr = name.replace("-", "_")
        try:
            y = getattr(data, attr).view(-1)
            v = float(y.mean().item()) if y.numel() > 0 else 0.0
        except Exception:
            return False
        if abs(v) > eps:
            return False
    return True


def filter_all_zero_samples(data_list, target_list, log_fn=None):
    kept, removed = [], 0
    for d in data_list:
        if is_all_zero_labels(d, target_list):
            removed += 1
            continue
        kept.append(d)
    if log_fn is not None:
        log_fn(f"[MoE/del_0] 全零过滤: 移除 {removed}，保留 {len(kept)} / {len(data_list)}")
    return kept


def _build_index(root: str, pattern: str, regex: str) -> Dict[Tuple[str, str, int], str]:
    """(suite, kernel, id) -> path；与 get_aligned_data / build_aligned_list 相同。"""
    idx: Dict[Tuple[str, str, int], str] = {}
    for f in glob.glob(os.path.join(root, "**", pattern), recursive=True):
        f = os.path.normpath(f)
        parts = f.split(os.sep)
        m = re.search(regex, parts[-1])
        if m and len(parts) >= 3:
            idx[(parts[-3], parts[-2], int(m.group()))] = f
    return idx


def build_aligned_list(
    cdfg_root: Optional[str] = None,
    text_root: Optional[str] = None,
    ast_root: Optional[str] = None,
    dataset_mode: Optional[str] = None,
) -> List[dict]:
    """
    三模态严格交集（Text 键为 i+1）。
    返回 [{"key": (suite, kernel, i), "cdfg": path, "text": path, "ast": path}, ...]
    """
    paths = default_data_paths(dataset_mode)
    cdfg_root = cdfg_root or paths["cdfg_root"]
    text_root = text_root or paths["text_root"]
    ast_root = ast_root or paths["ast_pt_root"]

    cdfg_idx = _build_index(cdfg_root, "data_*.pt", r"\d+")
    text_idx = _build_index(text_root, "*.pt", r"\d+")
    ast_idx = _build_index(ast_root, "*.ast.pt", r"\d+")
    aligned = []
    for k in cdfg_idx:
        suite, kernel, i = k
        t_key = (suite, kernel, i + 1)
        if t_key in text_idx and k in ast_idx:
            aligned.append(
                {
                    "key": k,
                    "cdfg": cdfg_idx[k],
                    "text": text_idx[t_key],
                    "ast": ast_idx[k],
                }
            )
    return aligned


def resolve_text_path(text_root: str, kernel: str, text_id: int) -> Optional[str]:
    for sub in ("poly", "machsuite"):
        p = join(text_root, sub, kernel, f"{text_id}.pt")
        if os.path.exists(p):
            return p
    return None


def resolve_ast_graph_path(ast_root: str, kernel: str, graph_index: int) -> Optional[str]:
    """AST：优先层级 {suite}/{kernel}/{i}.ast.pt，再尝试扁平 {i}.ast.pt。"""
    for suite in ("poly", "machsuite"):
        p = join(ast_root, suite, kernel, f"{int(graph_index)}.ast.pt")
        if os.path.exists(p):
            return p
    p = join(ast_root, f"{int(graph_index)}.ast.pt")
    return p if os.path.exists(p) else None


def resolve_ast_vec_path(ast_root: str, graph_index: int) -> Optional[str]:
    p = join(ast_root, f"{int(graph_index)}.ast.pt")
    return p if os.path.exists(p) else None


def resolve_ast_path(ast_root: str, ast_id: int) -> Optional[str]:
    return resolve_ast_vec_path(ast_root, ast_id)


def resolve_cdfg_path(cdfg_root: str, kernel: str, cdfg_id: int) -> Optional[str]:
    for suite in ("poly", "machsuite"):
        p = join(cdfg_root, suite, kernel, f"data_{int(cdfg_id)}.pt")
        if os.path.exists(p):
            return p
    p = join(cdfg_root, f"data_{int(cdfg_id)}.pt")
    return p if os.path.exists(p) else None


def resolve_c_path(c_src_root: str, kernel: str, c_id: int) -> Optional[str]:
    for suite in ("poly", "machsuite"):
        p = join(c_src_root, suite, kernel, f"{int(c_id)}.c")
        if os.path.exists(p):
            return p
    return None


def require_aligned_paths(
    cdfg_root: str,
    text_root: str,
    ast_root: str,
    kernel: str,
    graph_index: int,
) -> Tuple[str, str, str]:
    """
    严格对齐：CDFG=data_i, Text=i+1, AST=i.ast.pt；任一缺失直接报错。
    """
    cdfg = resolve_cdfg_path(cdfg_root, kernel, graph_index)
    text = resolve_text_path(text_root, kernel, graph_index + 1)
    ast = resolve_ast_graph_path(ast_root, kernel, graph_index)
    missing = []
    if cdfg is None:
        missing.append(
            f"CDFG: {cdfg_root}/{{poly|machsuite}}/{kernel}/data_{graph_index}.pt"
        )
    if text is None:
        missing.append(
            f"Text: {text_root}/{{poly|machsuite}}/{kernel}/{graph_index + 1}.pt"
        )
    if ast is None:
        missing.append(
            f"AST: {ast_root}/{{poly|machsuite}}/{kernel}/{graph_index}.ast.pt"
            f" 或 {ast_root}/{graph_index}.ast.pt"
        )
    if missing:
        raise FileNotFoundError(
            "三模态路径未对齐（get_aligned_data / build_aligned_list 约定），禁止静默填零:\n"
            "  - " + "\n  - ".join(missing)
        )
    return cdfg, text, ast


def modality_paths_from_sample(data) -> Tuple[str, str, str]:
    """若样本在加载时已写入对齐路径，直接返回。"""
    cdfg = getattr(data, "moe_cdfg_path", None)
    text = getattr(data, "moe_text_path", None)
    ast = getattr(data, "moe_ast_path", None)
    if cdfg and text and ast and all(os.path.exists(p) for p in (cdfg, text, ast)):
        return str(cdfg), str(text), str(ast)
    raise FileNotFoundError("样本缺少 moe_*_path 对齐字段")
