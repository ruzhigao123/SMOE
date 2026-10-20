"""
三模态注入 / collate（与 algorithm_select 对齐：CDFG data_i、Text i+1、AST i.ast.pt）。
缺失即报错，禁止静默填零。不写任何旧产物目录。
"""
from __future__ import annotations

import os
import sys

import torch
from torch_geometric.data import Batch

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)  # multimodal_moe
_HMOE_ROOT = os.path.dirname(_ROOT)
_HMOE_SRC = os.path.join(_HMOE_ROOT, "src")
for _p in (_HMOE_SRC, _HMOE_ROOT, _ROOT, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.config import FLAGS
from utils_new import (
    default_data_paths,
    extract_id_from_filename,
    modality_paths_from_sample,
    require_aligned_paths,
)

PATHS = default_data_paths()
CDFG_ROOT = PATHS["cdfg_root"]
TEXT_ROOT = PATHS["text_root"]
AST_ROOT = PATHS["ast_pt_root"]


def _as_row_vector(feat, dim: int, what: str) -> torch.Tensor:
    if hasattr(feat, "x") and not isinstance(feat, torch.Tensor):
        feat = feat.x
    if not isinstance(feat, torch.Tensor):
        feat = torch.tensor(feat, dtype=torch.float32)
    t = feat.float().detach()
    if t.dim() > 1:
        t = t.mean(dim=0)
    t = t.view(-1)
    if t.numel() != dim:
        raise RuntimeError(f"{what} 期望长度 {dim}，实际 {t.numel()}")
    return t


def _ast_to_fusion_vec(ast_obj, path: str) -> torch.Tensor:
    if isinstance(ast_obj, torch.Tensor):
        return _as_row_vector(ast_obj, 784, f"AST({path})")
    if hasattr(ast_obj, "x") and isinstance(ast_obj.x, torch.Tensor):
        x = ast_obj.x.float().detach()
        if x.dim() != 2:
            raise RuntimeError(f"AST.x 维数不对: {path} shape={tuple(x.shape)}")
        if x.size(1) == 784:
            return x.mean(dim=0)
        if x.size(1) > 784:
            return x[:, :784].mean(dim=0)
        raise RuntimeError(f"AST 节点特征宽度 < 784: {path}")
    raise RuntimeError(f"AST 既非 Tensor 也非图: {path}")


def _resolve_paths(file_name, kernel_name, sample=None):
    if sample is not None:
        try:
            return modality_paths_from_sample(sample)
        except FileNotFoundError:
            pass
    base_id = extract_id_from_filename(file_name)
    if base_id is None:
        raise RuntimeError(f"无法解析配置 id: file={file_name!r} kernel={kernel_name!r}")
    return require_aligned_paths(CDFG_ROOT, TEXT_ROOT, AST_ROOT, kernel_name, base_id)


def inject_modalities(data):
    """注入 cdfg_feat / text_feat[B,768] / ast_feat[B,784] / ast_graph。"""
    text_list, ast_vecs, ast_graphs = [], [], []
    files = data.file if isinstance(data.file, (list, tuple)) else list(data.file)
    kernels = data.kernel if isinstance(data.kernel, (list, tuple)) else list(data.kernel)
    primary_list = data.to_data_list()
    if len(primary_list) != len(files):
        raise RuntimeError(f"主 Batch 图数 {len(primary_list)} 与 file 数 {len(files)} 不一致")

    for j in range(len(files)):
        sample = primary_list[j]
        cdfg_path, t_path, a_path = _resolve_paths(str(files[j]), str(kernels[j]), sample)
        t_feat = torch.load(t_path, map_location="cpu")
        t_vec = _as_row_vector(t_feat, 768, f"Text({t_path})")
        a_obj = torch.load(a_path, map_location="cpu")
        a_vec = _ast_to_fusion_vec(a_obj, a_path)
        if not hasattr(a_obj, "x"):
            raise RuntimeError(f"需要 AST 图: {a_path}")
        for name in (
            "perf", "actual_perf", "util_LUT", "util_FF", "util_DSP", "util_BRAM",
            "quality", "total_LUT", "total_FF", "total_DSP", "total_BRAM",
        ):
            if hasattr(sample, name):
                setattr(a_obj, name, getattr(sample, name))
        a_obj.file = getattr(sample, "file", files[j])
        a_obj.kernel = kernels[j]
        text_list.append(t_vec)
        ast_vecs.append(a_vec)
        ast_graphs.append(a_obj)

    data.cdfg_feat = data
    data.text_feat = torch.stack(text_list).to(FLAGS.device)
    data.ast_feat = torch.stack(ast_vecs).to(FLAGS.device)
    data.ast_graph = Batch.from_data_list(ast_graphs).to(FLAGS.device)
    return data


def custom_collate(batch):
    clean = []
    best_labels = []
    for d in batch:
        if hasattr(d, "point") and isinstance(getattr(d, "point"), dict):
            try:
                delattr(d, "point")
            except Exception:
                pass
        if hasattr(d, "meta_feat"):
            try:
                delattr(d, "meta_feat")
            except Exception:
                pass
        if hasattr(d, "moe_best_scheme"):
            best_labels.append(d.moe_best_scheme.detach().cpu().clone())
            try:
                delattr(d, "moe_best_scheme")
            except Exception:
                pass
        clean.append(d)
    out = Batch.from_data_list(clean)
    if best_labels:
        # [B, n_targets]
        out.moe_best_scheme = torch.stack(best_labels, dim=0)
    return out
