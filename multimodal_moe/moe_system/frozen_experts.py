"""
冻结的 7 个已训练专家。

与 algorithm_select / make_labels / expert_rmse 对齐：
  融合 (all / ast_cdfg / ast_text / cdfg_text):
      GNN ← CDFG 图
      text_feat ← [768]
      ast_feat  ← [784]（AST 图 mean-pool）
  cdfg_only: GNN ← CDFG
  ast_only : GNN ← data.ast_graph（AST 图）
  LM       : text_feat ← [768]（CDFG 仅承载 batch/标签）

主 Batch = CDFG；标签取自 CDFG。缺失/错模态 → 直接报错。
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

from experts import EXPERT_MODE_NAMES

try:
    from expert_rmse_eval_train_split import (  # noqa: E402
        STATIC_FILES,
        TARGET_NAMES,
        ExpertModelCache,
        apply_eval_mock_flags,
        _silence_model_forward_print,
        FUSION_EXPERT_MODES,
        MODE_TO_FORWARD_ABLATION,
    )
except ImportError as _e:
    STATIC_FILES = TARGET_NAMES = ExpertModelCache = None  # type: ignore
    apply_eval_mock_flags = _silence_model_forward_print = None  # type: ignore
    FUSION_EXPERT_MODES = MODE_TO_FORWARD_ABLATION = None  # type: ignore
    _FROZEN_IMPORT_ERR = _e
else:
    _FROZEN_IMPORT_ERR = None

MODE_DIM = {
    "all": 153,
    "ast_cdfg": 153,
    "ast_text": 153,
    "cdfg_text": 153,
    "cdfg_only": 153,
    "ast_only": 784,
    "LM": 153,
}


def _require_row(feat: torch.Tensor, dim: int, name: str) -> torch.Tensor:
    if not isinstance(feat, torch.Tensor):
        raise RuntimeError(f"{name} 必须是 Tensor，得到 {type(feat)}")
    t = feat.float().detach().view(-1)
    if t.numel() != dim:
        raise RuntimeError(f"{name} 期望长度 {dim}，实际 {t.numel()} shape={tuple(feat.shape)}")
    return t


class FrozenExpertBank:
    def __init__(self, weight_dir: str, device):
        if _FROZEN_IMPORT_ERR is not None or ExpertModelCache is None:
            raise ImportError(
                "FrozenExpertBank 依赖 MPM algorithm_select；典型 MoE 不需要本模块。"
                f" 原始错误: {_FROZEN_IMPORT_ERR}"
            )
        self.weight_dir = weight_dir
        self.device = torch.device(device)
        self.cache = ExpertModelCache("static", "", weight_dir, self.device)
        self.modes = list(EXPERT_MODE_NAMES)
        missing = []
        for mode in self.modes:
            path = os.path.join(weight_dir, STATIC_FILES[mode])
            got = self.cache.get(mode, MODE_DIM[mode], None)
            if got is None:
                missing.append(path)
            else:
                model, _ = got
                model.eval()
                for p in model.parameters():
                    p.requires_grad_(False)
        if missing:
            raise FileNotFoundError(
                "未找到已训练专家权重，无法在选中方案后调用 .pth:\n" + "\n".join(missing)
            )

    def _attach_labels(self, graph, label_src):
        """专家 Net.forward 会读 perf/util-*；标签在 CDFG 主样本上。"""
        for name in list(TARGET_NAMES) + ["actual_perf"]:
            attr = name.replace("-", "_")
            if not hasattr(label_src, attr):
                raise RuntimeError(f"CDFG 主样本缺少标签字段 {attr}，无法调用冻结专家")
            setattr(graph, attr, getattr(label_src, attr))
        return graph

    @torch.no_grad()
    def _forward_out(self, mode: str, cdfg, text_vec, ast_vec, ast_graph, label_src):
        apply_eval_mock_flags(mode)
        model, _ = self.cache.get(mode, MODE_DIM[mode], None)
        if model is None:
            raise RuntimeError(f"专家未加载: mode={mode}")
        model.eval()

        if mode == "ast_only":
            if ast_graph is None or not hasattr(ast_graph, "x"):
                raise RuntimeError("ast_only 需要 data.ast_graph 中的 AST 图")
            graph = self._attach_labels(ast_graph, label_src)
            batch = Batch.from_data_list([graph]).to(self.device)
            with _silence_model_forward_print():
                out_dict, _, _ = model(batch, ablation_mode="all")
            return out_dict

        if mode == "cdfg_only":
            if cdfg is None:
                raise RuntimeError("cdfg_only 需要 CDFG 图")
            graph = self._attach_labels(cdfg, label_src)
            batch = Batch.from_data_list([graph]).to(self.device)
            with _silence_model_forward_print():
                out_dict, _, _ = model(batch, ablation_mode="all")
            return out_dict

        if mode == "LM":
            text = _require_row(text_vec, 768, "LM/text_feat")
            if cdfg is None:
                raise RuntimeError("LM 需要 CDFG 图承载标签/batch")
            cdfg = self._attach_labels(cdfg, label_src)
            data = Batch.from_data_list([cdfg]).to(self.device)
            data.text_feat = text.view(1, 768).to(self.device)
            with _silence_model_forward_print():
                out_dict, _, _ = model(data, ablation_mode="all")
            return out_dict

        if mode in FUSION_EXPERT_MODES:
            text = _require_row(text_vec, 768, f"{mode}/text_feat")
            astv = _require_row(ast_vec, 784, f"{mode}/ast_feat")
            if cdfg is None:
                raise RuntimeError(f"{mode} 需要 CDFG 图")
            cdfg = self._attach_labels(cdfg, label_src)
            data = Batch.from_data_list([cdfg]).to(self.device)
            data.text_feat = text.view(1, 768).to(self.device)
            data.ast_feat = astv.view(1, 784).to(self.device)
            with _silence_model_forward_print():
                out_dict, _, _ = model(data, ablation_mode=MODE_TO_FORWARD_ABLATION[mode])
            return out_dict

        raise RuntimeError(f"未知专家模式: {mode}")

    @torch.no_grad()
    def predict_batch(self, data) -> torch.Tensor:
        """
        返回 [B, 7, n_targets]。

        要求 inject_modalities 已注入:
          cdfg_feat / 主 Batch = CDFG
          text_feat [B,768], ast_feat [B,784], ast_graph = AST Batch
        """
        if not hasattr(data, "cdfg_feat") or data.cdfg_feat is None:
            raise RuntimeError("缺少 data.cdfg_feat：请先 inject_modalities")
        if not hasattr(data, "text_feat") or data.text_feat is None:
            raise RuntimeError("缺少 data.text_feat：请先 inject_modalities")
        if not hasattr(data, "ast_feat") or data.ast_feat is None:
            raise RuntimeError("缺少 data.ast_feat（784 维 AST mean-pool）")
        if not hasattr(data, "ast_graph") or data.ast_graph is None:
            raise RuntimeError("缺少 data.ast_graph（AST 图 Batch，供 ast_only）")

        cdfg_list = data.cdfg_feat.to_data_list()
        ast_graphs = data.ast_graph.to_data_list()
        texts = data.text_feat.detach()
        ast_vecs = data.ast_feat.detach()
        b = len(cdfg_list)
        if len(ast_graphs) != b or texts.size(0) != b or ast_vecs.size(0) != b:
            raise RuntimeError(
                f"batch 长度不一致: cdfg={b} ast_graph={len(ast_graphs)} "
                f"text={texts.size(0)} ast_feat={ast_vecs.size(0)}"
            )
        if texts.size(-1) != 768:
            raise RuntimeError(f"text_feat 末维应为 768，得到 {tuple(texts.shape)}")
        if ast_vecs.size(-1) != 784:
            raise RuntimeError(f"ast_feat 末维应为 784，得到 {tuple(ast_vecs.shape)}")

        n_t = len(TARGET_NAMES)
        out = torch.zeros(b, len(self.modes), n_t, device=self.device)
        for i in range(b):
            label_src = cdfg_list[i]  # 标签在 CDFG 上（与 make_labels 一致）
            for m, mode in enumerate(self.modes):
                pred = self._forward_out(
                    mode,
                    cdfg_list[i],
                    texts[i],
                    ast_vecs[i],
                    ast_graphs[i],
                    label_src,
                )
                for t, name in enumerate(TARGET_NAMES):
                    if name not in pred:
                        raise RuntimeError(f"专家 {mode} 未输出指标 {name}")
                    out[i, m, t] = pred[name].view(-1)[0].float()
        return out
