"""
冻结 AblationNet 专家库。

expert_source:
  cv10     — train_30 按 hold-out 折加载 cv10_outputs；test_10 用 ablation_outputs（当前默认）
  ablation — 全部样本用 ablation_outputs/{cdfg_only,ast_only,LM}/best.pt
"""
from __future__ import annotations

import json
import os
import sys
from os.path import abspath, dirname, join
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Batch

_THIS = dirname(abspath(__file__))
_MM = dirname(_THIS)
_HMOE = dirname(_MM)
for _p in (_HMOE, _MM, _THIS, join(_MM, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from multimodal_moe.src.model_ablation import AblationNet  # noqa: E402

EXPERT_MODES = ("cdfg_only", "ast_only", "LM")
DEFAULT_TARGETS = ["perf", "util-LUT", "util-FF", "util-DSP", "util-BRAM"]


def _mm_root() -> str:
    return _MM


def load_folds_json(path: Optional[str] = None) -> dict:
    path = path or join(_mm_root(), "cv10_outputs", "folds.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少 folds.json: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_kernel_split(path: Optional[str] = None) -> dict:
    path = path or join(_mm_root(), "ablation_outputs", "kernel_split.json")
    if not os.path.isfile(path):
        alt = join(_mm_root(), "cv10_outputs", "kernel_split_ref.json")
        if os.path.isfile(alt):
            path = alt
        else:
            raise FileNotFoundError(f"缺少 kernel_split.json: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_kernel_to_fold(folds_payload: dict) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for fd in folds_payload["folds"]:
        fid = int(fd["fold_id"])
        for k in fd["holdout_kernels"]:
            if k in mapping:
                raise ValueError(f"kernel {k} 出现在多个 holdout 折")
            mapping[k] = fid
    return mapping


def _ckpt_path_cv10(cv10_root: str, fold_id: int, mode: str) -> str:
    return join(cv10_root, f"fold_{fold_id}", mode, "best.pt")


def _ckpt_path_ablation(ablation_root: str, mode: str) -> str:
    return join(ablation_root, mode, "best.pt")


def _load_ablation_net(ckpt_path: str, device: torch.device) -> AblationNet:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"缺少专家权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    mode = ckpt.get("ablation_mode")
    if mode not in EXPERT_MODES:
        mode = os.path.basename(os.path.dirname(ckpt_path))
    target = list(ckpt.get("target") or DEFAULT_TARGETS)
    in_dim = int(ckpt["in_dim"])
    # ckpt 带超参则按 ckpt；缺省回退旧 MPM GCN
    if "gnn_type" in ckpt or "D" in ckpt:
        D = int(ckpt.get("D", 128))
        hidden_num = int(ckpt.get("hidden_num", D))
        gnn_layers = int(ckpt.get("gnn_layers", 5))
        dropout = float(ckpt.get("dropout", 0.2))
        gnn_type = str(ckpt.get("gnn_type", "gcn"))
        gnn_activation = str(
            ckpt.get("gnn_activation", "relu" if gnn_type == "gcn" else "elu")
        )
    else:
        D, hidden_num, gnn_layers, dropout, gnn_type, gnn_activation = (
            128,
            128,
            5,
            0.2,
            "gcn",
            "relu",
        )
    state = ckpt["model"]
    # State keys are authoritative, including old checkpoints without backbone metadata.
    has_comparative = any(k.startswith("cdfg_enc.comp_backend.") for k in state)
    has_standard = any(k.startswith("cdfg_enc.convs.") for k in state)
    if has_comparative == has_standard:
        raise ValueError(f"无法唯一识别专家 CDFG 骨干: {ckpt_path}")
    if has_standard:
        layer_ids = sorted({int(k.split(".")[2]) for k in state
                            if k.startswith("cdfg_enc.convs.")})
        if layer_ids != list(range(len(layer_ids))):
            raise ValueError(f"专家 GNN 层编号不连续: {ckpt_path}")
        gnn_layers = len(layer_ids)
        projection = state["cdfg_enc.first_MLP.weight"]
        D = int(projection.shape[0])
        hidden_num = int(projection.shape[1]) // 2
        gnn_type = "transformer" if "cdfg_enc.convs.0.lin_query.weight" in state else "gcn"
        gnn_activation = str(ckpt.get("gnn_activation", "relu" if gnn_type == "gcn" else "elu"))
    comparative_model = ckpt.get("comparative_model")
    if has_comparative and not comparative_model:
        raise ValueError(f"对比骨干专家缺少 comparative_model 元数据，无法可靠恢复: {ckpt_path}")
    model = AblationNet(
        in_cdfg=in_dim,
        target_list=target,
        ablation_mode=mode,
        D=D,
        hidden_num=hidden_num,
        gnn_layers=gnn_layers,
        dropout=dropout,
        out_dim=1,
        gnn_type=gnn_type,
        gnn_activation=gnn_activation,
        comparative_if=has_comparative,
        comparative_model=str(comparative_model or "ironman"),
    ).to(device)
    try:
        model.load_state_dict(state)
    except RuntimeError as error:
        raise RuntimeError(f"专家权重结构不匹配: {ckpt_path}\n{error}") from error
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class Cv10AblationExpertBank(nn.Module):
    """冻结三模态专家。predict → [B, 3, T]（顺序 = EXPERT_MODES）。"""

    def __init__(
        self,
        device: torch.device,
        cv10_root: Optional[str] = None,
        ablation_root: Optional[str] = None,
        folds_json: Optional[str] = None,
        kernel_split_json: Optional[str] = None,
        target_list: Optional[Sequence[str]] = None,
        preload_all: bool = True,
        expert_source: str = "cv10",
    ):
        super().__init__()
        if expert_source not in ("ablation", "cv10"):
            raise ValueError(f"expert_source 应为 ablation|cv10，得到 {expert_source}")
        self.expert_source = expert_source
        self.device = torch.device(device)
        self.cv10_root = cv10_root or join(_mm_root(), "cv10_outputs")
        self.ablation_root = ablation_root or join(_mm_root(), "ablation_outputs")
        self.target_list = list(target_list or DEFAULT_TARGETS)
        self.n_targets = len(self.target_list)

        split_payload = load_kernel_split(kernel_split_json)
        self.test_10 = list(split_payload["test_10"])
        if "train_30" in split_payload:
            self.train_30 = list(split_payload["train_30"])
        else:
            folds_payload = load_folds_json(folds_json)
            self.train_30 = list(folds_payload["train_30"])

        self.kernel_to_fold: Dict[str, int] = {}
        if expert_source == "cv10":
            folds_payload = load_folds_json(folds_json)
            folds_train_30 = list(folds_payload["train_30"])
            abl_train = set(self.train_30)
            fold_train = set(folds_train_30)
            if abl_train != fold_train:
                raise RuntimeError(
                    "kernel_split.json 与 folds.json 的 train_30 不一致"
                    f"（多半 seed 不同：ablation_seed={split_payload.get('seed')} "
                    f"folds_seed={folds_payload.get('seed')}）。\n"
                    f"  仅在 ablation: {sorted(abl_train - fold_train)}\n"
                    f"  仅在 folds: {sorted(fold_train - abl_train)}\n"
                    "请用同一 --seed 重跑：\n"
                    "  python -m multimodal_moe.src.main_ablation --seed 50 --subtask split_only\n"
                    "  python -m multimodal_moe.src.cv10_single_modal --seed 50 "
                    "--cv_subtask split_only --rebuild_folds\n"
                    "然后按新划分重训 ablation / cv10 专家。"
                )
            self.train_30 = folds_train_30
            self.kernel_to_fold = build_kernel_to_fold(folds_payload)
            missing_hold = [k for k in self.train_30 if k not in self.kernel_to_fold]
            if missing_hold:
                raise RuntimeError(f"train_30 中有核未出现在任何 holdout: {missing_hold}")

        overlap = set(self.train_30) & set(self.test_10)
        if overlap:
            raise RuntimeError(
                f"train_30 与 test_10 不应相交: {sorted(overlap)}\n"
                f"（ablation_seed={split_payload.get('seed')}；"
                "请保证 kernel_split.json 与 folds.json 用同一 seed 生成）"
            )

        self._cache: Dict[Tuple, Dict[str, AblationNet]] = {}
        self._missing: List[str] = []
        self.modal_dim: Optional[int] = None
        if preload_all:
            self._preload()
            if self._missing:
                raise FileNotFoundError(
                    "以下专家权重缺失:\n  " + "\n  ".join(self._missing)
                )

    def _register_bundle(self, key: Tuple, paths: Dict[str, str]):
        bundle = {}
        for mode in EXPERT_MODES:
            p = paths[mode]
            if not os.path.isfile(p):
                self._missing.append(p)
                continue
            bundle[mode] = _load_ablation_net(p, self.device)
        if len(bundle) == len(EXPERT_MODES):
            self._cache[key] = bundle
            if self.modal_dim is None:
                self.modal_dim = int(bundle["cdfg_only"].D)
            elif int(bundle["cdfg_only"].D) != int(self.modal_dim):
                raise RuntimeError(
                    f"专家 D 不一致: bank.modal_dim={self.modal_dim} "
                    f"vs {key} D={bundle['cdfg_only'].D}"
                )

    def _preload(self):
        abl_paths = {m: _ckpt_path_ablation(self.ablation_root, m) for m in EXPERT_MODES}
        self._register_bundle(("ablation",), abl_paths)
        if self.expert_source == "cv10":
            for fid in sorted(set(self.kernel_to_fold.values())):
                paths = {
                    m: _ckpt_path_cv10(self.cv10_root, fid, m) for m in EXPERT_MODES
                }
                self._register_bundle(("fold", fid), paths)

    def route_key(self, kernel: str) -> Tuple:
        if self.expert_source == "ablation":
            return ("ablation",)
        if kernel in self.kernel_to_fold:
            return ("fold", int(self.kernel_to_fold[kernel]))
        if kernel in set(self.test_10):
            return ("ablation",)
        raise KeyError(f"kernel={kernel} 既不在 train_30 holdout 映射也不在 test_10")

    def get_bundle(self, key: Tuple) -> Dict[str, AblationNet]:
        if key not in self._cache:
            if key[0] == "fold":
                fid = int(key[1])
                paths = {
                    m: _ckpt_path_cv10(self.cv10_root, fid, m) for m in EXPERT_MODES
                }
            else:
                paths = {
                    m: _ckpt_path_ablation(self.ablation_root, m) for m in EXPERT_MODES
                }
            miss = [p for p in paths.values() if not os.path.isfile(p)]
            if miss:
                raise FileNotFoundError("缺失权重:\n  " + "\n  ".join(miss))
            self._register_bundle(key, paths)
        return self._cache[key]

    @torch.no_grad()
    def _forward_bundle(self, bundle: Dict[str, AblationNet], batch: Batch) -> torch.Tensor:
        preds = []
        for mode in EXPERT_MODES:
            out_dict, _, _ = bundle[mode](batch)
            cols = []
            for t in self.target_list:
                if t not in out_dict:
                    raise RuntimeError(f"专家 {mode} 缺少目标 {t}")
                cols.append(out_dict[t].view(-1, 1))
            preds.append(torch.cat(cols, dim=1))
        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def predict_homogeneous_batch(self, batch: Batch, route_key: Tuple) -> torch.Tensor:
        bundle = self.get_bundle(route_key)
        return self._forward_bundle(bundle, batch.to(self.device))

    @torch.no_grad()
    def encode_gate_feat(self, batch: Batch, route_key: Tuple) -> torch.Tensor:
        """
        用冻结 AblationNet 三塔编码样本 → [B, 384]。
        含 CDFG 图结构 + Text + AST(含 pragma/HW)，随 kernel/配置变化。
        """
        bundle = self.get_bundle(route_key)
        batch = batch.to(self.device)
        # 任意专家都带齐三塔；用 cdfg_only 取编码即可
        net = bundle["cdfg_only"]
        h_g = net.cdfg_enc(batch)
        h_s = net.text_proj(batch.text_feat.float())
        h_a = net.ast_proj(batch.ast_feat.float())
        return torch.cat([h_g, h_s, h_a], dim=-1)

    @torch.no_grad()
    def encode_selector_feat(self, batch: Batch, route_key: Tuple) -> torch.Tensor:
        """Selector branch: each modality uses its own trained frozen expert.

        Retain CDFG/Text/AST concatenation order. The legacy encoder above is
        unchanged so existing checkpoints and experiments remain reproducible.
        """
        bundle = self.get_bundle(route_key)
        batch = batch.to(self.device)
        h_g = bundle["cdfg_only"].cdfg_enc(batch)
        h_s = bundle["LM"].text_proj(batch.text_feat.float())
        h_a = bundle["ast_only"].ast_proj(batch.ast_feat.float())
        if h_g.shape != h_s.shape or h_g.shape != h_a.shape:
            raise ValueError("selector 门控要求三专家编码的维度一致")
        return torch.cat([h_g, h_s, h_a], dim=-1)
