"""
阶段一：5 折 × 7 融合方案，在未见 fold kernel 上得到 OOF 误差标签。

- 仅使用 seed=64 的 15 个训练 kernel；不加载 model_select_weight。
- 每折：12 个 kernel 训 300 epoch，3 个未见 kernel 上算各方案误差。
- 产物写入 moe_pipeline_outputs/（fold_experts + labels），禁止写 save_models_and_data。
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from os.path import join
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)  # multimodal_moe
_HMOE_ROOT = os.path.dirname(_ROOT)
_HMOE_SRC = join(_HMOE_ROOT, "src")
for _p in (_HMOE_SRC, _HMOE_ROOT, _ROOT, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.config import FLAGS
from src.saver import saver
from src.utils import OurTimer, _get_y_with_target

try:
    from expert_eval_net import ExpertEvalNet
    from expert_rmse_eval_train_split import (
        MODE_TO_FORWARD_ABLATION,
        apply_eval_mock_flags,
        _silence_model_forward_print,
    )
except ImportError as _e:
    ExpertEvalNet = None  # type: ignore
    MODE_TO_FORWARD_ABLATION = apply_eval_mock_flags = _silence_model_forward_print = None  # type: ignore
    _LABEL_GEN_IMPORT_ERR = _e
else:
    _LABEL_GEN_IMPORT_ERR = None

from pipeline_paths import (
    SCHEME_DIM,
    SCHEME_NAMES,
    TARGET_NAMES,
    ensure_pipeline_dirs,
    fold_expert_path,
    labels_csv_path,
    labels_json_path,
    FOLD_EXPERT_DIR,
    LABEL_DIR,
)
from kernel_split import make_kernel_folds, partition_dataset, filter_by_kernels
from modal_io import custom_collate, inject_modalities


def _device():
    return torch.device(FLAGS.device if torch.cuda.is_available() or str(FLAGS.device) == "cpu" else "cpu")


def _build_expert(mode: str) -> ExpertEvalNet:
    apply_eval_mock_flags(mode)
    in_ch = SCHEME_DIM[mode]
    model = ExpertEvalNet(in_channels=in_ch, target_list=list(TARGET_NAMES), task="regression")
    return model.to(_device())


def _forward_expert(model, data, mode: str):
    ablation = MODE_TO_FORWARD_ABLATION.get(mode, "all")
    # ast_only / cdfg_only：主图分别为 AST / CDFG
    if mode == "ast_only":
        batch = data.ast_graph
        # 标签在 CDFG 主 Batch；拷到 AST batch
        for name in TARGET_NAMES:
            attr = name.replace("-", "_")
            if hasattr(data, attr):
                setattr(batch, attr, getattr(data, attr))
        with _silence_model_forward_print():
            return model(batch, ablation_mode="all")
    if mode == "cdfg_only":
        with _silence_model_forward_print():
            return model(data, ablation_mode="all")
    if mode == "LM":
        with _silence_model_forward_print():
            return model(data, ablation_mode="all")
    with _silence_model_forward_print():
        return model(data, ablation_mode=ablation)


def _train_one_expert(mode: str, train_list, fold_idx: int, epochs: Optional[int] = None):
    epochs = int(epochs if epochs is not None else getattr(FLAGS, "epoch_num", 300))
    save_path = fold_expert_path(mode, fold_idx)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 已有权重则直接加载（断点续跑），避免 5×7×300 重复训
    if os.path.isfile(save_path):
        model = _build_expert(mode)
        state = torch.load(save_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        model.to(_device())
        saver.info(f"[label_gen] resume 已有权重: {save_path}")
        return model, save_path

    model = _build_expert(mode)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loader = DataLoader(
        train_list,
        batch_size=getattr(FLAGS, "batch_size", 32),
        shuffle=True,
        pin_memory=True,
        collate_fn=custom_collate,
    )
    best_loss, best_state = 1e18, None
    train_curve = []

    for epoch in range(epochs):
        model.train()
        total = 0.0
        n_graph = 0
        timer = OurTimer()
        for data in tqdm(loader, desc=f"[label] {mode} fold{fold_idx} ep{epoch+1}", leave=False):
            data = data.to(_device())
            data = inject_modalities(data)
            optimizer.zero_grad()
            _, loss, _ = _forward_expert(model, data, mode)
            if not torch.is_tensor(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item()) * data.num_graphs
            n_graph += data.num_graphs
        avg = total / max(n_graph, 1)
        train_curve.append(avg)
        saver.log_info(
            f"[label_gen] mode={mode} fold={fold_idx} epoch={epoch+1}/{epochs} "
            f"train_loss={avg:.6e} time={timer.time_and_clear()}"
        )
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)

        if len(train_curve) > 50 and len(set(round(x, 8) for x in train_curve[-50:])) == 1:
            saver.log_info(f"[label_gen] early stop mode={mode} fold={fold_idx} @ epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, save_path)
    saver.info(f"[label_gen] saved {save_path} best_loss={best_loss:.6e}")
    return model, save_path


@torch.no_grad()
def _eval_oof_errors(model, mode: str, val_list) -> List[dict]:
    """
    对留出样本逐条记录 |pred-true|（每目标），供选择器标签。
    同时返回聚合用的 true/pred 列表以便算 RMSE。
    """
    model.eval()
    loader = DataLoader(
        val_list,
        batch_size=getattr(FLAGS, "batch_size", 32),
        shuffle=False,
        collate_fn=custom_collate,
    )
    rows = []
    agg = {t: {"true": [], "pred": []} for t in TARGET_NAMES}

    for data in tqdm(loader, desc=f"[label] eval {mode}", leave=False):
        data = data.to(_device())
        data = inject_modalities(data)
        out_dict, _, _ = _forward_expert(model, data, mode)

        kernels = list(data.kernel) if isinstance(data.kernel, (list, tuple)) else list(data.kernel)
        files = list(data.file) if isinstance(data.file, (list, tuple)) else list(data.file)
        B = len(kernels)

        for i in range(B):
            row = {
                "kernel": str(kernels[i]),
                "file": str(files[i]),
                "mode": mode,
                "errors": {},
            }
            for t in TARGET_NAMES:
                y = _get_y_with_target(data, t)
                pred = out_dict[t]
                if y is None:
                    continue
                yt = float(y.view(-1)[i].item())
                pt = float(pred.view(-1)[i].item())
                err = abs(pt - yt)
                row["errors"][t] = err
                agg[t]["true"].append(yt)
                agg[t]["pred"].append(pt)
            rows.append(row)
    return rows, agg


def _rmse(true_list, pred_list) -> float:
    if not true_list:
        return float("nan")
    t = np.asarray(true_list, dtype=np.float64)
    p = np.asarray(pred_list, dtype=np.float64)
    return float(np.sqrt(np.mean((t - p) ** 2)))


def run_label_gen(dataset, epochs: Optional[int] = None):
    """
    完整阶段一：5×7 训练 → OOF 误差 → 写出 labels JSON/CSV。
    """
    if _LABEL_GEN_IMPORT_ERR is not None or ExpertEvalNet is None:
        raise ImportError(
            "label_gen 依赖 MPM 的 algorithm_select（ExpertEvalNet 等），"
            "本 HierarchicalMoE 迁移包未携带该模块；典型 MoE 请用 "
            "`python multimodal_moe/moe_system/moe_main.py --subtask train`。"
            f" 原始错误: {_LABEL_GEN_IMPORT_ERR}"
        )
    ensure_pipeline_dirs()
    train_pool, _, train_names, heldout_names = partition_dataset(dataset)
    saver.info(
        f"[label_gen] seed=64 train_kernels({len(train_names)})={train_names} "
        f"heldout({len(heldout_names)})={heldout_names} (heldout 本阶段不用)"
    )
    from utils_new import filter_all_zero_samples
    before = len(train_pool)
    train_pool = filter_all_zero_samples(train_pool, TARGET_NAMES, log_fn=saver.log_info)
    saver.info(f"[label_gen] del_0: {before} → {len(train_pool)}")

    folds = make_kernel_folds(train_names, n_folds=5)
    # sample_key -> {target -> [err_mode0..6]}
    sample_errors: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {t: [None] * 7 for t in TARGET_NAMES})
    sample_meta: Dict[str, dict] = {}
    fold_rmse_report = []

    for fold_idx, (tr_names, val_names) in enumerate(folds):
        tr_list = filter_by_kernels(train_pool, tr_names)
        val_list = filter_by_kernels(train_pool, val_names)
        saver.info(
            f"[label_gen] ===== Fold {fold_idx}: train={tr_names} ({len(tr_list)}) "
            f"val={val_names} ({len(val_list)}) ====="
        )
        if not tr_list or not val_list:
            raise RuntimeError(f"Fold {fold_idx} 训练集或验证集为空")

        for mode_id, mode in enumerate(SCHEME_NAMES):
            saver.info(f"[label_gen] Fold {fold_idx} 训练方案 {mode} ({mode_id+1}/7)")
            model, ckpt = _train_one_expert(mode, tr_list, fold_idx, epochs=epochs)
            rows, agg = _eval_oof_errors(model, mode, val_list)

            mode_rmse = {t: _rmse(agg[t]["true"], agg[t]["pred"]) for t in TARGET_NAMES}
            fold_rmse_report.append(
                {"fold": fold_idx, "mode": mode, "val_kernels": "|".join(val_names), **mode_rmse}
            )
            saver.info(f"[label_gen] Fold {fold_idx} {mode} OOF RMSE: {mode_rmse}")

            for r in rows:
                key = f"{r['kernel']}::{r['file']}"
                sample_meta[key] = {"kernel": r["kernel"], "file": r["file"], "fold": fold_idx}
                for t in TARGET_NAMES:
                    if t in r["errors"]:
                        sample_errors[key][t][mode_id] = float(r["errors"][t])

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ----- 写出标签：与 make_labels 兼容的 key + 便于 MoE 的 flat csv -----
    label_dict = {}
    csv_rows = []
    for key, meta in sample_meta.items():
        for m_idx, t in enumerate(TARGET_NAMES):
            err_vec = sample_errors[key][t]
            if any(v is None for v in err_vec):
                continue
            err_vec = [round(float(v), 6) for v in err_vec]
            legacy_key = str((meta["kernel"], meta["file"], m_idx))
            label_dict[legacy_key] = err_vec
            best = int(np.argmin(err_vec))
            csv_rows.append(
                {
                    "kernel": meta["kernel"],
                    "file": meta["file"],
                    "fold": meta["fold"],
                    "metric": t,
                    "metric_id": m_idx,
                    "best_scheme": SCHEME_NAMES[best],
                    "best_scheme_id": best,
                    **{f"err_{SCHEME_NAMES[i]}": err_vec[i] for i in range(7)},
                }
            )

    jp = labels_json_path()
    cp = labels_csv_path()
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(label_dict, f, indent=2, ensure_ascii=False)
    with open(cp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
        if csv_rows:
            w.writeheader()
            w.writerows(csv_rows)

    report_path = join(LABEL_DIR, "fold_oof_rmse_report.csv")
    if fold_rmse_report:
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fold_rmse_report[0].keys()))
            w.writeheader()
            w.writerows(fold_rmse_report)

    saver.info(
        f"[label_gen] 完成: samples≈{len(sample_meta)}, label_entries={len(label_dict)}\n"
        f"  json={jp}\n  csv={cp}\n  report={report_path}\n  experts={FOLD_EXPERT_DIR}"
    )
    return jp, cp
