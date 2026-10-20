"""
典型 Hierarchical MoE 训练 / 推理。

- 15 kernel 按 kernel 7:3 训练（MSE 回归），best 按 val 上 **perf（latency）单指标 RMSE**
- 7 个未见 kernel 上直接用 MoE 预测（不加载 model_select_weight）
- 产物：moe_pipeline_outputs/hierarchical_moe/
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from os.path import join
from typing import Optional

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

from model_moe import HierarchicalModalMoE
from pipeline_paths import (
    TARGET_NAMES,
    ensure_pipeline_dirs,
    moe_best_path,
    moe_final_path,
    MOE_DIR,
    SELECTOR_TRAIN_RATIO,
)
from kernel_split import partition_dataset, filter_by_kernels, split_kernels_7_3
from modal_io import custom_collate, inject_modalities
from utils_new import filter_all_zero_samples


def _device():
    return torch.device(FLAGS.device)


def _target_list():
    return list(TARGET_NAMES)


def _build_moe(deg=None, epochs: Optional[int] = None) -> HierarchicalModalMoE:
    epochs = int(epochs if epochs is not None else getattr(FLAGS, "epoch_num", 300))
    # 对齐 AAAI'25 HierarchicalMoE：α=5e-3，warmup=epochs//2，交替 separate/joint
    model = HierarchicalModalMoE(
        cdfg_in_channels=153,
        ast_in_channels=784,
        target_list=_target_list(),
        deg=deg,
        expert_dim=256,
        gnn_out_dim=128,
        lambda_balance=getattr(FLAGS, "moe_lmbda", getattr(FLAGS, "moe_lambda_balance", 5e-3)),
        soft_epochs=getattr(FLAGS, "moe_soft_epochs", 20),
        expert_depth=getattr(FLAGS, "moe_expert_depth", 2),
        soft_mix_alpha=getattr(FLAGS, "moe_soft_mix_alpha", 0.5),
        hierarchical_moe_epoch=getattr(FLAGS, "hierarchical_moe_epoch", epochs // 2),
        hierarchical_alternate_train=getattr(FLAGS, "hierarchical_alternate_train", True),
    ).to(_device())
    return model


def _rmse_dict(points):
    out = {}
    for t, d in points.items():
        if not d["true"]:
            out[t] = float("nan")
            continue
        yt = np.asarray(d["true"], dtype=np.float64)
        yp = np.asarray(d["pred"], dtype=np.float64)
        out[t] = float(np.sqrt(np.mean((yt - yp) ** 2)))
    return out


@torch.no_grad()
def _eval_rmse(model, loader, epoch=None):
    """
    返回 (avg_loss, perf_rmse, rmse_dict, gate_mean)。
    gate_mean[task] = {CDFG, Text, AST} 在 loader 上的平均路由权重。
    选 best 只用 latency（perf）单个 RMSE。
    """
    model.eval()
    points = {t: {"true": [], "pred": []} for t in _target_list()}
    gate_sum = {t: np.zeros(3, dtype=np.float64) for t in _target_list()}
    gate_n = {t: 0 for t in _target_list()}
    loss_sum, n_g = 0.0, 0
    for data in loader:
        data = data.to(_device())
        data = inject_modalities(data)
        out_dict, total_l, _, _, gate_records = model(data, epoch=epoch)
        loss_sum += float(total_l.item()) * data.num_graphs
        n_g += data.num_graphs
        for t in _target_list():
            y = getattr(data, t.replace("-", "_"), None)
            if y is None or t not in out_dict:
                continue
            pred = out_dict[t].view(-1)
            yt = y.view(-1)
            points[t]["true"].extend([float(v) for v in yt.detach().cpu()])
            points[t]["pred"].extend([float(v) for v in pred.detach().cpu()])
            if t in gate_records and "weights" in gate_records[t]:
                w = gate_records[t]["weights"].detach().float().cpu().numpy()  # [B,3]
                gate_sum[t] += w.sum(axis=0)
                gate_n[t] += w.shape[0]
    rmse = _rmse_dict(points)
    perf_rmse = float(rmse.get("perf", float("nan")))
    avg_loss = loss_sum / max(n_g, 1)
    gate_mean = {}
    for t in _target_list():
        if gate_n[t] > 0:
            m = gate_sum[t] / gate_n[t]
            gate_mean[t] = {
                "CDFG": float(m[0]),
                "Text": float(m[1]),
                "AST": float(m[2]),
            }
    return avg_loss, perf_rmse, rmse, gate_mean


def _fmt_gate(gate_mean: dict) -> str:
    """紧凑打印各指标三专家权重。"""
    if not gate_mean:
        return "{}"
    parts = []
    for t, g in gate_mean.items():
        parts.append(
            f"{t}:[CDFG={g['CDFG']:.3f},Text={g['Text']:.3f},AST={g['AST']:.3f}]"
        )
    return " ".join(parts)


def train_moe(dataset, epochs: Optional[int] = None):
    """端到端 Hierarchical MoE 回归训练。"""
    ensure_pipeline_dirs()
    epochs = int(epochs if epochs is not None else getattr(FLAGS, "epoch_num", 300))

    train_pool, heldout_raw, train_names, heldout_names = partition_dataset(dataset)
    train_pool = filter_all_zero_samples(train_pool, TARGET_NAMES, log_fn=saver.log_info)
    heldout = filter_all_zero_samples(heldout_raw, TARGET_NAMES, log_fn=saver.log_info)

    sel_tr_kernels, sel_va_kernels = split_kernels_7_3(
        train_names, ratio=SELECTOR_TRAIN_RATIO, seed=64
    )
    tr_list = filter_by_kernels(train_pool, sel_tr_kernels)
    va_list = filter_by_kernels(train_pool, sel_va_kernels)
    if not tr_list or not va_list:
        raise RuntimeError("kernel 7:3 后 train/val 为空")

    saver.info(
        f"[moe] kernel 7:3 train={sel_tr_kernels} n={len(tr_list)} | "
        f"val={sel_va_kernels} n={len(va_list)}"
    )
    saver.info(f"[moe] heldout 监控 kernels={heldout_names} n={len(heldout)}")
    saver.info(f"[moe] 产物目录={MOE_DIR}（不加载 model_select_weight）")
    warmup = int(getattr(FLAGS, "hierarchical_moe_epoch", epochs // 2))
    saver.info(
        f"[moe] 防极化(论文): soft-softmax + CV(importance) λ="
        f"{getattr(FLAGS, 'moe_lmbda', getattr(FLAGS, 'moe_lambda_balance', 5e-3))} | "
        f"two-stage warmup={warmup}/{epochs} alternate="
        f"{getattr(FLAGS, 'hierarchical_alternate_train', True)} | "
        f"GNN=TransformerConv (HARP/HierarchicalMoE)"
    )

    bs = getattr(FLAGS, "batch_size", 32)
    tr_loader = DataLoader(tr_list, batch_size=bs, shuffle=True, collate_fn=custom_collate)
    va_loader = DataLoader(va_list, batch_size=bs, shuffle=False, collate_fn=custom_collate)
    heldout_loader = (
        DataLoader(heldout, batch_size=min(16, bs), shuffle=False, collate_fn=custom_collate)
        if heldout
        else None
    )

    model = _build_moe(epochs=epochs)
    best_rmse, best_state = 1e18, None
    ckpt = moe_best_path()
    if os.path.isfile(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        _, best_rmse, detail, gate_v = _eval_rmse(model, va_loader)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        saver.info(
            f"[moe] resume {ckpt} val_rmse(perf)={best_rmse:.6f} detail={detail}"
        )
        saver.info(f"[moe] resume gate_weights(val): {_fmt_gate(gate_v)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    hist = []

    for epoch in range(epochs):
        model.train()
        loss_sum, n_g = 0.0, 0
        for data in tqdm(tr_loader, desc=f"[moe] train ep{epoch+1}", leave=False):
            data = data.to(_device())
            data = inject_modalities(data)
            optimizer.zero_grad()
            _, total_l, _, _, _ = model(data, epoch=epoch)
            total_l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(total_l.item()) * data.num_graphs
            n_g += data.num_graphs
        tr_loss = loss_sum / max(n_g, 1)

        _, val_rmse, val_detail, gate_val = _eval_rmse(model, va_loader, epoch=epoch)
        ho_rmse, ho_detail, gate_ho = float("nan"), {}, {}
        if heldout_loader is not None:
            _, ho_rmse, ho_detail, gate_ho = _eval_rmse(
                model, heldout_loader, epoch=epoch
            )

        hist.append(
            {
                "epoch": epoch + 1,
                "train_loss": tr_loss,
                "val_rmse_perf": val_rmse,
                "heldout_rmse_perf": ho_rmse,
                **{f"val_{k}": v for k, v in val_detail.items()},
                **{f"heldout_{k}": v for k, v in ho_detail.items()},
                **{
                    f"gate_val_{t}_{name}": gate_val[t][name]
                    for t in gate_val
                    for name in ("CDFG", "Text", "AST")
                },
            }
        )
        stage = (
            "separate"
            if (
                epoch < getattr(model, "hierarchical_moe_epoch", epochs // 2)
                or (
                    getattr(model, "hierarchical_alternate_train", True)
                    and epoch % 2 == 1
                    and epoch >= getattr(model, "hierarchical_moe_epoch", epochs // 2)
                )
            )
            else "joint"
        )
        saver.log_info(
            f"[moe] Epoch {epoch+1}/{epochs} stage={stage} loss={tr_loss:.4f} "
            f"val_rmse(perf)={val_rmse:.6f} detail={val_detail} "
            f"heldout_rmse(perf)={ho_rmse} heldout={ho_detail}"
        )
        saver.log_info(f"[moe] gate_weights(val): {_fmt_gate(gate_val)}")
        if gate_ho:
            saver.log_info(f"[moe] gate_weights(heldout): {_fmt_gate(gate_ho)}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, moe_best_path())
            saver.info(
                f"[moe] 新最佳 val_rmse(perf)={best_rmse:.6f} -> {moe_best_path()}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), moe_final_path())

    hist_path = join(MOE_DIR, "moe_train_history.csv")
    write_header = not os.path.isfile(hist_path) or os.path.getsize(hist_path) == 0
    with open(hist_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()) if hist else [])
        if hist and write_header:
            w.writeheader()
        if hist:
            w.writerows(hist)
    saver.info(f"[moe] 训练结束 best_val_rmse(perf)={best_rmse:.6f} hist={hist_path}")
    return moe_best_path(), best_rmse


@torch.no_grad()
def inference_heldout(dataset):
    """7 未见 kernel：Hierarchical MoE 直接回归预测。"""
    ensure_pipeline_dirs()
    ckpt = moe_best_path()
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"MoE 权重不存在: {ckpt}，请先 train")

    _, heldout, train_names, heldout_names = partition_dataset(dataset)
    heldout = filter_all_zero_samples(heldout, TARGET_NAMES, log_fn=saver.log_info)
    saver.info(
        f"[infer] heldout={heldout_names} n={len(heldout)} (train kernels={train_names})"
    )

    model = _build_moe()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    model.eval()

    loader = DataLoader(
        heldout,
        batch_size=getattr(FLAGS, "batch_size", 16),
        shuffle=False,
        collate_fn=custom_collate,
    )
    rows = []
    points = {t: {"true": [], "pred": []} for t in TARGET_NAMES}
    gate_sum = {t: np.zeros(3, dtype=np.float64) for t in TARGET_NAMES}
    gate_n = {t: 0 for t in TARGET_NAMES}

    for data in tqdm(loader, desc="[infer] moe heldout"):
        data = data.to(_device())
        data = inject_modalities(data)
        out_dict, _, _, _, gate_records = model(data, epoch=None)
        kernels = list(data.kernel)
        files = list(data.file)
        B = len(kernels)
        for i in range(B):
            row = {"kernel": str(kernels[i]), "file": str(files[i])}
            for t in TARGET_NAMES:
                y = getattr(data, t.replace("-", "_"), None)
                if y is None or t not in out_dict:
                    continue
                pred = float(out_dict[t].view(-1)[i].item())
                true_v = float(y.view(-1)[i].item())
                row[f"{t}_pred"] = pred
                row[f"{t}_true"] = true_v
                row[f"{t}_abs_err"] = abs(pred - true_v)
                if t in gate_records and "weights" in gate_records[t]:
                    w = gate_records[t]["weights"][i].detach().float().cpu().numpy()
                    row[f"{t}_gate_cdfg"] = float(w[0])
                    row[f"{t}_gate_text"] = float(w[1])
                    row[f"{t}_gate_ast"] = float(w[2])
                    gate_sum[t] += w
                    gate_n[t] += 1
                points[t]["true"].append(true_v)
                points[t]["pred"].append(pred)
            rows.append(row)

    out_csv = join(MOE_DIR, "heldout_inference.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    gate_mean = {}
    for t in TARGET_NAMES:
        if gate_n[t] > 0:
            m = gate_sum[t] / gate_n[t]
            gate_mean[t] = {
                "CDFG": float(m[0]),
                "Text": float(m[1]),
                "AST": float(m[2]),
            }
    saver.info(f"[infer] gate_weights(heldout mean): {_fmt_gate(gate_mean)}")

    summary = []
    rmse = _rmse_dict(points)
    for t in TARGET_NAMES:
        entry = {"metric": t, "rmse": rmse[t]}
        if t in gate_mean:
            entry["gate"] = gate_mean[t]
        summary.append(entry)
        saver.info(f"[infer] {t} RMSE={rmse[t]:.6f}")
        if t in gate_mean:
            g = gate_mean[t]
            saver.info(
                f"[infer] {t} gate CDFG={g['CDFG']:.3f} Text={g['Text']:.3f} AST={g['AST']:.3f}"
            )
    with open(join(MOE_DIR, "heldout_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    saver.info(f"[infer] 写出 {out_csv}")
    return out_csv


# 兼容旧入口名
def train_selector(dataset, epochs=None):
    return train_moe(dataset, epochs=epochs)


def train_main(dataset):
    return train_moe(dataset)


def inference(dataset):
    return inference_heldout(dataset)
