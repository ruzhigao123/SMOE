"""
对协议中的 train_30 做 10 折交叉验证（单模态专家，供后续 MoE 使用）。

设定：
  - 30 个 kernel → 10 折，每折 hold-out 3 个未见 kernel，其余 27 个可训
  - 27 内再按约 8:2 划 train_kernel / val_kernel（选 best.pt 只用 val）
  - 每折 hold-out 的 3 个只做预测（打印 holdout RMSE，不参与选模）
  - 模态：cdfg_only / ast_only / LM（单 text）

为何不直接用 ablation_outputs/*/best.pt：
  那份权重在「几乎全部 30 个 kernel」上见过（24 训 + 6 验），若再拿去
  估未见 kernel / 训 MoE gate，专家已泄漏目标域信息，MoE 泛化会被高估。
  10 折专家保证：评估某 3 个未见 kernel 时，对应折的专家从未训过它们。

用法（在 HierarchicalMoE-master 下）:
  python -m multimodal_moe.src.cv10_single_modal --cv_subtask split_only
  # 默认：仅 3 单模态 × 10 折
  python -m multimodal_moe.src.cv10_single_modal --cv_subtask run_all
  # 双模态（2 特征融合）× 10 折
  python -m multimodal_moe.src.cv10_single_modal --cv_subtask run_all \
    --cv_modes cdfg_text ast_cdfg ast_text
  # 三模态 all × 10 折
  python -m multimodal_moe.src.cv10_single_modal --cv_subtask run_all --cv_modes all
  # 单模态 + 双融 + all（共 7 种）× 10 折
  python -m multimodal_moe.src.cv10_single_modal --cv_subtask run_all --cv_modes_preset all7
  # 某一模式补训全部折
  python -m multimodal_moe.src.cv10_single_modal --cv_subtask train_fold \
    --fold_id -1 --ablation_mode cdfg_text
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from os.path import abspath, dirname, join
from typing import Dict, List

import numpy as np
import torch
from numpy.random import RandomState
from torch.utils.data import DataLoader

# 注意：不要在解析本脚本参数前 import 会触发 config.FLAGS 的模块
_SRC = dirname(abspath(__file__))
_MM = dirname(_SRC)
_HMOE = dirname(_MM)
if _HMOE not in sys.path:
    sys.path.insert(0, _HMOE)

# 单模态（默认 run_all）
CV_MODES_SINGLE = ["cdfg_only", "ast_only", "LM"]
# 双模态融合（2 特征）
CV_MODES_PAIR = ["cdfg_text", "ast_cdfg", "ast_text"]
# 三模态融合（全部 3 塔；仓库无第 4 模态，勿与「4 个融合方案」混淆）
CV_MODES_TRIPLE = ["all"]
# 全部可训消融模式
CV_MODES_ALL = CV_MODES_SINGLE + CV_MODES_PAIR + CV_MODES_TRIPLE
# 兼容旧名
CV_MODES = list(CV_MODES_SINGLE)
N_FOLDS = 10
HOLD_OUT = 3


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def _infer_in_dim(data_list) -> int:
    return int(data_list[0].x.size(-1))


@torch.no_grad()
def evaluate(model, loader, device, target_list) -> Dict[str, float]:
    model.eval()
    preds = {t: [] for t in target_list}
    trues = {t: [] for t in target_list}
    for batch in loader:
        batch = batch.to(device)
        out, _, _ = model(batch)
        for t in target_list:
            if t not in out:
                continue
            y = getattr(batch, t.replace("-", "_"), None)
            if y is None:
                continue
            preds[t].append(out[t].detach().cpu().view(-1).numpy())
            trues[t].append(y.detach().cpu().view(-1).numpy())
    metrics = {}
    for t in target_list:
        if not preds[t]:
            continue
        p = np.concatenate(preds[t])
        g = np.concatenate(trues[t])
        metrics[f"rmse_{t}"] = _rmse(p, g)
    rmse_keys = [f"rmse_{t}" for t in target_list if f"rmse_{t}" in metrics]
    if rmse_keys:
        metrics["rmse_sum_targets"] = float(sum(metrics[k] for k in rmse_keys))
        metrics["rmse_mean_targets"] = float(np.mean([metrics[k] for k in rmse_keys]))
    return metrics


def _device(name: str) -> torch.device:
    d = torch.device(name if torch.cuda.is_available() or "cpu" in name else "cpu")
    if str(d).startswith("cuda") and not torch.cuda.is_available():
        d = torch.device("cpu")
    return d


def _format_metrics(prefix: str, metrics: Dict[str, float], targets: List[str]) -> str:
    parts = [f"{prefix}_rmse_sum5={metrics.get('rmse_sum_targets', float('nan')):.4f}"]
    for t in targets:
        k = f"rmse_{t}"
        if k in metrics:
            parts.append(f"{prefix}_{t}={metrics[k]:.4f}")
    return " ".join(parts)


def build_cv10_folds(
    train_30: List[str],
    seed: int = 50,
    n_folds: int = N_FOLDS,
    hold_out: int = HOLD_OUT,
    val_ratio: float = 0.2,
) -> List[dict]:
    assert len(train_30) == n_folds * hold_out, (
        f"需要 len(train_30)={n_folds * hold_out}，当前 {len(train_30)}"
    )
    rng = RandomState(seed)
    order = list(train_30)
    rng.shuffle(order)
    folds = []
    for i in range(n_folds):
        hold = order[i * hold_out : (i + 1) * hold_out]
        rest = [k for k in order if k not in set(hold)]
        n_val = max(1, int(round(len(rest) * val_ratio)))
        rng_f = RandomState(seed + 1000 + i)
        rest_shuffled = list(rest)
        rng_f.shuffle(rest_shuffled)
        val_k = rest_shuffled[:n_val]
        train_k = rest_shuffled[n_val:]
        folds.append(
            {
                "fold_id": i,
                "holdout_kernels": hold,
                "train_kernels": train_k,
                "val_kernels": val_k,
            }
        )
    return folds


def _lazy_imports():
    """延后导入，避免 config.FLAGS 抢解析 argv。"""
    from multimodal_moe.src.data_ablation import AblationDataset, build_dataset, collate_ablation
    from multimodal_moe.src.get_root_path import get_root_path
    from multimodal_moe.src.kernel_protocol import build_protocol
    from multimodal_moe.src.model_ablation import AblationNet
    from multimodal_moe.src.utils import create_dir_if_not_exists

    return {
        "AblationDataset": AblationDataset,
        "build_dataset": build_dataset,
        "collate_ablation": collate_ablation,
        "get_root_path": get_root_path,
        "build_protocol": build_protocol,
        "AblationNet": AblationNet,
        "create_dir_if_not_exists": create_dir_if_not_exists,
    }


def save_or_load_folds(args, I) -> List[dict]:
    I["create_dir_if_not_exists"](args.cv_save_dir)
    folds_path = join(args.cv_save_dir, "folds.json")
    proto = I["build_protocol"](
        seed=args.seed,
        save_json=join(args.cv_save_dir, "kernel_split_ref.json"),
    )
    train_30 = list(proto["train_30"])
    if os.path.isfile(folds_path) and not args.rebuild_folds:
        with open(folds_path, encoding="utf-8") as f:
            data = json.load(f)
        old_seed = data.get("seed")
        old_30 = set(data.get("train_30") or [])
        if old_seed is not None and int(old_seed) != int(args.seed):
            raise RuntimeError(
                f"已有 folds.json seed={old_seed}，当前 --seed={args.seed}。"
                f"请加 --rebuild_folds 重建，或删掉 {folds_path}"
            )
        if old_30 and old_30 != set(train_30):
            raise RuntimeError(
                f"已有 folds.json 的 train_30 与当前 protocol(seed={args.seed}) 不一致。"
                f"请加 --rebuild_folds 重建，或删掉 {folds_path}"
            )
        print(f"[cv10] loaded folds from {folds_path}", flush=True)
        return data["folds"]

    folds = build_cv10_folds(
        train_30,
        seed=args.seed,
        n_folds=args.cv_n_folds,
        hold_out=args.hold_out,
        val_ratio=args.val_ratio_kernels,
    )
    payload = {
        "seed": args.seed,
        "train_30": train_30,
        "n_folds": args.cv_n_folds,
        "hold_out": args.hold_out,
        "val_ratio_kernels": args.val_ratio_kernels,
        "folds": folds,
        "note": "holdout=未见3核; train/val 来自其余27核，选best只用val",
    }
    with open(folds_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[cv10] wrote {folds_path}", flush=True)
    for fd in folds:
        print(
            f"  fold={fd['fold_id']} holdout={fd['holdout_kernels']} "
            f"n_train={len(fd['train_kernels'])} n_val={len(fd['val_kernels'])}",
            flush=True,
        )
    return folds


def train_one_fold_mode(args, fold: dict, mode: str, I) -> dict:
    AblationDataset = I["AblationDataset"]
    build_dataset = I["build_dataset"]
    collate_ablation = I["collate_ablation"]
    AblationNet = I["AblationNet"]
    create_dir_if_not_exists = I["create_dir_if_not_exists"]

    device = _device(args.device)
    fold_id = int(fold["fold_id"])
    save_root = join(args.cv_save_dir, f"fold_{fold_id}", mode)
    create_dir_if_not_exists(save_root)
    best_path = join(save_root, "best.pt")
    history_path = join(save_root, "history.json")

    train_k = set(fold["train_kernels"])
    val_k = set(fold["val_kernels"])
    hold_k = set(fold["holdout_kernels"])
    all_needed = list(train_k | val_k | hold_k)

    from multimodal_moe.moe_system.utils_new import normalize_dataset_mode, set_dataset_mode

    ds_mode = normalize_dataset_mode(getattr(args, "dataset_mode", "filtered"))
    set_dataset_mode(ds_mode)
    skip_zero = getattr(args, "skip_all_zero_config", None)
    if skip_zero is None:
        skip_zero = ds_mode == "filtered"
    data_all = build_dataset(
        target_list=args.target,
        skip_all_zero=bool(skip_zero),
        kernels=all_needed,
        dataset_mode=ds_mode,
    )
    train_list = [d for d in data_all if getattr(d, "kernel", None) in train_k]
    val_list = [d for d in data_all if getattr(d, "kernel", None) in val_k]
    hold_list = [d for d in data_all if getattr(d, "kernel", None) in hold_k]

    print(
        f"[cv10/{mode}/fold{fold_id}] samples train/val/holdout="
        f"{len(train_list)}/{len(val_list)}/{len(hold_list)} "
        f"kernels {len(train_k)}/{len(val_k)}/{len(hold_k)} "
        f"holdout={fold['holdout_kernels']}",
        flush=True,
    )
    if len(train_list) == 0:
        raise RuntimeError(f"fold {fold_id} 训练集为空")

    in_dim = _infer_in_dim(train_list)
    model = AblationNet(
        in_cdfg=in_dim,
        target_list=args.target,
        ablation_mode=mode,
        D=args.D,
        hidden_num=args.hidden_num,
        gnn_layers=args.gnn_layers,
        dropout=args.dropout,
        out_dim=args.out_dim,
        gnn_type=getattr(args, "gnn_type", "gcn"),
        gnn_activation=getattr(args, "gnn_activation", "relu"),
    ).to(device)

    train_loader = DataLoader(
        AblationDataset(train_list),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_ablation,
    )
    val_loader = DataLoader(
        AblationDataset(val_list),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_ablation,
    )
    hold_loader = DataLoader(
        AblationDataset(hold_list),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_ablation,
    )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")
    start_epoch = 1
    history: List[dict] = []

    if args.resume and os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer") is not None:
            try:
                opt.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[cv10] WARNING optimizer not loaded: {e}", flush=True)
        best_val = float(ckpt.get("best_val_rmse_sum5", float("inf")))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if not np.isfinite(best_val):
            tmp = evaluate(model, val_loader, device, args.target)
            best_val = float(tmp.get("rmse_sum_targets", float("inf")))
        if os.path.isfile(history_path):
            try:
                with open(history_path, encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []
        print(
            f"[cv10/{mode}/fold{fold_id}] RESUME from epoch>{ckpt.get('epoch')} "
            f"best_val={best_val:.4f}",
            flush=True,
        )
    else:
        print(f"[cv10/{mode}/fold{fold_id}] train from scratch", flush=True)

    if start_epoch > args.epoch_num:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        hold_m = evaluate(model, hold_loader, device, args.target)
        out = {
            "fold_id": fold_id,
            "mode": mode,
            "holdout_kernels": fold["holdout_kernels"],
            "best_val_rmse_sum5": best_val,
            "holdout_metrics": hold_m,
            "checkpoint": best_path,
            "skipped_train": True,
        }
        with open(join(save_root, "result.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(
            f"[cv10/{mode}/fold{fold_id}] HOLDOUT {_format_metrics('hold', hold_m, args.target)}",
            flush=True,
        )
        return out

    for epoch in range(start_epoch, args.epoch_num + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            _, loss, _ = model(batch)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_m = evaluate(model, val_loader, device, args.target)
        hold_m = evaluate(model, hold_loader, device, args.target)
        val_score = val_m.get("rmse_sum_targets", 1e9)
        hold_score = hold_m.get("rmse_sum_targets", 1e9)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            **{f"val_{k}": v for k, v in val_m.items()},
            **{f"hold_{k}": v for k, v in hold_m.items()},
        }
        history.append(row)

        if epoch % args.print_every == 0 or epoch == start_epoch:
            print(
                f"[cv10/{mode}/fold{fold_id}] epoch={epoch} "
                f"train_loss={row['train_loss']:.4f} "
                f"val_rmse_sum5={val_score:.4f} "
                f"hold_rmse_sum5={hold_score:.4f} "
                + " ".join(
                    f"val_{t}={val_m[f'rmse_{t}']:.4f}"
                    for t in args.target
                    if f"rmse_{t}" in val_m
                )
                + " "
                + " ".join(
                    f"hold_{t}={hold_m[f'rmse_{t}']:.4f}"
                    for t in args.target
                    if f"rmse_{t}" in hold_m
                ),
                flush=True,
            )

        if val_score < best_val:
            prev = best_val
            best_val = val_score
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "epoch": epoch,
                    "best_val_rmse_sum5": best_val,
                    "ablation_mode": mode,
                    "fold_id": fold_id,
                    "in_dim": in_dim,
                    "target": list(args.target),
                    "D": int(args.D),
                    "hidden_num": int(args.hidden_num),
                    "gnn_layers": int(args.gnn_layers),
                    "dropout": float(args.dropout),
                    "gnn_type": str(getattr(args, "gnn_type", "gcn")),
                    "gnn_activation": str(getattr(args, "gnn_activation", "relu")),
                    "holdout_kernels": fold["holdout_kernels"],
                    "train_kernels": fold["train_kernels"],
                    "val_kernels": fold["val_kernels"],
                    "val_metrics": val_m,
                    "holdout_metrics": hold_m,
                },
                best_path,
            )
            print(
                f"[cv10/{mode}/fold{fold_id}] epoch={epoch} >>> 更新了 best.pt "
                f"(val_rmse_sum5: {prev:.4f} -> {best_val:.4f})",
                flush=True,
            )

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    hold_m = evaluate(model, hold_loader, device, args.target)
    out = {
        "fold_id": fold_id,
        "mode": mode,
        "holdout_kernels": fold["holdout_kernels"],
        "best_val_rmse_sum5": best_val,
        "best_epoch": ckpt.get("epoch"),
        "holdout_metrics": hold_m,
        "n_train": len(train_list),
        "n_val": len(val_list),
        "n_holdout": len(hold_list),
        "checkpoint": best_path,
    }
    with open(join(save_root, "result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(
        f"[cv10/{mode}/fold{fold_id}] DONE best_epoch={ckpt.get('epoch')} "
        f"best_val={best_val:.4f} | {_format_metrics('hold', hold_m, args.target)}",
        flush=True,
    )
    return out


def _select_folds(folds, start_fold: int, end_fold):
    out = []
    for f in folds:
        fid = int(f["fold_id"])
        if fid < int(start_fold):
            continue
        if end_fold is not None and fid > int(end_fold):
            continue
        out.append(f)
    return out


def run_all(args, I) -> List[dict]:
    folds = save_or_load_folds(args, I)
    start = int(getattr(args, "start_fold", 0))
    end = getattr(args, "end_fold", None)
    folds = _select_folds(folds, start, end)
    modes = args.cv_modes
    results = []
    print(f"[cv10] run_all modes={modes} folds={[f['fold_id'] for f in folds]}", flush=True)
    for mode in modes:
        for fold in folds:
            print("=" * 72, flush=True)
            results.append(train_one_fold_mode(args, fold, mode, I))
    summary_path = join(args.cv_save_dir, "summary_cv10.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    by_mode: Dict[str, List[float]] = {m: [] for m in modes}
    for r in results:
        m = r["mode"]
        s = r.get("holdout_metrics", {}).get("rmse_sum_targets")
        if s is not None:
            by_mode[m].append(float(s))
    print("=" * 72, flush=True)
    print("[cv10] holdout rmse_sum5 mean±std over folds:", flush=True)
    for m, xs in by_mode.items():
        if not xs:
            continue
        print(f"  {m}: {np.mean(xs):.4f} ± {np.std(xs):.4f} (n={len(xs)})", flush=True)
    print(f"[cv10] wrote {summary_path}", flush=True)
    return results


def _resolve_cv_modes(args) -> list:
    """
    优先级：非空 --cv_modes_preset > 显式 --cv_modes > 默认三单模态。
    fusion/fusion4 = 3 双融 + all（4 个融合方案）。
    """
    preset = str(getattr(args, "cv_modes_preset", "") or "").strip().lower()
    preset_map = {
        "single": CV_MODES_SINGLE,
        "pair": CV_MODES_PAIR,
        "fusion2": CV_MODES_PAIR,
        "triple": CV_MODES_TRIPLE,
        "fusion3": CV_MODES_TRIPLE,
        "fusion": CV_MODES_PAIR + CV_MODES_TRIPLE,
        "fusion4": CV_MODES_PAIR + CV_MODES_TRIPLE,
        "all7": CV_MODES_ALL,
        "all": CV_MODES_ALL,
        "full": CV_MODES_ALL,
    }
    if preset:
        if preset not in preset_map:
            raise ValueError(f"未知 cv_modes_preset={preset}")
        modes = list(preset_map[preset])
    else:
        modes = list(getattr(args, "cv_modes", None) or CV_MODES_SINGLE)
    bad = [m for m in modes if m not in CV_MODES_ALL]
    if bad:
        raise ValueError(f"未知 cv_modes={bad}，可选 {CV_MODES_ALL}")
    return modes


def parse_args():
    p = argparse.ArgumentParser(
        description="10-fold CV experts（单模态 / 双融 / 三融 all）"
    )
    p.add_argument(
        "--cv_subtask",
        default="run_all",
        choices=["split_only", "train_fold", "run_all"],
    )
    p.add_argument("--fold_id", type=int, default=-1, help="折号；-1 表示该模态跑完全部折")
    p.add_argument(
        "--start_fold",
        type=int,
        default=0,
        help="从该折开始（含）",
    )
    p.add_argument(
        "--end_fold",
        type=int,
        default=None,
        help="到该折结束（含）；默认到最后一折。补训 0-5 用 --end_fold 5",
    )
    p.add_argument(
        "--ablation_mode",
        type=str,
        default="cdfg_only",
        choices=CV_MODES_ALL,
        help="train_fold 时的单个模式",
    )
    p.add_argument(
        "--cv_modes",
        nargs="+",
        default=list(CV_MODES_SINGLE),
        help="run_all 要跑的模式列表，如 cdfg_text ast_cdfg all",
    )
    p.add_argument(
        "--cv_modes_preset",
        type=str,
        default="",
        choices=["", "single", "pair", "fusion2", "triple", "fusion3", "fusion", "fusion4", "all7", "all", "full"],
        help="快捷：single=三单模态；pair/fusion2=三双融；"
        "fusion/fusion4=三双融+all；all7=全部 7 种",
    )
    p.add_argument("--cv_n_folds", type=int, default=N_FOLDS)
    p.add_argument("--hold_out", type=int, default=HOLD_OUT)
    p.add_argument(
        "--seed",
        type=int,
        default=69,
        help="随机种子；cv_save_dir 追加 seed_{seed}（如 64..68）",
    )
    p.add_argument("--val_ratio_kernels", type=float, default=0.2)
    p.add_argument("--rebuild_folds", action="store_true")
    p.add_argument("--cv_save_dir", default=None, help="default: multimodal_moe/cv10_outputs")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epoch_num", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--D", type=int, default=128)
    p.add_argument("--hidden_num", type=int, default=128)
    p.add_argument("--out_dim", type=int, default=1)
    p.add_argument("--gnn_layers", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument(
        "--gnn_type",
        type=str,
        default="gcn",
        choices=["transformer", "gcn"],
    )
    p.add_argument(
        "--gnn_activation",
        type=str,
        default="relu",
        choices=["relu", "elu"],
    )
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--target",
        nargs="+",
        default=["perf", "util-LUT", "util-FF", "util-DSP", "util-BRAM"],
    )
    p.add_argument(
        "--dataset_mode",
        type=str,
        default="filtered",
        choices=["filtered", "not_0", "del_0", "keep_zero"],
        help="filtered=two_tower_datasets*；not_0=*_not_0",
    )
    p.add_argument(
        "--skip_all_zero_config",
        type=bool,
        default=None,
        help="默认 filtered=True、not_0=False",
    )
    p.add_argument("--print_every", type=int, default=1)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no_resume", action="store_false", dest="resume")
    return p.parse_args()


def main():
    args = parse_args()
    # 清空 argv，避免后续误 import config 时再次 parse 报错
    sys.argv = [sys.argv[0]]
    args.cv_modes = _resolve_cv_modes(args)
    I = _lazy_imports()
    from multimodal_moe.moe_system.utils_new import (
        default_output_dirs,
        ensure_seed_outdir,
        normalize_dataset_mode,
        set_dataset_mode,
    )

    ds = normalize_dataset_mode(getattr(args, "dataset_mode", "filtered"))
    set_dataset_mode(ds)
    args.dataset_mode = ds
    if args.cv_save_dir is None:
        args.cv_save_dir = default_output_dirs(ds, seed=int(args.seed))["cv10_root"]
    else:
        args.cv_save_dir = ensure_seed_outdir(args.cv_save_dir, int(args.seed))
    I["create_dir_if_not_exists"](args.cv_save_dir)
    print(
        f"[cv10] dataset_mode={getattr(args, 'dataset_mode', '?')} "
        f"seed={args.seed} cv_save_dir={args.cv_save_dir}\n"
        f"[cv10] modes={args.cv_modes}",
        flush=True,
    )

    if args.cv_subtask == "split_only":
        save_or_load_folds(args, I)
        return
    folds = save_or_load_folds(args, I)
    if args.cv_subtask == "train_fold":
        start = int(getattr(args, "start_fold", 0))
        end = getattr(args, "end_fold", None)
        if int(args.fold_id) < 0:
            for fold in _select_folds(folds, start, end):
                print("=" * 72, flush=True)
                train_one_fold_mode(args, fold, args.ablation_mode, I)
            return
        fold = next(f for f in folds if int(f["fold_id"]) == int(args.fold_id))
        train_one_fold_mode(args, fold, args.ablation_mode, I)
        return
    if args.cv_subtask == "run_all":
        run_all(args, I)
        return
    raise ValueError(args.cv_subtask)


if __name__ == "__main__":
    main()
