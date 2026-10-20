"""
单方案训练 / 测试评估。

对齐 MPM ``train_del_0.py`` 两套语义：
  (1) 无融合 train（1400 行起）：只跑图编码器 → MLP
  (2) 融合 train（1457 行注释版）：batch 注入 text_feat/ast_feat，
      ``model(..., ablation_mode=...)``；缺失模态在 fusion 前置零

本实现：
  - 单模态 ≈ (1)
  - 双/三模态 ≈ (2)：``collate_ablation`` 预先 stack text/ast（等价于注释版
    在循环里加载并 stack），再在 ``AblationNet`` 内置零 + TripleModalFusion
"""
from __future__ import annotations

import json
import os
from os.path import join
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import FLAGS
from .config_split import (
    filter_by_kernels,
    filter_by_keys,
    load_config_split,
    save_config_split,
    split_pool_train_val,
)
from .data_ablation import AblationDataset, build_dataset, collate_ablation
from .kernel_protocol import build_protocol
from .model_ablation import AblationNet
from .utils import create_dir_if_not_exists


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def _infer_in_dim(data_list) -> int:
    x = data_list[0].x
    return int(x.size(-1))


def _prepare_splits(proto: dict = None):
    """
    默认（最初方案）: kernel 30/10 + 24/6。
    可选 split_by=config: 30 核配置池内随机 8:2，test 仍为 test_10。
    """
    create_dir_if_not_exists(FLAGS.save_dir)
    split_by = str(getattr(FLAGS, "split_by", "kernel"))

    if proto is not None and proto.get("train_30") and proto.get("test_10"):
        kproto = proto
    else:
        kproto = build_protocol(
            seed=FLAGS.seed,
            save_json=join(FLAGS.save_dir, "kernel_split.json"),
        )

    train_30 = list(kproto["train_30"])
    test_10 = list(kproto["test_10"])
    all_needed = list(dict.fromkeys(train_30 + test_10))
    from multimodal_moe.moe_system.utils_new import normalize_dataset_mode, set_dataset_mode

    ds_mode = normalize_dataset_mode(getattr(FLAGS, "dataset_mode", "filtered"))
    set_dataset_mode(ds_mode)
    skip_zero = getattr(FLAGS, "skip_all_zero_config", None)
    if skip_zero is None:
        skip_zero = ds_mode == "filtered"
    data_all = build_dataset(
        target_list=FLAGS.target,
        skip_all_zero=bool(skip_zero),
        kernels=all_needed,
        dataset_mode=ds_mode,
    )
    test_list = filter_by_kernels(data_all, test_10)

    # ---- 最初方案：24/6 核 ----
    if split_by != "config":
        train_24 = list(kproto["train_24"])
        val_6 = list(kproto["val_6"])
        train_list = filter_by_kernels(data_all, train_24)
        val_list = filter_by_kernels(data_all, val_6)
        split_info = {
            "split_by": "kernel",
            "seed": FLAGS.seed,
            "train_30": train_30,
            "train_24": train_24,
            "val_6": val_6,
            "test_10": test_10,
            "dropped_2": kproto.get("dropped_2", []),
            "n_train": len(train_list),
            "n_val": len(val_list),
            "n_test": len(test_list),
        }
        print(
            f"[split] 最初方案 kernel 24/6/10 "
            f"samples={len(train_list)}/{len(val_list)}/{len(test_list)}",
            flush=True,
        )
        return train_list, val_list, test_list, split_info

    # ---- 可选：30 核配置随机 8:2 ----
    train_val_pool = filter_by_kernels(data_all, train_30)
    split_path = join(FLAGS.save_dir, "config_split.json")
    train_ratio = float(getattr(FLAGS, "train_ratio", 0.8))

    reuse = None
    if proto is not None and proto.get("split_by") == "config" and proto.get("train_keys"):
        reuse = proto
    elif os.path.isfile(split_path):
        cand = load_config_split(split_path)
        if (
            cand.get("split_by") == "config"
            and cand.get("train_30") == train_30
            and cand.get("test_10") == test_10
            and cand.get("train_keys")
        ):
            reuse = cand

    if reuse is not None:
        train_list = filter_by_keys(train_val_pool, reuse["train_keys"])
        val_list = filter_by_keys(train_val_pool, reuse["val_keys"])
        if len(train_list) + len(val_list) != len(train_val_pool):
            reuse = None

    if reuse is None:
        tv = split_pool_train_val(
            train_val_pool, seed=FLAGS.seed, train_ratio=train_ratio
        )
        train_list, val_list = tv["train_list"], tv["val_list"]
        split_info = {
            "split_by": "config",
            "seed": FLAGS.seed,
            "train_ratio": train_ratio,
            "train_30": train_30,
            "test_10": test_10,
            "dropped_2": kproto.get("dropped_2", []),
            "train_keys": tv["train_keys"],
            "val_keys": tv["val_keys"],
            "n_train": len(train_list),
            "n_val": len(val_list),
            "n_test": len(test_list),
            "n_train_val_pool": len(train_val_pool),
        }
        save_config_split(split_info, split_path)
    else:
        split_info = dict(reuse)
        split_info["n_train"] = len(train_list)
        split_info["n_val"] = len(val_list)
        split_info["n_test"] = len(test_list)

    print(
        f"[split] kernel30/10 + config8:2 "
        f"samples={len(train_list)}/{len(val_list)}/{len(test_list)}",
        flush=True,
    )
    return train_list, val_list, test_list, split_info


def _mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


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
        metrics[f"mse_{t}"] = _mse(p, g)
    rmse_keys = [f"rmse_{t}" for t in target_list if f"rmse_{t}" in metrics]
    mse_keys = [f"mse_{t}" for t in target_list if f"mse_{t}" in metrics]
    if rmse_keys:
        metrics["rmse_sum_targets"] = float(sum(metrics[k] for k in rmse_keys))
        metrics["rmse_mean_targets"] = float(np.mean([metrics[k] for k in rmse_keys]))
    if mse_keys:
        metrics["mse_sum_targets"] = float(sum(metrics[k] for k in mse_keys))
        metrics["mse_mean_targets"] = float(np.mean([metrics[k] for k in mse_keys]))
    return metrics


def _graph_embed_feats(data_list: list) -> np.ndarray:
    """无门控时：用 CDFG 节点特征均值做 coreset 向量。"""
    rows = []
    for d in data_list:
        x = d.x.float()
        if x.dim() == 2:
            rows.append(x.mean(dim=0).cpu().numpy())
        else:
            rows.append(x.view(-1).cpu().numpy())
    # pad / truncate to same dim
    dim = max(r.shape[0] for r in rows)
    out = np.zeros((len(rows), dim), dtype=np.float32)
    for i, r in enumerate(rows):
        n = min(dim, r.shape[0])
        out[i, :n] = r[:n]
    return out


def _kmeans_indices(feats: np.ndarray, k: int, seed: int) -> List[int]:
    n = feats.shape[0]
    k = max(1, min(int(k), n))
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, random_state=int(seed), n_init=10)
        labels = km.fit_predict(feats)
        centers = km.cluster_centers_
        chosen = []
        for c in range(k):
            idxs = np.where(labels == c)[0]
            if len(idxs) == 0:
                continue
            dist = np.linalg.norm(feats[idxs] - centers[c], axis=1)
            chosen.append(int(idxs[int(np.argmin(dist))]))
        # 不足 k 时补随机
        if len(chosen) < k:
            rest = [i for i in range(n) if i not in set(chosen)]
            rng = np.random.RandomState(int(seed))
            need = k - len(chosen)
            if rest:
                extra = rng.choice(rest, size=min(need, len(rest)), replace=False).tolist()
                chosen.extend(int(i) for i in extra)
        return chosen[:k]
    except Exception as e:
        print(f"[finetune] kmeans 失败，回退 random: {e}", flush=True)
        rng = np.random.RandomState(int(seed))
        return rng.choice(n, size=k, replace=False).tolist()


def _split_kshot_by_kernel(
    data_list: list,
    k: int,
    coreset: str,
    seed: int,
) -> Tuple[list, list, dict]:
    from collections import defaultdict

    by_k = defaultdict(list)
    for d in data_list:
        by_k[str(getattr(d, "kernel", ""))].append(d)
    train_sel, query, stats = [], [], {}
    rng = np.random.RandomState(int(seed))
    coreset = str(coreset).lower()
    for kernel in sorted(by_k.keys()):
        items = by_k[kernel]
        n = len(items)
        k_use = int(k)
        if k_use >= n:
            k_use = max(n - 1, 0)
        if k_use <= 0:
            query.extend(items)
            stats[kernel] = {"n": n, "k": 0, "method": coreset}
            continue
        if coreset == "kmeans":
            feats = _graph_embed_feats(items)
            idxs = _kmeans_indices(feats, k_use, seed + hash(kernel) % 10007)
        else:
            idxs = rng.choice(n, size=k_use, replace=False).tolist()
        sel = set(int(i) for i in idxs)
        for i, d in enumerate(items):
            (train_sel if i in sel else query).append(d)
        stats[kernel] = {"n": n, "k": len(sel), "method": coreset}
    return train_sel, query, stats


def _build_model_from_ckpt(ckpt: dict, mode: str, in_dim: int, target_list, device):
    # 恢复对比开关，保证 CDFGEncoder 建对骨干
    if "comparative_if" in ckpt:
        FLAGS.comparative_if = bool(ckpt["comparative_if"])
    if "comparative_model" in ckpt:
        FLAGS.comparative_model = str(ckpt["comparative_model"])
    model = AblationNet(
        in_cdfg=in_dim,
        target_list=target_list,
        ablation_mode=mode,
        D=int(ckpt.get("D", FLAGS.D)),
        hidden_num=int(ckpt.get("hidden_num", FLAGS.hidden_num)),
        gnn_layers=int(ckpt.get("gnn_layers", FLAGS.gnn_layers)),
        dropout=float(ckpt.get("dropout", FLAGS.dropout)),
        out_dim=FLAGS.out_dim,
        gnn_type=str(ckpt.get("gnn_type", getattr(FLAGS, "gnn_type", "gcn"))),
        gnn_activation=str(
            ckpt.get("gnn_activation", getattr(FLAGS, "gnn_activation", "relu"))
        ),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model


def train_one_mode(ablation_mode: str = None, proto: dict = None) -> dict:
    mode = ablation_mode or FLAGS.ablation_mode
    device = _resolve_device()

    save_root = join(FLAGS.save_dir, mode)
    create_dir_if_not_exists(save_root)
    best_path = join(save_root, "best.pt")
    last_path = join(save_root, "last.pt")  # 每轮覆盖：中断后续训用
    history_path = join(save_root, "history.json")

    train_list, val_list, test_list, split_info = _prepare_splits(proto=proto)

    print(
        f"[{mode}] samples train/val/test = "
        f"{len(train_list)}/{len(val_list)}/{len(test_list)} "
        f"split_by={split_info.get('split_by')}",
        flush=True,
    )
    if len(train_list) == 0:
        raise RuntimeError("训练集为空：请确认 two_tower_datasets 与 AST 已生成并对齐")

    in_dim = _infer_in_dim(train_list)
    model = AblationNet(
        in_cdfg=in_dim,
        target_list=FLAGS.target,
        ablation_mode=mode,
        D=FLAGS.D,
        hidden_num=FLAGS.hidden_num,
        gnn_layers=FLAGS.gnn_layers,
        dropout=FLAGS.dropout,
        out_dim=FLAGS.out_dim,
        gnn_type=getattr(FLAGS, "gnn_type", "gcn"),
        gnn_activation=getattr(FLAGS, "gnn_activation", "relu"),
    ).to(device)

    train_loader = DataLoader(
        AblationDataset(train_list),
        batch_size=FLAGS.batch_size,
        shuffle=True,
        collate_fn=collate_ablation,
    )
    val_loader = DataLoader(
        AblationDataset(val_list),
        batch_size=FLAGS.batch_size,
        shuffle=False,
        collate_fn=collate_ablation,
    )
    test_loader = DataLoader(
        AblationDataset(test_list),
        batch_size=FLAGS.batch_size,
        shuffle=False,
        collate_fn=collate_ablation,
    )

    opt = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    # 选模指标：val 上 5 个目标 RMSE 之和；test 仅打印不参与选模
    best_val = float("inf")
    start_epoch = 1
    history = []
    bad_epochs = 0

    def _ckpt_payload(epoch: int, val_m: dict, test_m: dict, *, is_best: bool) -> dict:
        return {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "best_val_rmse_sum5": best_val,
            "ablation_mode": mode,
            "in_dim": in_dim,
            "target": list(FLAGS.target),
            "D": int(FLAGS.D),
            "hidden_num": int(FLAGS.hidden_num),
            "gnn_layers": int(FLAGS.gnn_layers),
            "dropout": float(FLAGS.dropout),
            "gnn_type": str(getattr(FLAGS, "gnn_type", "gcn")),
            "gnn_activation": str(getattr(FLAGS, "gnn_activation", "relu")),
            "comparative_if": bool(getattr(FLAGS, "comparative_if", False)),
            "comparative_model": str(getattr(FLAGS, "comparative_model", "ironman")),
            "val_metrics": val_m,
            "test_metrics": test_m,
            "proto": split_info,
            "ckpt_kind": "best" if is_best else "last",
        }

    # 续训优先 last.pt（当前进度）；best.pt 只提供 best_val 历史最优
    resume_path = None
    if getattr(FLAGS, "resume", True):
        if os.path.isfile(last_path):
            resume_path = last_path
        elif os.path.isfile(best_path):
            resume_path = best_path
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device)
        if int(ckpt.get("in_dim", in_dim)) != in_dim:
            raise RuntimeError(
                f"{os.path.basename(resume_path)} in_dim={ckpt.get('in_dim')} "
                f"与当前数据 in_dim={in_dim} 不一致"
            )
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer") is not None:
            try:
                opt.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[{mode}] WARNING: optimizer 状态未加载 ({e})，仅恢复模型权重", flush=True)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        # best_val：优先独立 best.pt，否则用 resume ckpt 内字段
        if os.path.isfile(best_path):
            try:
                best_ckpt = torch.load(best_path, map_location="cpu")
                best_val = float(
                    best_ckpt.get(
                        "best_val_rmse_sum5",
                        best_ckpt.get("best_test_rmse_sum5", float("inf")),
                    )
                )
            except Exception:
                best_val = float(
                    ckpt.get(
                        "best_val_rmse_sum5",
                        ckpt.get("best_test_rmse_sum5", float("inf")),
                    )
                )
        else:
            best_val = float(
                ckpt.get(
                    "best_val_rmse_sum5",
                    ckpt.get("best_test_rmse_sum5", float("inf")),
                )
            )
        if not np.isfinite(best_val):
            model.eval()
            tmp = evaluate(model, val_loader, device, FLAGS.target)
            best_val = float(tmp.get("rmse_sum_targets", float("inf")))
            print(
                f"[{mode}] old checkpoint: init best_val_rmse_sum5={best_val:.4f} from current weights",
                flush=True,
            )
        if os.path.isfile(history_path):
            try:
                with open(history_path, encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []
        print(
            f"[{mode}] RESUME from {resume_path} "
            f"(last_epoch={ckpt.get('epoch', '?')}, best_val_rmse_sum5={best_val:.4f}) "
            f"-> continue epoch {start_epoch}..{FLAGS.epoch_num}",
            flush=True,
        )
    else:
        print(f"[{mode}] train from scratch -> epochs 1..{FLAGS.epoch_num}", flush=True)

    if start_epoch > FLAGS.epoch_num:
        print(
            f"[{mode}] already finished epoch_num={FLAGS.epoch_num}; "
            f"raise --epoch_num to continue, or delete last.pt/best.pt / --no_resume to retrain",
            flush=True,
        )
        load_path = best_path if os.path.isfile(best_path) else last_path
        ckpt = torch.load(load_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        test_m = evaluate(model, test_loader, device, FLAGS.target)
        result = {
            "ablation_mode": mode,
            "best_val_rmse_sum5": best_val,
            "test_metrics": test_m,
            "n_train": len(train_list),
            "n_val": len(val_list),
            "n_test": len(test_list),
            "checkpoint": best_path if os.path.isfile(best_path) else load_path,
            "last_checkpoint": last_path if os.path.isfile(last_path) else None,
            "resumed": True,
            "skipped_train": True,
        }
        with open(join(save_root, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[{mode}] TEST\n{_format_metrics(test_m)}", flush=True)
        return result

    for epoch in range(start_epoch, FLAGS.epoch_num + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            _, loss, _ = model(batch)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_m = evaluate(model, val_loader, device, FLAGS.target)
        test_m = evaluate(model, test_loader, device, FLAGS.target)
        val_score = val_m.get("rmse_sum_targets", 1e9)
        test_score = test_m.get("rmse_sum_targets", 1e9)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            **{f"val_{k}": v for k, v in val_m.items()},
            **{f"test_{k}": v for k, v in test_m.items()},
        }
        history.append(row)

        if epoch % FLAGS.print_every == 0 or epoch == start_epoch:
            parts = [
                f"[{mode}] epoch={epoch}",
                f"train_loss={row['train_loss']:.4f}",
                f"val_rmse_sum5={val_score:.4f}",
                f"test_rmse_sum5={test_score:.4f}",
            ]
            for t in FLAGS.target:
                k = f"rmse_{t}"
                if k in val_m:
                    parts.append(f"val_{t}={val_m[k]:.4f}")
            for t in FLAGS.target:
                k = f"rmse_{t}"
                if k in test_m:
                    parts.append(f"test_{t}={test_m[k]:.4f}")
            print(" ".join(parts), flush=True)

        # 仅用 val 五指标 RMSE 之和选 best.pt；test 只打印
        if val_score < best_val:
            prev = best_val
            best_val = val_score
            bad_epochs = 0
            torch.save(_ckpt_payload(epoch, val_m, test_m, is_best=True), best_path)
            print(
                f"[{mode}] epoch={epoch} >>> 更新了 best.pt "
                f"(val_rmse_sum5: {prev:.4f} -> {best_val:.4f})",
                flush=True,
            )
        else:
            bad_epochs += 1
            patience = int(getattr(FLAGS, "patience", 0) or 0)
            if patience > 0 and bad_epochs >= patience:
                # 早停前仍写入 last，便于查看中断/停止时的权重
                torch.save(_ckpt_payload(epoch, val_m, test_m, is_best=False), last_path)
                with open(history_path, "w", encoding="utf-8") as f:
                    json.dump(history, f)
                print(
                    f"[{mode}] early stop at epoch={epoch} "
                    f"(no val improve for {patience} epochs, best={best_val:.4f})",
                    flush=True,
                )
                break

        # 每轮覆盖 last.pt：中断后 resume 从上一完整 epoch 继续
        torch.save(_ckpt_payload(epoch, val_m, test_m, is_best=False), last_path)

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f)

    load_path = best_path if os.path.isfile(best_path) else last_path
    ckpt = torch.load(load_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_m = evaluate(model, test_loader, device, FLAGS.target)
    best_epoch = None
    if os.path.isfile(best_path):
        best_epoch = int(torch.load(best_path, map_location="cpu").get("epoch", 0) or 0) or None
    last_epoch = None
    if os.path.isfile(last_path):
        last_epoch = int(torch.load(last_path, map_location="cpu").get("epoch", 0) or 0) or None
    result = {
        "ablation_mode": mode,
        "best_val_rmse_sum5": best_val,
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "test_metrics": test_m,
        "n_train": len(train_list),
        "n_val": len(val_list),
        "n_test": len(test_list),
        "checkpoint": best_path if os.path.isfile(best_path) else load_path,
        "last_checkpoint": last_path if os.path.isfile(last_path) else None,
    }
    with open(join(save_root, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f)
    print(
        f"[{mode}] DONE best_epoch={ckpt.get('epoch')} "
        f"best_val_rmse_sum5={best_val:.4f}\n"
        f"TEST\n{_format_metrics(test_m)}",
        flush=True,
    )
    return result


def _resolve_device():
    device = torch.device(FLAGS.device if torch.cuda.is_available() or "cpu" in FLAGS.device else "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    return device


def _format_metrics(metrics: Dict[str, float]) -> str:
    """分行打印：先 sum5，再每个 target 的 mse/rmse。"""
    mse5 = metrics.get("mse_sum_targets", float("nan"))
    rmse5 = metrics.get("rmse_sum_targets", float("nan"))
    lines = [f"  mse_sum5={mse5:.4f}  rmse_sum5={rmse5:.4f}"]
    for t in FLAGS.target:
        mk, rk = f"mse_{t}", f"rmse_{t}"
        if mk not in metrics and rk not in metrics:
            continue
        mse_s = f"{metrics[mk]:.4f}" if mk in metrics else "nan"
        rmse_s = f"{metrics[rk]:.4f}" if rk in metrics else "nan"
        lines.append(f"  {t:12s}  mse={mse_s}  rmse={rmse_s}")
    return "\n".join(lines)


def finetune_one_mode(ablation_mode: str = None, proto: dict = None) -> dict:
    """
    对齐 HMoE / Soft MoE 小样本适应：
      加载源域 best.pt → 每个 test_10 核取 k 条（kmeans|random）微调整网
      → 核内剩余配置作 query 评测；汇总 mse/rmse sum5。
    """
    mode = ablation_mode or FLAGS.ablation_mode
    device = _resolve_device()
    k = int(getattr(FLAGS, "transfer_k_shot", 50))
    coreset = str(getattr(FLAGS, "coreset", "kmeans"))
    ft_epochs = int(getattr(FLAGS, "ft_epoch_num", None) or FLAGS.epoch_num)
    ft_lr = float(getattr(FLAGS, "ft_lr", None) or FLAGS.lr)

    ckpt_path = (
        getattr(FLAGS, "pretrained_path", None)
        or FLAGS.model_path
        or join(FLAGS.save_dir, mode, "best.pt")
    )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"微调需要源域权重: {ckpt_path}\n"
            "请先 --subtask train（同 comparative_if/model），"
            "或 --pretrained_path / --model_path 指定 best.pt"
        )

    save_root = join(FLAGS.save_dir, mode, f"finetune_k{k}_{coreset}")
    create_dir_if_not_exists(save_root)

    # 划分：只要 test_10 上的样本
    train_list, val_list, test_list, split_info = _prepare_splits(proto=proto)
    test_10 = list(split_info.get("test_10") or [])
    if not test_10:
        raise RuntimeError("split 中无 test_10")
    # test_list 已是 test_10 核；再按核做 k-shot
    support, query, k_stats = _split_kshot_by_kernel(
        test_list, k=k, coreset=coreset, seed=int(FLAGS.seed)
    )
    print(
        f"[finetune/{mode}] ckpt={ckpt_path}\n"
        f"  k={k} coreset={coreset} epochs={ft_epochs} lr={ft_lr}\n"
        f"  support={len(support)} query={len(query)} kernels={len(k_stats)}",
        flush=True,
    )
    for kn, st in k_stats.items():
        print(f"  kernel={kn} n={st['n']} k={st['k']}", flush=True)

    if len(support) == 0 or len(query) == 0:
        raise RuntimeError("k-shot 后 support/query 为空，请检查 test_10 样本数与 k")

    ckpt = torch.load(ckpt_path, map_location=device)
    mode = ckpt.get("ablation_mode", mode)
    target_list = list(ckpt.get("target") or FLAGS.target)
    in_dim = int(ckpt.get("in_dim") or _infer_in_dim(support))
    model = _build_model_from_ckpt(ckpt, mode, in_dim, target_list, device)
    opt = torch.optim.Adam(model.parameters(), lr=ft_lr, weight_decay=FLAGS.weight_decay)

    train_loader = DataLoader(
        AblationDataset(support),
        batch_size=min(FLAGS.batch_size, len(support)),
        shuffle=True,
        collate_fn=collate_ablation,
    )
    query_loader = DataLoader(
        AblationDataset(query),
        batch_size=FLAGS.batch_size,
        shuffle=False,
        collate_fn=collate_ablation,
    )

    # 适应前 query 基线
    base_m = evaluate(model, query_loader, device, target_list)
    print(
        f"[finetune/{mode}] BEFORE  "
        f"mse_sum5={base_m.get('mse_sum_targets', float('nan')):.4f} "
        f"rmse_sum5={base_m.get('rmse_sum_targets', float('nan')):.4f}",
        flush=True,
    )

    best_train = float("inf")
    best_query = float("inf")
    best_query_m = None
    history = []
    train_model_path = join(save_root, "train_model.pt")
    last_path = join(save_root, "last.pt")

    for epoch in range(1, ft_epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"ft/{mode}/ep{epoch}", leave=False):
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            _, loss, _ = model(batch)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(losses)) if losses else float("nan")
        query_m = evaluate(model, query_loader, device, target_list)
        q_mse = float(query_m.get("mse_sum_targets", float("inf")))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "query_mse_sum5": q_mse,
                "query_rmse_sum5": float(query_m.get("rmse_sum_targets", float("nan"))),
            }
        )
        payload = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "ablation_mode": mode,
            "in_dim": in_dim,
            "target": list(target_list),
            "D": int(ckpt.get("D", FLAGS.D)),
            "hidden_num": int(ckpt.get("hidden_num", FLAGS.hidden_num)),
            "gnn_layers": int(ckpt.get("gnn_layers", FLAGS.gnn_layers)),
            "dropout": float(ckpt.get("dropout", FLAGS.dropout)),
            "gnn_type": str(ckpt.get("gnn_type", getattr(FLAGS, "gnn_type", "gcn"))),
            "gnn_activation": str(
                ckpt.get("gnn_activation", getattr(FLAGS, "gnn_activation", "relu"))
            ),
            "comparative_if": bool(
                ckpt.get("comparative_if", getattr(FLAGS, "comparative_if", False))
            ),
            "comparative_model": str(
                ckpt.get("comparative_model", getattr(FLAGS, "comparative_model", "ironman"))
            ),
            "transfer_k_shot": k,
            "coreset": coreset,
            "pretrained_path": ckpt_path,
            "query_metrics": query_m,
            "kshot_stats": k_stats,
            "proto": split_info,
            "before_query_metrics": base_m,
        }
        torch.save(payload, last_path)
        # 对齐 HMoE：按 train loss 存 train_model
        if train_loss < best_train:
            best_train = train_loss
            torch.save(payload, train_model_path)
        if q_mse < best_query:
            best_query = q_mse
            best_query_m = query_m
            torch.save(payload, join(save_root, "best_query.pt"))

        if epoch % max(1, int(getattr(FLAGS, "print_every", 1))) == 0 or epoch == ft_epochs:
            print(
                f"[finetune/{mode}] epoch={epoch}/{ft_epochs} "
                f"train_loss={train_loss:.4f} "
                f"query_mse_sum5={q_mse:.4f} "
                f"query_rmse_sum5={query_m.get('rmse_sum_targets', float('nan')):.4f}",
                flush=True,
            )

    # 最终：用 train_model 再评一次 query（与 HMoE 习惯一致）
    final_ckpt = torch.load(train_model_path, map_location=device)
    model = _build_model_from_ckpt(final_ckpt, mode, in_dim, target_list, device)
    final_m = evaluate(model, query_loader, device, target_list)
    out = {
        "ablation_mode": mode,
        "pretrained_path": ckpt_path,
        "train_model_path": train_model_path,
        "transfer_k_shot": k,
        "coreset": coreset,
        "ft_epoch_num": ft_epochs,
        "n_support": len(support),
        "n_query": len(query),
        "kshot_stats": k_stats,
        "before_query_metrics": base_m,
        "final_query_metrics": final_m,
        "best_query_metrics": best_query_m,
        "comparative_if": bool(getattr(FLAGS, "comparative_if", False)),
        "comparative_model": str(getattr(FLAGS, "comparative_model", "")),
        "seed": int(FLAGS.seed),
        "history": history,
    }
    out_path = join(save_root, "finetune_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    hist_path = join(save_root, "history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(
        f"[finetune/{mode}] DONE\n"
        f"  BEFORE mse_sum5={base_m.get('mse_sum_targets', float('nan')):.4f}\n"
        f"  AFTER  mse_sum5={final_m.get('mse_sum_targets', float('nan')):.4f} "
        f"(train_model)\n"
        f"  BEST_Q mse_sum5={best_query:.4f}\n"
        f"  wrote {out_path}",
        flush=True,
    )
    return out


def infer_one_mode(ablation_mode: str = None) -> dict:
    """加载 best.pt，在对应划分的 val/test 上评估（不训练）。"""
    mode = ablation_mode or FLAGS.ablation_mode
    device = _resolve_device()
    ckpt_path = FLAGS.model_path or join(FLAGS.save_dir, mode, "best.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"找不到 checkpoint: {ckpt_path}；请先 --subtask train，或用 --model_path 指定"
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    mode = ckpt.get("ablation_mode", mode)
    target_list = list(ckpt.get("target") or FLAGS.target)
    in_dim = int(ckpt["in_dim"])
    split_info = dict(ckpt.get("proto") or {})
    # 兼容旧 ckpt：只有 train_24/val_6/test_10 → kernel 划分
    if not split_info.get("split_by"):
        if "train_24" in split_info and "test_10" in split_info:
            split_info["split_by"] = "kernel"
        elif "train_keys" in split_info:
            split_info["split_by"] = "config"

    old_split_by = getattr(FLAGS, "split_by", "config")
    if split_info.get("split_by"):
        FLAGS.split_by = split_info["split_by"]
    try:
        _, val_list, test_list, used_split = _prepare_splits(
            proto=split_info if split_info else None
        )
    finally:
        FLAGS.split_by = old_split_by

    print(
        f"[inference/{mode}] ckpt={ckpt_path} "
        f"split_by={used_split.get('split_by')} "
        f"samples val/test={len(val_list)}/{len(test_list)}",
        flush=True,
    )

    model = AblationNet(
        in_cdfg=in_dim,
        target_list=target_list,
        ablation_mode=mode,
        D=int(ckpt.get("D", FLAGS.D)),
        hidden_num=int(ckpt.get("hidden_num", FLAGS.hidden_num)),
        gnn_layers=int(ckpt.get("gnn_layers", FLAGS.gnn_layers)),
        dropout=float(ckpt.get("dropout", FLAGS.dropout)),
        out_dim=FLAGS.out_dim,
        gnn_type=str(ckpt.get("gnn_type", getattr(FLAGS, "gnn_type", "gcn"))),
        gnn_activation=str(
            ckpt.get("gnn_activation", getattr(FLAGS, "gnn_activation", "relu"))
        ),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_loader = DataLoader(
        AblationDataset(val_list),
        batch_size=FLAGS.batch_size,
        shuffle=False,
        collate_fn=collate_ablation,
    )
    test_loader = DataLoader(
        AblationDataset(test_list),
        batch_size=FLAGS.batch_size,
        shuffle=False,
        collate_fn=collate_ablation,
    )
    val_m = evaluate(model, val_loader, device, target_list)
    test_m = evaluate(model, test_loader, device, target_list)
    print(f"[inference/{mode}] VAL\n{_format_metrics(val_m)}", flush=True)
    print(f"[inference/{mode}] TEST\n{_format_metrics(test_m)}", flush=True)

    out = {
        "ablation_mode": mode,
        "checkpoint": ckpt_path,
        "val_metrics": val_m,
        "test_metrics": test_m,
        "n_val": len(val_list),
        "n_test": len(test_list),
        "split": {
            k: used_split[k]
            for k in used_split
            if k not in ("train_keys", "val_keys", "test_keys")
        },
    }
    save_root = join(FLAGS.save_dir, mode)
    create_dir_if_not_exists(save_root)
    out_path = join(save_root, "inference_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[inference/{mode}] wrote {out_path}", flush=True)
    return out


def run_all_modes() -> List[dict]:
    # 先固定划分，再依次训各模式（保证七模式同一 split）
    _, _, _, split_info = _prepare_splits(proto=None)
    results = []
    for mode in FLAGS.ABLATION_MODES:
        print("=" * 60)
        print("Running ablation_mode =", mode)
        results.append(train_one_mode(mode, proto=split_info))
    summary_path = join(FLAGS.save_dir, "summary_7modes.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Wrote", summary_path)
    return results
