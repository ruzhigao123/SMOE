"""
冻结 CV10 专家 + MoE 门控训练 / 小样本适应。

源域 train：
- 30 kernel（folds.json train_30）内 8:2 → train / val
- test = 协议 test_10；专家默认用 cv10 折权重（test_10 回退 ablation）
- best.pth：默认 beat_first——优先 val 上打败最强单专家的 epoch
- last.pth：每轮覆盖保存当前 epoch（跑到第 N 轮即为第 N 轮权重）
- best_{metric}.pth：分指标选模；结束 head2head

小样本 finetune（对齐 HierarchicalMoE 论文）：
- 加载预训练门控；专家仍冻结
- 每个 test_10 核选 transfer_k_shot 条（coreset=kmeans|random）微调
- 核内剩余配置为 query；无独立 val；按 train_loss 存 train_model.pth
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from os.path import abspath, dirname, join
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS = dirname(abspath(__file__))
_MM = dirname(_THIS)
_HMOE = dirname(_MM)
_SRC = join(_MM, "src")
for _p in (_HMOE, _MM, _THIS, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from multimodal_moe.src.data_ablation import (  # noqa: E402
    AblationDataset,
    build_dataset,
    collate_ablation,
)
from multimodal_moe.src.utils import create_dir_if_not_exists  # noqa: E402

from multimodal_moe.moe_system.cv10_expert_bank import (  # noqa: E402
    DEFAULT_TARGETS,
    EXPERT_MODES,
    Cv10AblationExpertBank,
    load_folds_json,
    load_kernel_split,
)
from multimodal_moe.moe_system.kernel_split import filter_by_kernels, split_kernels_7_3  # noqa: E402
from multimodal_moe.moe_system.moe_cv10_model import FrozenExpertSelector  # noqa: E402
from multimodal_moe.moe_system.selector_moe import (  # noqa: E402
    DEFAULT_SELECTOR_CONFIG, first_order_mldg_step, kernel_family,
    sample_family_episode, selector_config_from_args, split_family_kernels,
)


def _device(name: str) -> torch.device:
    d = torch.device(name if torch.cuda.is_available() or "cpu" in name else "cpu")
    if str(d).startswith("cuda") and not torch.cuda.is_available():
        d = torch.device("cpu")
    return d


def _mse(pred: np.ndarray, true: np.ndarray) -> float:
    """评估/选模用 MSE（对齐 HierarchicalMoE 论文离线指标）。"""
    return float(np.mean((pred - true) ** 2))


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def _bucket_by_route(data_list, bank: Cv10AblationExpertBank):
    buckets: Dict[Tuple, list] = defaultdict(list)
    for d in data_list:
        k = str(getattr(d, "kernel", ""))
        buckets[bank.route_key(k)].append(d)
    return buckets


def _make_loaders(buckets, batch_size: int, shuffle: bool):
    loaders = []
    for key, items in buckets.items():
        if not items:
            continue
        ld = DataLoader(
            AblationDataset(items),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_ablation,
        )
        loaders.append((key, ld))
    return loaders


def _encode_gate_feat(bank, batch, key, gate_strategy="legacy"):
    if gate_strategy == "selector":
        return bank.encode_selector_feat(batch, key)
    return bank.encode_gate_feat(batch, key)


def _checkpoint_strategy(ckpt):
    return ckpt.get("gate_strategy", (ckpt.get("split") or {}).get("gate_strategy", "legacy"))


def _gate_metadata(model):
    metadata = {"gate_strategy": model.gate_strategy}
    if model.gate_strategy == "selector":
        metadata["selector_config"] = dict(model.selector_config)
    return metadata


def _restore_selector_source_args(args, ckpt, pretrained):
    if _checkpoint_strategy(ckpt) != "selector":
        raise RuntimeError("--gate_strategy selector 的微调必须加载 selector 源域权重；请先用该方案 train")
    split = ckpt.get("split") or {}
    config = {**DEFAULT_SELECTOR_CONFIG, **ckpt.get("selector_config", split.get("selector_config", {}))}
    for name, value in config.items():
        if name not in getattr(args, "_explicit_selector_options", set()):
            setattr(args, "selector_" + name, value)
    for name in ("gate_mode", "mix_mode", "min_special", "balance_coef", "perf_loss_weight",
                 "prior_ast_on_perf", "prior_text_on_util", "use_prior", "use_pred_aware",
                 "use_residual", "pred_slot_dim", "gate_use_pragma", "pragma_proj_dim"):
        if name in split:
            setattr(args, name, split[name])
    if args.gate_use_pragma:
        args._selector_pragma_encoder = split.get("pragma_encoder") or join(dirname(pretrained), "pragma_encoder.json")


def _selector_structure_metadata(args):
    return {name: getattr(args, name) for name in (
        "gate_mode", "mix_mode", "min_special", "balance_coef", "perf_loss_weight",
        "prior_ast_on_perf", "prior_text_on_util", "use_prior", "use_pred_aware",
        "use_residual", "pred_slot_dim", "gate_use_pragma", "pragma_proj_dim",
    ) if hasattr(args, name)}


@torch.no_grad()
def _fit_selector_statistics(model, bank, train_list, batch_size, device):
    if model.gate_strategy != "selector" or bool(model.gate.scorer.statistics_fitted):
        return
    predictions, truths, summaries = [], [], []
    for key, loader in _make_loaders(_bucket_by_route(train_list, bank), batch_size, shuffle=False):
        for batch in loader:
            batch = batch.to(device)
            prediction = bank.predict_homogeneous_batch(batch, key)
            features = _encode_gate_feat(bank, batch, key, "selector")
            truth = torch.stack([
                getattr(batch, target.replace("-", "_")).view(-1).float()
                for target in model.target_list
            ], dim=1)
            predictions.append(prediction.cpu())
            truths.append(truth.cpu())
            summaries.append(model.gate.scorer.candidate_summaries(features).cpu())
    if not predictions:
        raise ValueError("selector 训练统计没有样本")
    model.gate.scorer.fit_statistics(torch.cat(predictions), torch.cat(truths), torch.cat(summaries))
    print(f"[moe_cv10/selector] fitted statistics on {sum(map(len, predictions))} TRAIN points only", flush=True)


def _train_selector_mldg_epoch(model, bank, train_list, opt, device, epoch, seed):
    config = model.selector_config
    by_kernel = defaultdict(list)
    for sample in train_list:
        by_kernel[str(sample.kernel)].append(sample)
    rng = random.Random(int(seed) + 1009 * epoch)

    def objective(kernels):
        losses = []
        for kernel in kernels:
            pool = by_kernel[kernel]
            items = rng.sample(pool, min(len(pool), int(config["points_per_kernel"])))
            for key, loader in _make_loaders(_bucket_by_route(items, bank), len(items), shuffle=False):
                for batch in loader:
                    batch = batch.to(device)
                    with torch.no_grad():
                        prediction = bank.predict_homogeneous_batch(batch, key)
                        features = _encode_gate_feat(bank, batch, key, "selector")
                    fused, weights = model.fuse(features, prediction, pragma_feat=_batch_pragma(batch))
                    target_losses, _ = model.selector_target_losses(fused, batch, weights, epoch, meta=True)
                    losses.append(target_losses)
        return torch.stack(losses).mean(0)

    for group in opt.param_groups:
        group["lr"] = float(config["meta_lr"])
    results = []
    for _ in range(int(config["mldg_steps"])):
        support, meta = sample_family_episode(
            list(by_kernel), int(config["support_families"]), int(config["meta_families"]), rng
        )
        results.append(first_order_mldg_step(
            model, opt, lambda: objective(support), lambda: objective(meta), float(config["inner_lr"])
        ))
    metrics = {name: float(np.mean([row[name] for row in results])) for name in results[0]}
    metrics["train_stage"] = "mldg"
    return [0.25 * row["support_loss"] + 0.75 * row["meta_loss"] for row in results], metrics


@torch.no_grad()
def evaluate_experts_alone(
    bank: Cv10AblationExpertBank,
    data_list,
    batch_size: int,
    device: torch.device,
    target_list: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """
    与 MoE 同一批样本上，分别评测三个冻结专家 + 均匀混合 + oracle(每样本每指标选最优专家)。
    用于判断：MoE 差是门控问题，还是专家本身 MSE 就高。
    """
    buckets = _bucket_by_route(data_list, bank)
    # expert e, target t
    preds = {
        m: {t: [] for t in target_list} for m in list(EXPERT_MODES) + ["uniform", "oracle"]
    }
    trues = {t: [] for t in target_list}
    for key, loader in _make_loaders(buckets, batch_size, shuffle=False):
        for batch in loader:
            batch = batch.to(device)
            ep = bank.predict_homogeneous_batch(batch, key)  # [B,3,T]
            B = ep.size(0)
            y_cols = []
            for i, t in enumerate(target_list):
                attr = t.replace("-", "_")
                y = getattr(batch, attr, None)
                if y is None:
                    y_cols.append(None)
                    continue
                yv = y.detach().cpu().view(-1).numpy()
                trues[t].append(yv)
                y_cols.append(yv)
                ep_np = ep[:, :, i].detach().cpu().numpy()  # [B,3]
                for ei, mode in enumerate(EXPERT_MODES):
                    preds[mode][t].append(ep_np[:, ei])
                preds["uniform"][t].append(ep_np.mean(axis=1))
                # oracle: pick expert closest to y
                err = np.abs(ep_np - yv.reshape(-1, 1))
                best_e = err.argmin(axis=1)
                preds["oracle"][t].append(ep_np[np.arange(B), best_e])

    out: Dict[str, Dict[str, float]] = {}
    for name in list(EXPERT_MODES) + ["uniform", "oracle"]:
        metrics: Dict[str, float] = {}
        for t in target_list:
            if not preds[name][t] or not trues[t]:
                continue
            p = np.concatenate(preds[name][t])
            g = np.concatenate(trues[t])
            metrics[f"mse_{t}"] = _mse(p, g)
        keys = [f"mse_{t}" for t in target_list if f"mse_{t}" in metrics]
        if keys:
            metrics["mse_sum_targets"] = float(sum(metrics[k] for k in keys))
            metrics["mse_mean_targets"] = float(np.mean([metrics[k] for k in keys]))
        out[name] = metrics
    return out


@torch.no_grad()
def evaluate_selector(
    model: FrozenExpertSelector,
    bank: Cv10AblationExpertBank,
    data_list,
    batch_size: int,
    device: torch.device,
    target_list: Sequence[str],
) -> Dict[str, float]:
    model.eval()
    buckets = _bucket_by_route(data_list, bank)
    preds = {t: [] for t in target_list}
    trues = {t: [] for t in target_list}
    weight_chunks: List[torch.Tensor] = []
    for key, loader in _make_loaders(buckets, batch_size, shuffle=False):
        for batch in loader:
            batch = batch.to(device)
            expert_preds = bank.predict_homogeneous_batch(batch, key)
            gate_feat = _encode_gate_feat(bank, batch, key, model.gate_strategy)
            fused, w = model.fuse(gate_feat, expert_preds, pragma_feat=_batch_pragma(batch))
            weight_chunks.append(w.detach().cpu())
            for i, t in enumerate(target_list):
                attr = t.replace("-", "_")
                y = getattr(batch, attr, None)
                if y is None:
                    continue
                preds[t].append(fused[:, i].detach().cpu().view(-1).numpy())
                trues[t].append(y.detach().cpu().view(-1).numpy())
    if weight_chunks:
        model.gate._last_weights = torch.cat(weight_chunks, dim=0)
    metrics = {}
    for t in target_list:
        if not preds[t]:
            continue
        p = np.concatenate(preds[t])
        g = np.concatenate(trues[t])
        metrics[f"mse_{t}"] = _mse(p, g)
        metrics[f"rmse_{t}"] = _rmse(p, g)
    mse_keys = [f"mse_{t}" for t in target_list if f"mse_{t}" in metrics]
    if mse_keys:
        metrics["mse_sum_targets"] = float(sum(metrics[k] for k in mse_keys))
        metrics["mse_mean_targets"] = float(np.mean([metrics[k] for k in mse_keys]))
    rmse_keys = [f"rmse_{t}" for t in target_list if f"rmse_{t}" in metrics]
    if rmse_keys:
        metrics["rmse_sum_targets"] = float(sum(metrics[k] for k in rmse_keys))
        metrics["rmse_mean_targets"] = float(np.mean([metrics[k] for k in rmse_keys]))
    return metrics


def _format_metrics(
    prefix: str,
    metrics: Dict[str, float],
    targets: Sequence[str],
    *,
    include_rmse: bool = False,
) -> str:
    parts = [f"{prefix}_mse_sum5={metrics.get('mse_sum_targets', float('nan')):.4f}"]
    for t in targets:
        k = f"mse_{t}"
        if k in metrics:
            # 训练日志保持短标签；inference 需同时打 rmse 时用 mse_ 前缀避免歧义
            tag = f"{prefix}_mse_{t}" if include_rmse else f"{prefix}_{t}"
            parts.append(f"{tag}={metrics[k]:.4f}")
    if include_rmse:
        parts.append(
            f"{prefix}_rmse_sum5={metrics.get('rmse_sum_targets', float('nan')):.4f}"
        )
        for t in targets:
            k = f"rmse_{t}"
            if k in metrics:
                parts.append(f"{prefix}_rmse_{t}={metrics[k]:.4f}")
    return " ".join(parts)


def _best_single_expert(
    expert_baselines: Dict[str, Dict[str, float]],
    target_list: Sequence[str],
    metric_key: str = "mse_sum_targets",
) -> Tuple[Optional[str], float]:
    """在 cdfg_only/ast_only/LM 中取指定 metric 最低者。"""
    best_name, best_v = None, float("inf")
    for name in EXPERT_MODES:
        m = expert_baselines.get(name) or {}
        v = float(m.get(metric_key, float("inf")))
        if np.isfinite(v) and v < best_v:
            best_v, best_name = v, name
    return best_name, best_v


def build_head2head(
    expert_baselines: Dict[str, Dict[str, float]],
    moe_metrics: Dict[str, float],
    target_list: Sequence[str],
    split_name: str,
    beat_margin: float = 0.0,
) -> dict:
    """
    同一划分上：单专家 / uniform / oracle / MoE 并排，并判定 MoE 是否有效。
    有效定义：moe_mse_sum5 + margin < 最强单专家 mse_sum5。
    """
    rows = {}
    for name in list(EXPERT_MODES) + ["uniform", "oracle", "MoE"]:
        src = moe_metrics if name == "MoE" else (expert_baselines.get(name) or {})
        rows[name] = {
            "mse_sum5": float(src.get("mse_sum_targets", float("nan"))),
            **{t: float(src.get(f"mse_{t}", float("nan"))) for t in target_list},
        }

    best_exp, best_exp_sum5 = _best_single_expert(expert_baselines, target_list)
    moe_sum5 = rows["MoE"]["mse_sum5"]
    gap = (
        float(best_exp_sum5 - moe_sum5)
        if np.isfinite(best_exp_sum5) and np.isfinite(moe_sum5)
        else float("nan")
    )
    effective = bool(
        np.isfinite(gap) and moe_sum5 + float(beat_margin) < best_exp_sum5
    )

    per_target = {}
    n_beat_t = 0
    for t in target_list:
        key = f"mse_{t}"
        be_name, be_v = _best_single_expert(expert_baselines, target_list, key)
        mv = float(moe_metrics.get(key, float("nan")))
        beat_t = bool(np.isfinite(mv) and np.isfinite(be_v) and mv + beat_margin < be_v)
        if beat_t:
            n_beat_t += 1
        per_target[t] = {
            "best_expert": be_name,
            "best_expert_mse": be_v if np.isfinite(be_v) else None,
            "moe_mse": mv if np.isfinite(mv) else None,
            "gap": (float(be_v - mv) if np.isfinite(be_v) and np.isfinite(mv) else None),
            "moe_beats_expert": beat_t,
        }

    verdict = {
        "split": split_name,
        "metric": "mse",
        "best_single_expert": best_exp,
        "best_single_sum5": best_exp_sum5 if np.isfinite(best_exp_sum5) else None,
        "moe_sum5": moe_sum5 if np.isfinite(moe_sum5) else None,
        "gap_sum5_expert_minus_moe": gap if np.isfinite(gap) else None,
        "beat_margin": float(beat_margin),
        "moe_effective": effective,
        "n_targets_moe_beats_expert": n_beat_t,
        "n_targets": len(target_list),
        "note": (
            "有效: MoE mse_sum5 + margin < 最强单专家；"
            "无效则门控无增益，应检查专家/选模"
            if not effective
            else "有效: MoE 相对最强单专家有增益（MSE）"
        ),
    }
    return {"rows": rows, "per_target": per_target, "verdict": verdict}


def format_head2head_table(report: dict, target_list: Sequence[str]) -> str:
    v = report["verdict"]
    rows = report["rows"]
    lines = [
        f"======== head-to-head (MSE) @ {v['split']} ========",
        f"{'method':12s}"
        + f"{'sum5':>10s}"
        + "".join(f"{t:>12s}" for t in target_list),
    ]
    order = list(EXPERT_MODES) + ["uniform", "oracle", "MoE"]
    for name in order:
        r = rows.get(name) or {}
        line = f"{name:12s}{r.get('mse_sum5', float('nan')):10.4f}"
        for t in target_list:
            line += f"{r.get(t, float('nan')):12.4f}"
        if name == v.get("best_single_expert"):
            line += "  <-- best single"
        if name == "MoE":
            line += "  <-- MoE"
        lines.append(line)
    eff = "YES" if v.get("moe_effective") else "NO"
    lines.append(
        f"verdict: moe_effective={eff} | best_expert={v.get('best_single_expert')} "
        f"mse_sum5={v.get('best_single_sum5')} | moe_mse_sum5={v.get('moe_sum5')} "
        f"| gap(expert-moe)={v.get('gap_sum5_expert_minus_moe')} "
        f"| targets_beat={v.get('n_targets_moe_beats_expert')}/{v.get('n_targets')}"
    )
    lines.append(f"note: {v.get('note')}")
    lines.append("=" * 40)
    return "\n".join(lines)


def _load_selector_ckpt(model: FrozenExpertSelector, ckpt: dict) -> None:
    saved_strategy = _checkpoint_strategy(ckpt)
    if saved_strategy != model.gate_strategy:
        raise RuntimeError(f"checkpoint gate_strategy={saved_strategy}，当前为 {model.gate_strategy}；请使用对应方案的权重")
    if model.gate_strategy == "selector" and list(ckpt.get("target") or model.target_list) != model.target_list:
        raise RuntimeError("selector checkpoint 的指标顺序与当前配置不同")
    model.gate.load_state_dict(ckpt["gate"])
    if model.gate_strategy == "selector" and not bool(model.gate.scorer.statistics_fitted):
        raise RuntimeError("selector checkpoint 缺少已拟合的训练统计量")
    if model.pragma_proj is not None:
        if ckpt.get("pragma_proj") is not None:
            model.pragma_proj.load_state_dict(ckpt["pragma_proj"])
        else:
            raise RuntimeError("checkpoint 缺少 pragma_proj，与当前 use_pragma=True 不兼容")
    if model.residual is not None:
        if ckpt.get("residual") is not None:
            model.residual.load_state_dict(ckpt["residual"])
        else:
            raise RuntimeError("checkpoint 缺少 residual，与当前 use_residual=True 不兼容")


def _selector_state_dict(model: FrozenExpertSelector) -> dict:
    out = {"gate": model.gate.state_dict(), **_gate_metadata(model)}
    if model.pragma_proj is not None:
        out["pragma_proj"] = model.pragma_proj.state_dict()
    if model.residual is not None:
        out["residual"] = model.residual.state_dict()
    return out


def _metric_ckpt_name(target: str) -> str:
    """util-LUT → best_util_LUT.pth（与分指标独立选模文件名）。"""
    safe = str(target).replace("-", "_")
    return f"best_{safe}.pth"


def _metric_ckpt_path(save_dir: str, target: str) -> str:
    return join(save_dir, _metric_ckpt_name(target))


def _batch_pragma(batch) -> Optional[torch.Tensor]:
    return getattr(batch, "pragma_feat", None)


def _prepare_pragma_features(args, train_list, val_list, test_list, save_dir: str):
    """可选：fit encoder → attach pragma_feat；返回 (use_pragma, pragma_in_dim, enc_path)。"""
    use_pragma = bool(getattr(args, "gate_use_pragma", False))
    if not use_pragma:
        return False, 0, None

    from multimodal_moe.moe_system.pragma_io import (
        attach_pragma_features,
        default_c_src_roots,
        default_points_roots,
        fit_pragma_encoder_from_train,
    )

    ds_mode = getattr(args, "dataset_mode", None)
    points_roots = list(default_points_roots(ds_mode))
    if getattr(args, "points_root", None):
        points_roots = [args.points_root] + [r for r in points_roots if r != args.points_root]
    c_roots = list(default_c_src_roots(ds_mode))
    if getattr(args, "c_src_root", None):
        c_roots = [args.c_src_root] + [r for r in c_roots if r != args.c_src_root]

    source_encoder = getattr(args, "_selector_pragma_encoder", None)
    if source_encoder and getattr(args, "gate_strategy", "legacy") == "selector":
        from multimodal_moe.moe_system.pragma_features import PragmaFeatureEncoder
        enc = PragmaFeatureEncoder.load(source_encoder)
        fit_stats = {"n_keys": len(enc.keys_), "train_with_pragma": "reused-source", "train_total": len(train_list)}
    else:
        enc, fit_stats = fit_pragma_encoder_from_train(
            train_list, points_roots=points_roots, c_src_roots=c_roots
        )
    enc_path = join(save_dir, "pragma_encoder.json")
    enc.save(enc_path)
    st_tr = attach_pragma_features(
        train_list, enc, points_roots=points_roots, c_src_roots=c_roots
    )
    st_va = attach_pragma_features(
        val_list, enc, points_roots=points_roots, c_src_roots=c_roots
    )
    st_te = attach_pragma_features(
        test_list, enc, points_roots=points_roots, c_src_roots=c_roots
    )
    print(
        f"[moe_cv10] gate_use_pragma=True encoder_dim={enc.dim_} keys={fit_stats.get('n_keys')} "
        f"train_points={fit_stats.get('train_with_pragma')}/{fit_stats.get('train_total')} "
        f"attach found/missing "
        f"train={st_tr['found']}/{st_tr['missing']} "
        f"val={st_va['found']}/{st_va['missing']} "
        f"test={st_te['found']}/{st_te['missing']}",
        flush=True,
    )
    if st_tr["found"] == 0:
        print(
            "[moe_cv10] WARNING: 训练集未解析到任何 pragma/point；"
            "请检查 --points_root 或 --c_src_root（.c / points_list.pkl）",
            flush=True,
        )
    return True, int(enc.dim_), enc_path


@torch.no_grad()
def _embed_gate_feats(
    items,
    bank: Cv10AblationExpertBank,
    device: torch.device,
    batch_size: int,
    gate_strategy: str = "legacy",
) -> np.ndarray:
    """按 DataLoader 顺序提取门控编码（对齐 HMoE coreset 用 hidden 做 kmeans）。"""
    if not items:
        return np.zeros((0, 1), dtype=np.float32)
    key = bank.route_key(str(getattr(items[0], "kernel", "")))
    chunks = []
    loader = DataLoader(
        AblationDataset(items),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ablation,
    )
    for batch in loader:
        batch = batch.to(device)
        feat = _encode_gate_feat(bank, batch, key, gate_strategy)
        chunks.append(feat.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1), dtype=np.float32)


def _kmeans_indices(feats: np.ndarray, k: int, seed: int) -> List[int]:
    n = int(feats.shape[0])
    k = min(int(k), n)
    if k <= 0:
        return []
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, random_state=int(seed), n_init=10, max_iter=300)
        labels = km.fit_predict(feats)
        centroids = km.cluster_centers_
    except Exception as e:
        print(f"[moe_cv10/ft] kmeans 失败，回退 random: {e}", flush=True)
        rng = np.random.RandomState(seed)
        return sorted(rng.choice(n, size=k, replace=False).tolist())
    selected: List[int] = []
    used = set()
    for c in range(k):
        idxs = np.where(labels == c)[0]
        if idxs.size == 0:
            continue
        dist = np.linalg.norm(feats[idxs] - centroids[c], axis=1)
        order = idxs[np.argsort(dist)]
        pick = None
        for j in order:
            jj = int(j)
            if jj not in used:
                pick = jj
                break
        if pick is None:
            continue
        used.add(pick)
        selected.append(pick)
    if len(selected) < k:
        remain = [i for i in range(n) if i not in used]
        selected.extend(remain[: k - len(selected)])
    return sorted(selected)


def select_kshot_per_kernel(
    data_list,
    k: int,
    coreset: str,
    seed: int,
    bank: Cv10AblationExpertBank,
    device: torch.device,
    batch_size: int,
    gate_strategy: str = "legacy",
) -> Tuple[list, list, dict]:
    """
    对齐 HMoE src/data.py split_dataset(finetune)：每个未见核取 k 条适应，其余当 query/test；无 val。
    coreset=kmeans：在门控 384 维编码上聚类，取距质心最近样本。
    """
    by_k: Dict[str, list] = defaultdict(list)
    for d in data_list:
        by_k[str(getattr(d, "kernel", ""))].append(d)
    train_sel: list = []
    query: list = []
    stats: dict = {}
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
            feats = _embed_gate_feats(items, bank, device, batch_size, gate_strategy)
            idxs = _kmeans_indices(feats, k_use, seed)
        else:
            idxs = rng.choice(n, size=k_use, replace=False).tolist()
        sel = set(int(i) for i in idxs)
        for i, d in enumerate(items):
            if i in sel:
                train_sel.append(d)
            else:
                query.append(d)
        stats[kernel] = {
            "n": n,
            "k": len(sel),
            "method": coreset,
            "cfg_ids": [
                int(getattr(items[i], "cfg_id", -1)) for i in sorted(sel)
            ],
        }
    return train_sel, query, stats


def _build_selector(args, bank, target_list, use_pragma, pragma_in_dim, device):
    if getattr(args, "gate_strategy", "legacy") == "selector":
        torch.manual_seed(int(args.seed))
    return FrozenExpertSelector(
        bank,
        target_list=target_list,
        prior_ast_on_perf=getattr(args, "prior_ast_on_perf", 0.5),
        prior_text_on_util=getattr(args, "prior_text_on_util", 0.5),
        use_prior=getattr(args, "use_prior", True),
        min_special=float(getattr(args, "min_special", 0.0)),
        balance_coef=float(getattr(args, "balance_coef", 0.01)),
        gate_mode=str(getattr(args, "gate_mode", "free")),
        mix_mode=str(getattr(args, "mix_mode", "soft")),
        use_pred_aware=bool(getattr(args, "use_pred_aware", True)),
        use_residual=bool(getattr(args, "use_residual", False)),
        perf_loss_weight=float(getattr(args, "perf_loss_weight", 3.0)),
        pred_slot_dim=int(getattr(args, "pred_slot_dim", 16)),
        use_pragma=use_pragma,
        pragma_in_dim=pragma_in_dim,
        pragma_proj_dim=int(getattr(args, "pragma_proj_dim", 64)),
        gate_strategy=str(getattr(args, "gate_strategy", "legacy")),
        selector_config=selector_config_from_args(args),
    ).to(device)


def finetune_moe_cv10(args, train_30: List[str], test_10: List[str]) -> dict:
    """
    论文小样本适应：预训练门控 → 每个 test_10 核 k 条微调（专家冻结）→ 剩余当 query。
    无独立 val；按 train_loss 存 train_model.pth（对齐 HMoE train_model_state_dict）。
    """
    device = _device(args.device)
    target_list = list(args.target)
    k = int(getattr(args, "transfer_k_shot", 50))
    coreset = str(getattr(args, "coreset", "kmeans"))
    base_save = args.save_dir
    save_dir = getattr(args, "ft_save_dir", None) or join(
        base_save, f"finetune_k{k}_{coreset}"
    )
    create_dir_if_not_exists(save_dir)
    user_pretrained = getattr(args, "pretrained_path", None) or getattr(
        args, "model_path", None
    )
    pretrained = user_pretrained or join(base_save, "best.pth")
    if not os.path.isfile(pretrained):
        raise FileNotFoundError(
            f"微调需要预训练门控: {pretrained}\n"
            "请先 --subtask train，或指定 --pretrained_path / --model_path"
        )
    source_ckpt = None
    if getattr(args, "gate_strategy", "legacy") == "selector":
        source_ckpt = torch.load(pretrained, map_location=device)
        _restore_selector_source_args(args, source_ckpt, pretrained)
        target_list = list(source_ckpt.get("target") or target_list)
    # 提醒：源域长训若只更新了 last.pth、best 仍停在很早的 epoch，微调会拿到旧权重
    if not user_pretrained:
        src_last = join(base_save, "last.pth")
        if os.path.isfile(src_last) and os.path.isfile(pretrained):
            try:
                _be = int(torch.load(pretrained, map_location="cpu").get("epoch", 0) or 0)
                _le = int(torch.load(src_last, map_location="cpu").get("epoch", 0) or 0)
                if _le > _be:
                    print(
                        f"[moe_cv10/ft] WARNING: 默认加载 best.pth (epoch={_be})，"
                        f"但同目录 last.pth 已到 epoch={_le}。\n"
                        f"  best 只在选模更优时更新；长训续跑的权重在 last.pth。\n"
                        f"  若要用最后一轮源域权重微调，请加：\n"
                        f"  --pretrained_path {src_last}",
                        flush=True,
                    )
            except Exception:
                pass
    ft_lr = float(getattr(args, "ft_lr", None) or args.lr)
    ft_epochs = int(getattr(args, "ft_epoch_num", None) or args.epoch_num)

    from multimodal_moe.moe_system.utils_new import (
        default_data_paths,
        normalize_dataset_mode,
        set_dataset_mode,
    )

    ds_mode = normalize_dataset_mode(getattr(args, "dataset_mode", "filtered"))
    set_dataset_mode(ds_mode)
    skip_zero = getattr(args, "skip_all_zero_config", None)
    if skip_zero is None:
        skip_zero = ds_mode == "filtered"
    paths = default_data_paths(ds_mode)
    print(
        f"[moe_cv10/ft] HierarchicalMoE-style few-shot\n"
        f"  k={k} coreset={coreset} lr={ft_lr} epochs={ft_epochs}\n"
        f"  pretrained={pretrained}\n"
        f"  save_dir={save_dir}\n"
        f"  dataset_mode={ds_mode} cdfg={paths['cdfg_root']}",
        flush=True,
    )
    data_all = build_dataset(
        target_list=target_list,
        skip_all_zero=bool(skip_zero),
        kernels=list(test_10),
        dataset_mode=ds_mode,
    )
    pool = filter_by_kernels(data_all, test_10)
    if not pool:
        raise RuntimeError("test_10 无样本，无法 few-shot")

    print("[moe_cv10/ft] loading frozen experts...", flush=True)
    bank = Cv10AblationExpertBank(
        device=device,
        cv10_root=args.cv10_root,
        ablation_root=args.ablation_root,
        folds_json=args.folds_json,
        kernel_split_json=args.kernel_split_json,
        target_list=target_list,
        preload_all=True,
        expert_source=getattr(args, "expert_source", "cv10"),
    )
    shot_list, query_list, kshot_stats = select_kshot_per_kernel(
        pool,
        k=k,
        coreset=coreset,
        seed=int(args.seed),
        bank=bank,
        device=device,
        batch_size=args.batch_size,
        gate_strategy=getattr(args, "gate_strategy", "legacy"),
    )
    split_info = {
        "finetune": True,
        "transfer_k_shot": k,
        "coreset": coreset,
        "seed": args.seed,
        "train_30": train_30,
        "test_10": test_10,
        "n_shot": len(shot_list),
        "n_query": len(query_list),
        "per_kernel": kshot_stats,
        "pretrained": pretrained,
        "note": "论文：每未见核 k 条适应、无 val；query=核内剩余配置",
    }
    split_info["gate_strategy"] = getattr(args, "gate_strategy", "legacy")
    if split_info["gate_strategy"] == "selector":
        split_info.update(_selector_structure_metadata(args))
        split_info["selector_config"] = selector_config_from_args(args)
    with open(join(save_dir, "kshot_split.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    print(
        f"[moe_cv10/ft] samples shot/query = {len(shot_list)}/{len(query_list)} "
        f"kernels={len(kshot_stats)}",
        flush=True,
    )
    for kn, st in kshot_stats.items():
        print(f"  {kn}: n={st['n']} k={st['k']}", flush=True)
    if not shot_list:
        raise RuntimeError("k-shot 训练集为空")

    use_pragma, pragma_in_dim, pragma_enc_path = _prepare_pragma_features(
        args, shot_list, [], query_list, save_dir
    )
    split_info["gate_use_pragma"] = use_pragma
    split_info["pragma_in_dim"] = pragma_in_dim
    split_info["pragma_proj_dim"] = int(getattr(args, "pragma_proj_dim", 64))
    split_info["pragma_encoder"] = pragma_enc_path
    with open(join(save_dir, "kshot_split.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print("[moe_cv10/ft] === expert baselines on QUERY (held-out configs) ===", flush=True)
    base_q = evaluate_experts_alone(
        bank, query_list, args.batch_size, device, target_list
    )
    for name, m in base_q.items():
        print(
            f"[moe_cv10/ft] baseline_query/{name} "
            f"{_format_metrics('query', m, target_list)}",
            flush=True,
        )
    with open(join(save_dir, "expert_baseline_query.json"), "w", encoding="utf-8") as f:
        json.dump(base_q, f, indent=2)

    model = _build_selector(args, bank, target_list, use_pragma, pragma_in_dim, device)
    ckpt0 = source_ckpt if source_ckpt is not None else torch.load(pretrained, map_location=device)
    try:
        _load_selector_ckpt(model, ckpt0)
    except RuntimeError as e:
        raise RuntimeError(f"预训练门控与当前结构不兼容: {e}") from e
    print(
        f"[moe_cv10/ft] loaded source pretrained {pretrained} "
        f"(source_epoch={ckpt0.get('epoch')})",
        flush=True,
    )

    opt = torch.optim.Adam(
        model.trainable_parameters(), lr=ft_lr, weight_decay=args.weight_decay
    )
    best_train = float("inf")
    best_path = join(save_dir, "train_model.pth")  # 对齐论文 train_model_state_dict
    last_path = join(save_dir, "last.pth")
    history_path = join(save_dir, "history.json")
    history: List[dict] = []
    start_epoch = 1
    end_epoch = ft_epochs
    resumed_ft = False

    # 再次微调：默认接着上次 ft 权重（train_model.pth）；可选 last.pth
    # --no_resume / --ft_resume_from none → 只用源域预训练，epoch 从 1 重开
    ft_resume_from = str(getattr(args, "ft_resume_from", "train_model") or "train_model")
    if not getattr(args, "resume", True):
        ft_resume_from = "none"
    resume_candidates = {
        "train_model": best_path,
        "last": last_path,
    }
    if ft_resume_from == "auto":
        # 优先 train_model；没有则 last
        ft_resume_path = (
            best_path
            if os.path.isfile(best_path)
            else (last_path if os.path.isfile(last_path) else None)
        )
    elif ft_resume_from in resume_candidates:
        ft_resume_path = resume_candidates[ft_resume_from]
        if ft_resume_path and not os.path.isfile(ft_resume_path):
            alt = last_path if ft_resume_from == "train_model" else best_path
            if os.path.isfile(alt):
                print(
                    f"[moe_cv10/ft] {ft_resume_path} 不存在，回退到 {alt}",
                    flush=True,
                )
                ft_resume_path = alt
            else:
                ft_resume_path = None
    else:
        ft_resume_path = None

    if ft_resume_path and os.path.isfile(ft_resume_path):
        ft_ckpt = torch.load(ft_resume_path, map_location=device)
        try:
            _load_selector_ckpt(model, ft_ckpt)
            if ft_ckpt.get("optimizer") is not None:
                try:
                    opt.load_state_dict(ft_ckpt["optimizer"])
                except Exception as e:
                    print(f"[moe_cv10/ft] WARNING optimizer not loaded: {e}", flush=True)
            # train_model 带 best train_loss；last 可能没有更优记录 → 仍读 train_model 的 loss
            best_train = float(ft_ckpt.get("train_loss", float("inf")))
            if ft_resume_path == last_path and os.path.isfile(best_path):
                try:
                    tm = torch.load(best_path, map_location="cpu")
                    best_train = float(tm.get("train_loss", best_train))
                except Exception:
                    pass
            ckpt_epoch = int(ft_ckpt.get("epoch", 0))
            start_epoch = ckpt_epoch + 1
            if start_epoch <= ft_epochs:
                end_epoch = ft_epochs
            else:
                end_epoch = ckpt_epoch + ft_epochs
            if os.path.isfile(history_path):
                try:
                    with open(history_path, encoding="utf-8") as f:
                        history = json.load(f)
                except Exception:
                    history = []
            resumed_ft = True
            print(
                f"[moe_cv10/ft] RESUME from previous finetune: {ft_resume_path} "
                f"(ft_epoch={ckpt_epoch}) → train {start_epoch}..{end_epoch} "
                f"| tracked_best_train_loss={best_train:.4f}",
                flush=True,
            )
        except RuntimeError as e:
            print(
                f"[moe_cv10/ft] WARNING ft ckpt incompatible; "
                f"keep source pretrained weights.\n  detail: {e}",
                flush=True,
            )
    elif ft_resume_from != "none":
        print(
            f"[moe_cv10/ft] 无上次微调权重（{save_dir} 下无 train_model.pth/last.pth），"
            f"从源域预训练起步，epoch 1..{end_epoch}",
            flush=True,
        )
    else:
        print(
            f"[moe_cv10/ft] --no_resume / ft_resume_from=none：忽略上次微调，"
            f"从源域预训练重开 epoch 1..{end_epoch}",
            flush=True,
        )

    start_m = evaluate_selector(
        model, bank, query_list, args.batch_size, device, target_list
    )
    zero_m = start_m  # 报告里仍叫 zero-shot：未 resume 时即源域；resume 时为续训起点
    tag = "resume-start" if resumed_ft else "zero-shot"
    print(
        f"[moe_cv10/ft] {tag} QUERY {_format_metrics('query', start_m, target_list)}",
        flush=True,
    )

    train_buckets = _bucket_by_route(shot_list, bank)
    uniform_eval_only = str(getattr(model, "mix_mode", "soft")) == "uniform"
    if uniform_eval_only:
        print(
            "[moe_cv10/ft] mix_mode=uniform: 权重固定为 1/3，"
            "不反传，只在 query 上评估",
            flush=True,
        )
        end_epoch = start_epoch

    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        losses = []
        if not uniform_eval_only:
            for key, loader in _make_loaders(train_buckets, args.batch_size, shuffle=True):
                for batch in loader:
                    batch = batch.to(device)
                    with torch.no_grad():
                        expert_preds = bank.predict_homogeneous_batch(batch, key)
                        gate_feat = _encode_gate_feat(bank, batch, key, model.gate_strategy)
                    fused, w = model.fuse(
                        gate_feat, expert_preds, pragma_feat=_batch_pragma(batch)
                    )
                    loss, _ = model.loss_from_fused(fused, batch, weights=w, epoch=epoch)
                    if not loss.requires_grad:
                        losses.append(float(loss.detach().cpu()))
                        continue
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(losses)) if losses else float("nan")
        query_m = evaluate_selector(
            model, bank, query_list, args.batch_size, device, target_list
        )
        shot_m = evaluate_selector(
            model, bank, shot_list, args.batch_size, device, target_list
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"shot_{kk}": vv for kk, vv in shot_m.items()},
            **{f"query_{kk}": vv for kk, vv in query_m.items()},
        }
        history.append(row)
        if epoch % int(args.print_every) == 0 or epoch == 1:
            print(
                f"[moe_cv10/ft] epoch={epoch} train_loss={train_loss:.4f} "
                f"{_format_metrics('shot', shot_m, target_list)} "
                f"{_format_metrics('query', query_m, target_list)}",
                flush=True,
            )
            print(f"[moe_cv10/ft] epoch={epoch} {model.format_gate_weights()}", flush=True)

        payload = {
            **_selector_state_dict(model),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "shot_metrics": shot_m,
            "query_metrics": query_m,
            "target": target_list,
            "split": split_info,
            "ckpt_kind": "few_shot_train",
        }
        torch.save({**payload, "ckpt_kind": "last"}, last_path)
        if np.isfinite(train_loss) and train_loss < best_train:
            prev = best_train
            best_train = train_loss
            torch.save(payload, best_path)
            print(
                f"[moe_cv10/ft] epoch={epoch} >>> 更新了 train_model.pth "
                f"(train_loss: {prev:.4f} -> {best_train:.4f})",
                flush=True,
            )
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f)

    ckpt = torch.load(best_path, map_location=device) if os.path.isfile(best_path) else None
    if ckpt is not None:
        _load_selector_ckpt(model, ckpt)
    query_final = evaluate_selector(
        model, bank, query_list, args.batch_size, device, target_list
    )
    h2h = build_head2head(
        base_q,
        query_final,
        target_list,
        "query",
        beat_margin=float(getattr(args, "beat_margin", 0.0)),
    )
    report = (
        "[moe_cv10/ft] zero-shot "
        + _format_metrics("query", zero_m, target_list)
        + "\n"
        + format_head2head_table(h2h, target_list)
        + "\n"
    )
    print("\n[moe_cv10/ft] === FINAL (query = unseen configs of test_10) ===", flush=True)
    print(report, flush=True)
    with open(join(save_dir, "head2head_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(join(save_dir, "head2head.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"zero_shot_query": zero_m, "few_shot_query": h2h},
            f,
            indent=2,
        )
    result = {
        "finetune": True,
        "transfer_k_shot": k,
        "coreset": coreset,
        "pretrained": pretrained,
        "best_train_loss": best_train if np.isfinite(best_train) else None,
        "best_epoch": None if ckpt is None else ckpt.get("epoch"),
        "zero_shot_query": zero_m,
        "few_shot_query": query_final,
        "head2head_query": h2h,
        "n_shot": len(shot_list),
        "n_query": len(query_list),
        "checkpoint": best_path if os.path.isfile(best_path) else None,
        "split": split_info,
    }
    with open(join(save_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(
        f"[moe_cv10/ft] DONE epoch={result['best_epoch']} "
        f"train_loss={best_train:.4f} | "
        f"{_format_metrics('query', query_final, target_list)}",
        flush=True,
    )
    return result


def train_moe_cv10(args) -> dict:
    device = _device(args.device)
    save_dir = args.save_dir
    create_dir_if_not_exists(save_dir)
    best_path = join(save_dir, "best.pth")
    last_path = join(save_dir, "last.pth")
    history_path = join(save_dir, "history.json")
    target_list = list(args.target)

    split_payload = load_kernel_split(args.kernel_split_json)
    test_10 = list(split_payload["test_10"])
    if "train_30" in split_payload:
        train_30 = list(split_payload["train_30"])
    else:
        folds_payload = load_folds_json(args.folds_json)
        train_30 = list(folds_payload["train_30"])

    if getattr(args, "finetune", False):
        return finetune_moe_cv10(args, train_30, test_10)

    if getattr(args, "gate_strategy", "legacy") == "selector":
        train_names, val_names = split_family_kernels(train_30, ratio=args.train_ratio, seed=args.seed)
    else:
        train_names, val_names = split_kernels_7_3(train_30, ratio=args.train_ratio, seed=args.seed)
    split_info = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "expert_source": getattr(args, "expert_source", "cv10"),
        "gate_mode": getattr(args, "gate_mode", "free"),
        "mix_mode": getattr(args, "mix_mode", "soft"),
        "prior_ast_on_perf": getattr(args, "prior_ast_on_perf", 0.5),
        "prior_text_on_util": getattr(args, "prior_text_on_util", 0.5),
        "use_prior": getattr(args, "use_prior", True),
        "min_special": getattr(args, "min_special", 0.0),
        "balance_coef": getattr(args, "balance_coef", 0.01),
        "perf_loss_weight": getattr(args, "perf_loss_weight", 3.0),
        "use_pred_aware": getattr(args, "use_pred_aware", True),
        "pred_slot_dim": getattr(args, "pred_slot_dim", 16),
        "use_residual": getattr(args, "use_residual", False),
        "patience": getattr(args, "patience", 0),
        "gate_input_cond": True,
        "train_30": train_30,
        "train_kernels": train_names,
        "val_kernels": val_names,
        "test_10": test_10,
        "n_train": len(train_names),
        "n_val": len(val_names),
        "n_test": len(test_10),
    }
    split_info["gate_strategy"] = getattr(args, "gate_strategy", "legacy")
    if split_info["gate_strategy"] == "selector":
        split_info["selector_config"] = selector_config_from_args(args)
        split_info["family_split"] = True
    with open(join(save_dir, "moe_kernel_split.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    print(
        f"[moe_cv10] kernels train/val/test = "
        f"{len(train_names)}/{len(val_names)}/{len(test_10)}",
        flush=True,
    )

    all_kernels = list(dict.fromkeys(train_30 + test_10))
    from multimodal_moe.moe_system.utils_new import (
        default_data_paths,
        normalize_dataset_mode,
        set_dataset_mode,
    )

    ds_mode = normalize_dataset_mode(getattr(args, "dataset_mode", "filtered"))
    set_dataset_mode(ds_mode)
    skip_zero = getattr(args, "skip_all_zero_config", None)
    if skip_zero is None:
        skip_zero = ds_mode == "filtered"
    paths = default_data_paths(ds_mode)
    print(
        f"[moe_cv10] dataset_mode={ds_mode} skip_all_zero={bool(skip_zero)} "
        f"cdfg={paths['cdfg_root']}",
        flush=True,
    )
    data_all = build_dataset(
        target_list=target_list,
        skip_all_zero=bool(skip_zero),
        kernels=all_kernels,
        dataset_mode=ds_mode,
    )
    split_info["dataset_mode"] = ds_mode
    split_info["skip_all_zero_config"] = bool(skip_zero)
    train_list = filter_by_kernels(data_all, train_names)
    val_list = filter_by_kernels(data_all, val_names)
    test_list = filter_by_kernels(data_all, test_10)
    print(
        f"[moe_cv10] samples train/val/test = "
        f"{len(train_list)}/{len(val_list)}/{len(test_list)}",
        flush=True,
    )
    if not train_list:
        raise RuntimeError("训练集为空")

    use_pragma, pragma_in_dim, pragma_enc_path = _prepare_pragma_features(
        args, train_list, val_list, test_list, save_dir
    )
    split_info["gate_use_pragma"] = use_pragma
    split_info["pragma_in_dim"] = pragma_in_dim
    split_info["pragma_proj_dim"] = int(getattr(args, "pragma_proj_dim", 64))
    split_info["pragma_encoder"] = pragma_enc_path
    with open(join(save_dir, "moe_kernel_split.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print(
        f"[moe_cv10] expert_source={getattr(args, 'expert_source', 'cv10')} "
        f"train_ratio={args.train_ratio} "
        f"mix_mode={getattr(args, 'mix_mode', 'soft')} "
        f"gate_mode={getattr(args, 'gate_mode', 'free')} "
        f"min_special={getattr(args, 'min_special', 0.0)} "
        f"pred_aware={getattr(args, 'use_pred_aware', True)} "
        f"residual={getattr(args, 'use_residual', False)} "
        f"perf_loss_weight={getattr(args, 'perf_loss_weight', 3.0)} "
        f"balance_coef={getattr(args, 'balance_coef', 0.01)} "
        f"patience={getattr(args, 'patience', 0)} "
        f"gate_use_pragma={use_pragma}",
        flush=True,
    )
    print("[moe_cv10] loading frozen AblationNet experts...", flush=True)
    bank = Cv10AblationExpertBank(
        device=device,
        cv10_root=args.cv10_root,
        ablation_root=args.ablation_root,
        folds_json=args.folds_json,
        kernel_split_json=args.kernel_split_json,
        target_list=target_list,
        preload_all=True,
        expert_source=getattr(args, "expert_source", "cv10"),
    )

    # 同协议单专家基线：与 MoE 对照的尺子（先定最强单专家，再谈门控是否有效）
    print("[moe_cv10] === expert baselines (same val/test as MoE) ===", flush=True)
    expert_baselines: Dict[str, Dict[str, Dict[str, float]]] = {}
    beat_margin = float(getattr(args, "beat_margin", 0.0))
    ckpt_select = str(getattr(args, "ckpt_select", "beat_first"))
    for split_name, split_list in (("val", val_list), ("test", test_list)):
        base = evaluate_experts_alone(
            bank, split_list, args.batch_size, device, target_list
        )
        expert_baselines[split_name] = base
        for name, m in base.items():
            print(
                f"[moe_cv10] baseline_{split_name}/{name} "
                f"{_format_metrics(split_name, m, target_list)}",
                flush=True,
            )
        be_name, be_sum5 = _best_single_expert(base, target_list)
        print(
            f"[moe_cv10] baseline_{split_name}/BEST_SINGLE={be_name} "
            f"sum5={be_sum5:.4f}  (MoE 须明显低于此才算有效)",
            flush=True,
        )
        with open(
            join(save_dir, f"expert_baseline_{split_name}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(base, f, indent=2)
    val_best_expert_name, val_best_expert_sum5 = _best_single_expert(
        expert_baselines["val"], target_list
    )
    print(
        f"[moe_cv10] ckpt_select={ckpt_select} beat_margin={beat_margin} "
        f"val_best_expert={val_best_expert_name} sum5={val_best_expert_sum5:.4f}",
        flush=True,
    )

    model = _build_selector(args, bank, target_list, use_pragma, pragma_in_dim, device)
    opt = torch.optim.Adam(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")  # sum5 或 gap 选模用的主分数（越小越好；gap 模式存 -gap）
    best_select_score = float("inf")
    ever_beat_expert = False
    # 每个指标单独记最优 val MSE，并写 best_{metric}.pth
    best_per_target = {t: float("inf") for t in target_list}
    best_per_target_beat = {t: False for t in target_list}
    best_epoch_per_target = {t: None for t in target_list}
    for t in target_list:
        mp = _metric_ckpt_path(save_dir, t)
        if os.path.isfile(mp):
            try:
                mck = torch.load(mp, map_location="cpu")
                best_per_target[t] = float(
                    mck.get(
                        "best_val_mse_target",
                        mck.get("best_val_rmse_target", float("inf")),
                    )
                )
                best_epoch_per_target[t] = mck.get("epoch")
                best_per_target_beat[t] = bool(mck.get("beats_best_expert", False))
            except Exception:
                pass
    start_epoch = 1
    end_epoch = int(args.epoch_num)
    history: List[dict] = []
    resumed_from = None

    # 默认从上次 best.pth 接着训；仅 --no_resume 才从零开始
    if getattr(args, "resume", True) and os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device)
        try:
            _load_selector_ckpt(model, ckpt)
        except RuntimeError as e:
            if model.gate_strategy == "selector":
                raise RuntimeError(
                    "selector checkpoint 与当前门控不兼容；请使用独立 --save_dir，"
                    "或保持源域结构参数一致。" + str(e)
                ) from e
            print(
                f"[moe_cv10] WARNING checkpoint incompatible; "
                f"ignore resume and train from scratch.\n  detail: {e}",
                flush=True,
            )
            ckpt = None
        if ckpt is not None:
            if ckpt.get("optimizer") is not None:
                try:
                    opt.load_state_dict(ckpt["optimizer"])
                except Exception as e:
                    print(f"[moe_cv10] WARNING optimizer not loaded: {e}", flush=True)
            best_val = float(
                ckpt.get(
                    "best_val_mse_sum5",
                    ckpt.get("best_val_rmse_sum5", float("inf")),
                )
            )
            best_select_score = float(
                ckpt.get("best_select_score", best_val)
            )
            ever_beat_expert = bool(ckpt.get("ever_beat_expert", False))
            ckpt_epoch = int(ckpt.get("epoch", 0))
            resumed_from = best_path
            start_epoch = ckpt_epoch + 1
            # 未跑满 epoch_num → 补完；已跑满 → 再训一整轮 epoch_num（仍从 best 权重起步）
            if start_epoch <= int(args.epoch_num):
                end_epoch = int(args.epoch_num)
            else:
                end_epoch = ckpt_epoch + int(args.epoch_num)
            if os.path.isfile(history_path):
                try:
                    with open(history_path, encoding="utf-8") as f:
                        history = json.load(f)
                except Exception:
                    history = []
            print(
                f"[moe_cv10] RESUME from {best_path} "
                f"(ckpt_epoch={ckpt_epoch}) → train epochs "
                f"{start_epoch}..{end_epoch} | best_val={best_val:.4f} "
                f"select_score={best_select_score:.4f} beat_expert={ever_beat_expert}",
                flush=True,
            )
        else:
            print("[moe_cv10] train gate from scratch", flush=True)
    elif not getattr(args, "resume", True) and os.path.isfile(best_path):
        print(
            f"[moe_cv10] --no_resume: ignore existing {best_path}, train from scratch",
            flush=True,
        )
    else:
        print("[moe_cv10] no best.pth yet; train gate from scratch", flush=True)

    print(
        "[moe_cv10] per-target ckpt: "
        + ", ".join(_metric_ckpt_name(t) for t in target_list)
        + " (+ best.pth for sum5)",
        flush=True,
    )

    _fit_selector_statistics(model, bank, train_list, args.batch_size, device)
    train_buckets = _bucket_by_route(train_list, bank)
    mldg_available = len({kernel_family(str(sample.kernel)) for sample in train_list}) >= 2
    if model.gate_strategy == "selector" and not mldg_available:
        print("[moe_cv10/selector] 训练集只有一个家族，使用监督训练（无法构造跨家族 MLDG）", flush=True)
    # 固定 1/3：专家冻结且权重是常数，loss 没有 grad_fn，反传会报
    # "does not require grad and does not have a grad_fn"。指标与训练轮数无关。
    uniform_eval_only = str(getattr(model, "mix_mode", "soft")) == "uniform"
    if uniform_eval_only:
        print(
            "[moe_cv10] mix_mode=uniform: 三专家固定各 1/3，"
            "跳过反传，只评估一次并写入 best.pth",
            flush=True,
        )
        end_epoch = start_epoch

    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        # bank 始终 eval / no_grad
        losses = []
        stage_metrics = {}
        use_mldg = (
            not uniform_eval_only
            and model.gate_strategy == "selector"
            and mldg_available
            and epoch > int(model.selector_config["supervised_epochs"])
        )
        if uniform_eval_only:
            pass
        elif use_mldg:
            losses, stage_metrics = _train_selector_mldg_epoch(
                model, bank, train_list, opt, device, epoch, args.seed
            )
        else:
            auxiliary_parts = defaultdict(list)
            for key, loader in _make_loaders(train_buckets, args.batch_size, shuffle=True):
                for batch in loader:
                    batch = batch.to(device)
                    with torch.no_grad():
                        expert_preds = bank.predict_homogeneous_batch(batch, key)
                        gate_feat = _encode_gate_feat(bank, batch, key, model.gate_strategy)
                    fused, w = model.fuse(
                        gate_feat, expert_preds, pragma_feat=_batch_pragma(batch)
                    )
                    loss, parts = model.loss_from_fused(fused, batch, weights=w, epoch=epoch)
                    if not loss.requires_grad:
                        losses.append(float(loss.detach().cpu()))
                        continue
                    opt.zero_grad()
                    loss.backward()
                    if model.gate_strategy == "selector":
                        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                        for name, value in parts.items():
                            if name.startswith("selector_"):
                                auxiliary_parts[name].append(float(value.detach()))
                    opt.step()
                    losses.append(float(loss.detach().cpu()))
            if model.gate_strategy == "selector":
                stage_metrics = {name: float(np.mean(values)) for name, values in auxiliary_parts.items()}
                stage_metrics["train_stage"] = "supervised"

        val_m = evaluate_selector(
            model, bank, val_list, args.batch_size, device, target_list
        )
        test_m = evaluate_selector(
            model, bank, test_list, args.batch_size, device, target_list
        )
        val_score = val_m.get("mse_sum_targets", 1e9)
        train_loss = float(np.mean(losses)) if losses else float("nan")

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_m.items()},
            **stage_metrics,
            **{f"test_{k}": v for k, v in test_m.items()},
            "gate_weights": model.gate.weight_summary(
                target_list=target_list,
                expert_names=("CDFG", "AST", "Text"),
            ),
        }
        history.append(row)

        if epoch % args.print_every == 0 or epoch == start_epoch:
            print(
                f"[moe_cv10] epoch={epoch} train_loss={train_loss:.4f} "
                f"{_format_metrics('val', val_m, target_list)} "
                f"{_format_metrics('test', test_m, target_list)}",
                flush=True,
            )
            print(f"[moe_cv10] epoch={epoch} {model.format_gate_weights()}", flush=True)
            if stage_metrics:
                print(f"[moe_cv10/selector] {stage_metrics}", flush=True)

        gate_w = model.gate.weight_summary(
            target_list=target_list,
            expert_names=("CDFG", "AST", "Text"),
        )
        ckpt_base = {
            **_selector_state_dict(model),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "target": target_list,
            "split": split_info,
            "val_metrics": val_m,
            "test_metrics": test_m,
            "gate_weights": gate_w,
            "expert_modes": list(EXPERT_MODES),
            **stage_metrics,
        }

        # 分指标：优先「打过该指标最强单专家」；否则仅在尚未 beat 时用最低 MSE 兜底
        for t in target_list:
            key = f"mse_{t}"
            score_t = float(val_m.get(key, float("inf")))
            if not np.isfinite(score_t):
                continue
            be_name_t, be_v_t = _best_single_expert(
                expert_baselines["val"], target_list, key
            )
            beats_t = score_t + beat_margin < be_v_t
            better = False
            if beats_t and (
                (not best_per_target_beat[t]) or score_t < best_per_target[t]
            ):
                better = True
            elif (not beats_t) and (not best_per_target_beat[t]) and score_t < best_per_target[t]:
                better = True
            if not better:
                continue
            prev_t = best_per_target[t]
            best_per_target[t] = score_t
            if beats_t:
                best_per_target_beat[t] = True
            best_epoch_per_target[t] = epoch
            t_path = _metric_ckpt_path(save_dir, t)
            torch.save(
                {
                    **ckpt_base,
                    "best_val_mse_sum5": float(val_score),
                    "best_val_mse_target": score_t,
                    "select_target": t,
                    "ckpt_kind": "per_target",
                    "beats_best_expert": beats_t,
                    "best_single_expert": be_name_t,
                    "best_single_mse": float(be_v_t),
                },
                t_path,
            )
            tag = "beat_expert" if beats_t else "mse_only"
            print(
                f"[moe_cv10] epoch={epoch} >>> 更新了 {_metric_ckpt_name(t)} ({tag}) "
                f"(val_{t}: {prev_t:.4f} -> {score_t:.4f}; "
                f"best_single={be_name_t}:{be_v_t:.4f})",
                flush=True,
            )

        # best.pth：按 ckpt_select 与最强单专家对照
        beats_sum5 = bool(
            np.isfinite(val_score)
            and np.isfinite(val_best_expert_sum5)
            and val_score + beat_margin < val_best_expert_sum5
        )
        gap = (
            float(val_best_expert_sum5 - val_score)
            if np.isfinite(val_best_expert_sum5) and np.isfinite(val_score)
            else float("-inf")
        )
        if ckpt_select == "sum5":
            select_score = val_score
            allow = True
        elif ckpt_select == "gap":
            # 越大越好 → 存 -gap 使「越小越好」统一
            select_score = -gap
            allow = True
        else:  # beat_first
            if beats_sum5:
                select_score = val_score
                allow = True
            elif not ever_beat_expert:
                # 尚未出现过打过专家的 epoch：暂用 sum5 兜底
                select_score = val_score
                allow = True
            else:
                # 已有打过专家的 best：不再用未打过的 epoch 覆盖
                allow = False
                select_score = float("inf")

        updated_best = False
        first_beat = bool(ckpt_select == "beat_first" and beats_sum5 and not ever_beat_expert)
        if allow and np.isfinite(select_score) and (
            first_beat or select_score < best_select_score
        ):
            prev = best_val
            best_val = val_score
            best_select_score = select_score if not first_beat else val_score
            if beats_sum5:
                ever_beat_expert = True
            torch.save(
                {
                    **ckpt_base,
                    "best_val_mse_sum5": best_val,
                    "best_select_score": best_select_score,
                    "ckpt_kind": ckpt_select,
                    "ckpt_select": ckpt_select,
                    "beats_best_expert": beats_sum5,
                    "ever_beat_expert": ever_beat_expert,
                    "best_single_expert": val_best_expert_name,
                    "best_single_sum5": float(val_best_expert_sum5),
                    "gap_expert_minus_moe": gap if np.isfinite(gap) else None,
                    "beat_margin": beat_margin,
                },
                best_path,
            )
            updated_best = True
            print(
                f"[moe_cv10] epoch={epoch} >>> 更新了 best.pth "
                f"(select={ckpt_select}, val_sum5: {prev:.4f} -> {best_val:.4f}, "
                f"beats_expert={beats_sum5}, first_beat={first_beat}, gap={gap:.4f}, "
                f"best_single={val_best_expert_name}:{val_best_expert_sum5:.4f})",
                flush=True,
            )
            h2h_val = build_head2head(
                expert_baselines["val"],
                val_m,
                target_list,
                "val",
                beat_margin=beat_margin,
            )
            print(format_head2head_table(h2h_val, target_list), flush=True)
            print(
                f"[moe_cv10] epoch={epoch} best_gate {model.format_gate_weights()}",
                flush=True,
            )

        # last.pth：无条件覆盖为当前 epoch（跑到 500 即为第 500 轮）
        torch.save(
            {
                **ckpt_base,
                "best_val_mse_sum5": float(val_score),
                "best_select_score": best_select_score,
                "ckpt_kind": "last",
                "ckpt_select": ckpt_select,
                "beats_best_expert": beats_sum5,
                "ever_beat_expert": ever_beat_expert,
                "best_single_expert": val_best_expert_name,
                "best_single_sum5": float(val_best_expert_sum5),
                "gap_expert_minus_moe": gap if np.isfinite(gap) else None,
                "beat_margin": beat_margin,
            },
            last_path,
        )
        if epoch == end_epoch or epoch % int(args.print_every) == 0:
            print(
                f"[moe_cv10] epoch={epoch} >>> 保存了 last.pth",
                flush=True,
            )

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f)

    # ---- 训练结束：汇总 sum5 + 分指标 ckpt ----
    per_target_ckpts = {}
    for t in target_list:
        t_path = _metric_ckpt_path(save_dir, t)
        if not os.path.isfile(t_path):
            continue
        t_ckpt = torch.load(t_path, map_location=device)
        _load_selector_ckpt(model, t_ckpt)
        t_test = evaluate_selector(
            model, bank, test_list, args.batch_size, device, target_list
        )
        per_target_ckpts[t] = {
            "path": t_path,
            "best_epoch": t_ckpt.get("epoch"),
            "best_val_mse_target": float(t_ckpt.get("best_val_mse_target", float("nan"))),
            "test_mse_target": t_test.get(f"mse_{t}"),
            "test_mse_sum5": t_test.get("mse_sum_targets"),
            "test_metrics": t_test,
        }
        print(
            f"[moe_cv10] DONE/{_metric_ckpt_name(t)} "
            f"best_epoch={t_ckpt.get('epoch')} "
            f"val_{t}={t_ckpt.get('best_val_mse_target')} "
            f"test_{t}={t_test.get(f'mse_{t}', float('nan')):.4f} "
            f"test_sum5={t_test.get('mse_sum_targets', float('nan')):.4f}",
            flush=True,
        )

    head2head_val = None
    head2head_test = None
    if not os.path.isfile(best_path):
        print(
            "[moe_cv10] WARNING 无 best.pth（选模从未更新）；"
            "result 以分指标 ckpt 为准",
            flush=True,
        )
        test_m = {}
        val_m = {}
        best_epoch = None
        gate_w_done = None
    else:
        ckpt = torch.load(best_path, map_location=device)
        try:
            _load_selector_ckpt(model, ckpt)
        except RuntimeError as e:
            raise RuntimeError(
                f"best.pth 与当前模型结构不兼容，请加 --no_resume 重新训练。\n{e}"
            ) from e
        test_m = evaluate_selector(
            model, bank, test_list, args.batch_size, device, target_list
        )
        val_m = evaluate_selector(
            model, bank, val_list, args.batch_size, device, target_list
        )
        best_epoch = ckpt.get("epoch")
        gate_w_done = model.gate.weight_summary(
            target_list=target_list,
            expert_names=("CDFG", "AST", "Text"),
        )
        print(
            f"[moe_cv10] DONE best.pth epoch={best_epoch} "
            f"best_val_sum5={best_val:.4f} | {_format_metrics('test', test_m, target_list)}",
            flush=True,
        )
        print(f"[moe_cv10] DONE {model.format_gate_weights()}", flush=True)

        # 核心对照：同一 val/test 上单专家 vs MoE(best)
        head2head_val = build_head2head(
            expert_baselines["val"], val_m, target_list, "val", beat_margin=beat_margin
        )
        head2head_test = build_head2head(
            expert_baselines["test"], test_m, target_list, "test", beat_margin=beat_margin
        )
        report_txt = (
            format_head2head_table(head2head_val, target_list)
            + "\n\n"
            + format_head2head_table(head2head_test, target_list)
            + "\n"
        )
        print("\n[moe_cv10] === FINAL CONTRAST (same split as MoE) ===", flush=True)
        print(report_txt, flush=True)
        with open(join(save_dir, "head2head_report.txt"), "w", encoding="utf-8") as f:
            f.write(report_txt)
        with open(join(save_dir, "head2head.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"val": head2head_val, "test": head2head_test},
                f,
                indent=2,
            )
        if not head2head_test["verdict"]["moe_effective"]:
            print(
                "[moe_cv10] WARNING: MoE(best) 在 test 上未低于最强单专家 → "
                "门控无效/无增益；勿只看 val_sum5。",
                flush=True,
            )

    index = {
        "sum5": {
            "path": best_path if os.path.isfile(best_path) else None,
            "kind": ckpt_select,
            "ever_beat_expert": ever_beat_expert,
        },
        "last": {
            "path": last_path if os.path.isfile(last_path) else None,
            "kind": "last",
            "epoch": end_epoch if os.path.isfile(last_path) else None,
        },
        "per_target": {
            t: {
                "path": _metric_ckpt_path(save_dir, t),
                "file": _metric_ckpt_name(t),
                "best_epoch": best_epoch_per_target.get(t),
                "best_val_mse": best_per_target.get(t),
            }
            for t in target_list
        },
    }
    with open(join(save_dir, "checkpoints_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    result = {
        "best_val_mse_sum5": best_val if np.isfinite(best_val) else None,
        "best_epoch": best_epoch,
        "resumed_from": resumed_from,
        "train_epoch_range": [start_epoch, end_epoch],
        "ckpt_select": ckpt_select,
        "beat_margin": beat_margin,
        "ever_beat_expert": ever_beat_expert,
        "val_best_single_expert": val_best_expert_name,
        "val_best_single_sum5": float(val_best_expert_sum5),
        "val_metrics": val_m,
        "test_metrics": test_m,
        "gate_weights": gate_w_done,
        "head2head_val": head2head_val,
        "head2head_test": head2head_test,
        "moe_effective_test": (
            None
            if head2head_test is None
            else head2head_test["verdict"]["moe_effective"]
        ),
        "n_train": len(train_list),
        "n_val": len(val_list),
        "n_test": len(test_list),
        "checkpoint": best_path if os.path.isfile(best_path) else None,
        "last_checkpoint": last_path if os.path.isfile(last_path) else None,
        "checkpoints_per_target": per_target_ckpts,
        "best_epoch_per_target": best_epoch_per_target,
        "best_val_per_target": {
            t: (float(v) if np.isfinite(v) else None) for t, v in best_per_target.items()
        },
        "checkpoints_index": index,
        "split": split_info,
    }
    with open(join(save_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


@torch.no_grad()
def inference_moe_cv10(args) -> dict:
    device = _device(args.device)
    if getattr(args, "model_path", None):
        best_path = args.model_path
    elif getattr(args, "select_target", None):
        best_path = _metric_ckpt_path(args.save_dir, args.select_target)
    else:
        best_path = join(args.save_dir, "best.pth")
    if not os.path.isfile(best_path):
        raise FileNotFoundError(best_path)
    print(f"[moe_cv10/infer] load {best_path}", flush=True)
    ckpt = torch.load(best_path, map_location=device)
    target_list = list(ckpt.get("target") or DEFAULT_TARGETS)
    split = ckpt.get("split") or {}
    test_10 = list(split.get("test_10") or load_kernel_split(args.kernel_split_json)["test_10"])
    val_names = list(split.get("val_kernels") or [])

    bank = Cv10AblationExpertBank(
        device=device,
        cv10_root=args.cv10_root,
        ablation_root=args.ablation_root,
        folds_json=args.folds_json,
        kernel_split_json=args.kernel_split_json,
        target_list=target_list,
        preload_all=True,
        expert_source=str(
            split.get("expert_source", getattr(args, "expert_source", "cv10"))
        ),
    )
    use_pragma = bool(split.get("gate_use_pragma", getattr(args, "gate_use_pragma", False)))
    pragma_in_dim = int(split.get("pragma_in_dim", 0))
    pragma_proj_dim = int(split.get("pragma_proj_dim", getattr(args, "pragma_proj_dim", 64)))
    model = FrozenExpertSelector(
        bank,
        target_list=target_list,
        prior_ast_on_perf=float(split.get("prior_ast_on_perf", getattr(args, "prior_ast_on_perf", 0.5))),
        prior_text_on_util=float(split.get("prior_text_on_util", getattr(args, "prior_text_on_util", 0.5))),
        use_prior=bool(split.get("use_prior", getattr(args, "use_prior", True))),
        min_special=float(split.get("min_special", getattr(args, "min_special", 0.0))),
        balance_coef=float(split.get("balance_coef", getattr(args, "balance_coef", 0.01))),
        gate_mode=str(split.get("gate_mode", getattr(args, "gate_mode", "free"))),
        mix_mode=str(split.get("mix_mode", getattr(args, "mix_mode", "soft"))),
        use_pred_aware=bool(split.get("use_pred_aware", getattr(args, "use_pred_aware", True))),
        use_residual=bool(split.get("use_residual", getattr(args, "use_residual", False))),
        perf_loss_weight=float(split.get("perf_loss_weight", getattr(args, "perf_loss_weight", 3.0))),
        pred_slot_dim=int(split.get("pred_slot_dim", getattr(args, "pred_slot_dim", 16))),
        use_pragma=use_pragma,
        pragma_in_dim=pragma_in_dim,
        pragma_proj_dim=pragma_proj_dim,
        gate_strategy=_checkpoint_strategy(ckpt),
        selector_config=ckpt.get("selector_config", split.get("selector_config", DEFAULT_SELECTOR_CONFIG)),
    ).to(device)
    try:
        _load_selector_ckpt(model, ckpt)
    except RuntimeError as e:
        raise RuntimeError(
            f"checkpoint 与当前模型结构不兼容，请先重新 train。\n{e}"
        ) from e

    from multimodal_moe.moe_system.utils_new import normalize_dataset_mode, set_dataset_mode

    ds_mode = normalize_dataset_mode(
        split.get("dataset_mode", getattr(args, "dataset_mode", "filtered"))
    )
    set_dataset_mode(ds_mode)
    skip_zero = split.get("skip_all_zero_config", getattr(args, "skip_all_zero_config", None))
    if skip_zero is None:
        skip_zero = ds_mode == "filtered"
    data_all = build_dataset(
        target_list=target_list,
        skip_all_zero=bool(skip_zero),
        kernels=test_10 + val_names,
        dataset_mode=ds_mode,
    )
    test_list = filter_by_kernels(data_all, test_10)
    if use_pragma:
        # Attach the original source-domain pragma vocabulary below.
        from multimodal_moe.moe_system.pragma_features import PragmaFeatureEncoder
        from multimodal_moe.moe_system.pragma_io import (
            attach_pragma_features,
            default_c_src_roots,
            default_points_roots,
        )

        enc_path = split.get("pragma_encoder") or join(dirname(best_path), "pragma_encoder.json")
        if not os.path.isfile(enc_path):
            raise FileNotFoundError(f"缺少 pragma_encoder.json: {enc_path}")
        enc = PragmaFeatureEncoder.load(enc_path)
        points_roots = list(default_points_roots())
        if getattr(args, "points_root", None):
            points_roots = [args.points_root] + [
                r for r in points_roots if r != args.points_root
            ]
        c_roots = list(default_c_src_roots())
        if getattr(args, "c_src_root", None):
            c_roots = [args.c_src_root] + [r for r in c_roots if r != args.c_src_root]
        st = attach_pragma_features(
            test_list, enc, points_roots=points_roots, c_src_roots=c_roots
        )
        print(
            f"[moe_cv10/infer] pragma attach found/missing={st['found']}/{st['missing']} dim={st['dim']}",
            flush=True,
        )
    if model.gate_strategy == "selector" and split.get("finetune"):
        # A finetuned checkpoint is evaluated on held-out configs, not its shots.
        shot_ids = {kernel: set(stats.get("cfg_ids", []))
                    for kernel, stats in split.get("per_kernel", {}).items()}
        if not shot_ids:
            raise ValueError("selector 微调 checkpoint 缺少 support/query 划分")
        test_list = [sample for sample in test_list
                     if int(sample.cfg_id) not in shot_ids.get(str(sample.kernel), set())]
    test_m = evaluate_selector(model, bank, test_list, args.batch_size, device, target_list)
    print(
        f"[moe_cv10/infer] {_format_metrics('test', test_m, target_list, include_rmse=True)}",
        flush=True,
    )
    print(f"[moe_cv10/infer] {model.format_gate_weights()}", flush=True)
    out = {
        "test_metrics": test_m,
        "checkpoint": best_path,
        "gate_strategy": model.gate_strategy,
        "n_test": len(test_list),
        "gate_weights": model.gate.weight_summary(
            target_list=target_list,
            expert_names=("CDFG", "AST", "Text"),
        ),
    }
    create_dir_if_not_exists(args.save_dir)
    with open(join(args.save_dir, "inference_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out
