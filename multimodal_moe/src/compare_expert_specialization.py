"""
对比 cdfg_only / ast_only / LM 在不同 kernel、不同指标上的优劣。

用法（在 HierarchicalMoE-master 下）:
  python -m multimodal_moe.src.compare_expert_specialization
  python -m multimodal_moe.src.compare_expert_specialization --dataset_mode not_0 --split test
  python -m multimodal_moe.src.compare_expert_specialization --ablation_root .../ablation_outputs

输出目录默认: {ablation_root}/expert_compare/
  - per_kernel_target_rmse.csv   每个 kernel×target×mode 的 RMSE
  - win_by_target.json           各指标上谁赢、赢几次
  - win_by_kernel.json           各核上谁赢（按五指标 RMSE 和）
  - summary.json                 总表 + 胜率
  - report.txt                   人可读摘要
"""
from __future__ import annotations

import argparse
import csv
import json
import os
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
for _p in (_HMOE, _MM, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODES = ("cdfg_only", "ast_only", "LM")
DEFAULT_TARGETS = ["perf", "util-LUT", "util-FF", "util-DSP", "util-BRAM"]


def parse_args():
    p = argparse.ArgumentParser(description="Compare single-modal expert specialization")
    p.add_argument(
        "--ablation_root",
        default=None,
        help="含 {cdfg_only,ast_only,LM}/best.pt 的目录；默认随 dataset_mode",
    )
    p.add_argument(
        "--dataset_mode",
        default="filtered",
        choices=["filtered", "not_0", "del_0", "keep_zero"],
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["test", "val", "train", "all_hold"],
        help="test=test_10；val=val_6；train=train_24；all_hold=train_30+test_10",
    )
    p.add_argument("--seed", type=int, default=None, help="默认读 kernel_split.json 或 50")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--target",
        nargs="+",
        default=DEFAULT_TARGETS,
    )
    p.add_argument(
        "--ckpt_name",
        default="best.pt",
        help="每个模态子目录下的权重文件名",
    )
    p.add_argument("--out_dir", default=None, help="默认 {ablation_root}/expert_compare")
    return p.parse_args()


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    if pred.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def _load_model(ckpt_path: str, device: torch.device, target_list: Sequence[str]):
    from multimodal_moe.src.model_ablation import AblationNet

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"缺少权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    mode = ckpt.get("ablation_mode") or os.path.basename(os.path.dirname(ckpt_path))
    targets = list(ckpt.get("target") or target_list)
    in_dim = int(ckpt["in_dim"])
    model = AblationNet(
        in_cdfg=in_dim,
        target_list=targets,
        ablation_mode=mode,
        D=int(ckpt.get("D", 128)),
        hidden_num=int(ckpt.get("hidden_num", ckpt.get("D", 128))),
        gnn_layers=int(ckpt.get("gnn_layers", 5)),
        dropout=float(ckpt.get("dropout", 0.2)),
        out_dim=1,
        gnn_type=str(ckpt.get("gnn_type", "gcn")),
        gnn_activation=str(ckpt.get("gnn_activation", "relu")),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, targets, ckpt


@torch.no_grad()
def collect_preds(
    model,
    loader,
    device: torch.device,
    target_list: Sequence[str],
) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """
    返回:
      by_kernel[kernel][target] = {"pred": [...], "true": [...]}  (list of 1d arrays)
    """
    buckets: Dict[str, Dict[str, Dict[str, List]]] = defaultdict(
        lambda: {t: {"pred": [], "true": []} for t in target_list}
    )
    for batch in loader:
        batch = batch.to(device)
        out, _, _ = model(batch)
        # PyG Batch: 字符串属性 → list
        kernels = list(batch.kernel) if hasattr(batch, "kernel") else None
        if kernels is None:
            raise RuntimeError("batch 缺少 kernel 字段，无法按核统计")
        n = len(kernels)
        for t in target_list:
            if t not in out:
                continue
            y = getattr(batch, t.replace("-", "_"), None)
            if y is None:
                continue
            pred = out[t].detach().cpu().view(-1).numpy()
            true = y.detach().cpu().view(-1).numpy()
            if pred.shape[0] != n or true.shape[0] != n:
                raise RuntimeError(
                    f"长度不一致: target={t} pred={pred.shape} true={true.shape} n={n}"
                )
            for i, k in enumerate(kernels):
                buckets[k][t]["pred"].append(np.array([pred[i]], dtype=np.float64))
                buckets[k][t]["true"].append(np.array([true[i]], dtype=np.float64))
    # concat
    out_map: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for k, td in buckets.items():
        out_map[k] = {}
        for t, d in td.items():
            if not d["pred"]:
                continue
            out_map[k][t] = {
                "pred": np.concatenate(d["pred"]),
                "true": np.concatenate(d["true"]),
            }
    return out_map


def rmse_table(
    by_kernel: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    target_list: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """kernel -> target -> rmse；另加 __ALL__ 全体样本。"""
    table: Dict[str, Dict[str, float]] = {}
    all_p = {t: [] for t in target_list}
    all_g = {t: [] for t in target_list}
    for k, td in by_kernel.items():
        table[k] = {}
        for t in target_list:
            if t not in td:
                table[k][t] = float("nan")
                continue
            r = _rmse(td[t]["pred"], td[t]["true"])
            table[k][t] = r
            all_p[t].append(td[t]["pred"])
            all_g[t].append(td[t]["true"])
    table["__ALL__"] = {}
    for t in target_list:
        if all_p[t]:
            table["__ALL__"][t] = _rmse(np.concatenate(all_p[t]), np.concatenate(all_g[t]))
        else:
            table["__ALL__"][t] = float("nan")
    return table


def sum5(row: Dict[str, float], target_list: Sequence[str]) -> float:
    vals = [row[t] for t in target_list if t in row and np.isfinite(row[t])]
    return float(sum(vals)) if vals else float("nan")


def main():
    args = parse_args()
    sys.argv = [sys.argv[0]]  # 避免后续误 import config 再 parse

    from multimodal_moe.moe_system.utils_new import (
        default_output_dirs,
        normalize_dataset_mode,
        set_dataset_mode,
    )
    from multimodal_moe.src.config_split import filter_by_kernels
    from multimodal_moe.src.data_ablation import AblationDataset, build_dataset, collate_ablation
    from multimodal_moe.src.kernel_protocol import build_protocol
    from multimodal_moe.src.utils import create_dir_if_not_exists

    ds = normalize_dataset_mode(args.dataset_mode)
    set_dataset_mode(ds)
    outs = default_output_dirs(ds)
    ablation_root = args.ablation_root or outs["ablation_root"]
    out_dir = args.out_dir or join(ablation_root, "expert_compare")
    create_dir_if_not_exists(out_dir)

    split_json = join(ablation_root, "kernel_split.json")
    if os.path.isfile(split_json):
        with open(split_json, encoding="utf-8") as f:
            proto = json.load(f)
        seed = int(args.seed if args.seed is not None else proto.get("seed", 50))
    else:
        seed = int(args.seed if args.seed is not None else 50)
        proto = build_protocol(seed=seed, save_json=split_json)

    train_30 = list(proto["train_30"])
    test_10 = list(proto["test_10"])
    train_24 = list(proto.get("train_24") or [])
    val_6 = list(proto.get("val_6") or [])
    if args.split == "test":
        kernels = test_10
        split_name = "test_10"
    elif args.split == "val":
        kernels = val_6 or train_30
        split_name = "val_6"
    elif args.split == "train":
        kernels = train_24 or train_30
        split_name = "train_24"
    else:
        kernels = list(dict.fromkeys(train_30 + test_10))
        split_name = "train_30+test_10"

    skip_zero = ds == "filtered"
    data_all = build_dataset(
        target_list=args.target,
        skip_all_zero=skip_zero,
        kernels=kernels,
        dataset_mode=ds,
    )
    data_list = filter_by_kernels(data_all, kernels)
    if not data_list:
        raise RuntimeError(f"划分 {split_name} 无样本")

    device = torch.device(
        args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu"
    )
    loader = DataLoader(
        AblationDataset(data_list),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_ablation,
    )

    print(
        f"[compare] dataset_mode={ds} ablation_root={ablation_root}\n"
        f"  split={split_name} kernels={len(kernels)} samples={len(data_list)} "
        f"seed={seed} device={device}",
        flush=True,
    )

    mode_tables: Dict[str, Dict[str, Dict[str, float]]] = {}
    mode_meta = {}
    for mode in MODES:
        ckpt_path = join(ablation_root, mode, args.ckpt_name)
        print(f"[compare] load {mode}: {ckpt_path}", flush=True)
        model, targets, ckpt = _load_model(ckpt_path, device, args.target)
        # 只评估用户指定且模型有的目标
        use_targets = [t for t in args.target if t in targets]
        by_k = collect_preds(model, loader, device, use_targets)
        mode_tables[mode] = rmse_table(by_k, use_targets)
        mode_meta[mode] = {
            "ckpt": ckpt_path,
            "epoch": ckpt.get("epoch"),
            "best_val_rmse_sum5": ckpt.get("best_val_rmse_sum5"),
            "gnn_type": ckpt.get("gnn_type"),
            "D": ckpt.get("D"),
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    target_list = list(args.target)
    kernel_names = sorted(k for k in next(iter(mode_tables.values())) if k != "__ALL__")

    # ---- CSV: kernel, target, cdfg_only, ast_only, LM, winner, margin ----
    csv_path = join(out_dir, "per_kernel_target_rmse.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["kernel", "target", "n_hint", *MODES, "winner", "best_rmse", "second_rmse", "gap"]
        )
        for k in kernel_names + ["__ALL__"]:
            for t in target_list:
                scores = {m: mode_tables[m].get(k, {}).get(t, float("nan")) for m in MODES}
                finite = [(m, s) for m, s in scores.items() if np.isfinite(s)]
                if not finite:
                    continue
                finite.sort(key=lambda x: x[1])
                winner, best = finite[0]
                second = finite[1][1] if len(finite) > 1 else float("nan")
                gap = float(second - best) if np.isfinite(second) else float("nan")
                w.writerow(
                    [
                        k,
                        t,
                        "",
                        *[f"{scores[m]:.6f}" if np.isfinite(scores[m]) else "" for m in MODES],
                        winner,
                        f"{best:.6f}",
                        f"{second:.6f}" if np.isfinite(second) else "",
                        f"{gap:.6f}" if np.isfinite(gap) else "",
                    ]
                )

    # ---- wins by target (count kernels where mode is best) ----
    win_by_target: Dict[str, Dict[str, int]] = {
        t: {m: 0 for m in MODES} for t in target_list
    }
    win_by_target_detail: Dict[str, Dict[str, List[str]]] = {
        t: {m: [] for m in MODES} for t in target_list
    }
    for t in target_list:
        for k in kernel_names:
            scores = [(m, mode_tables[m][k][t]) for m in MODES]
            scores = [(m, s) for m, s in scores if np.isfinite(s)]
            if not scores:
                continue
            scores.sort(key=lambda x: x[1])
            winner = scores[0][0]
            win_by_target[t][winner] += 1
            win_by_target_detail[t][winner].append(k)

    # ---- wins by kernel (sum5) ----
    win_by_kernel: Dict[str, dict] = {}
    win_count_sum5 = {m: 0 for m in MODES}
    for k in kernel_names:
        scores = {m: sum5(mode_tables[m][k], target_list) for m in MODES}
        finite = [(m, s) for m, s in scores.items() if np.isfinite(s)]
        finite.sort(key=lambda x: x[1])
        winner = finite[0][0] if finite else None
        if winner:
            win_count_sum5[winner] += 1
        win_by_kernel[k] = {
            "rmse": scores,
            "winner": winner,
            "sum5": {m: scores[m] for m in MODES},
        }

    # ---- overall ----
    overall = {
        m: {
            **mode_tables[m]["__ALL__"],
            "rmse_sum5": sum5(mode_tables[m]["__ALL__"], target_list),
        }
        for m in MODES
    }

    # pairwise: fraction of (kernel,target) where A < B
    pairwise = {}
    for a in MODES:
        for b in MODES:
            if a >= b:
                continue
            better_a = better_b = ties = 0
            for k in kernel_names:
                for t in target_list:
                    sa = mode_tables[a][k][t]
                    sb = mode_tables[b][k][t]
                    if not (np.isfinite(sa) and np.isfinite(sb)):
                        continue
                    if sa < sb - 1e-12:
                        better_a += 1
                    elif sb < sa - 1e-12:
                        better_b += 1
                    else:
                        ties += 1
            pairwise[f"{a}_vs_{b}"] = {
                f"{a}_better": better_a,
                f"{b}_better": better_b,
                "ties": ties,
                "n": better_a + better_b + ties,
            }

    summary = {
        "dataset_mode": ds,
        "ablation_root": ablation_root,
        "split": split_name,
        "seed": seed,
        "n_kernels": len(kernel_names),
        "n_samples": len(data_list),
        "targets": target_list,
        "modes": list(MODES),
        "mode_meta": mode_meta,
        "overall_rmse": overall,
        "win_count_by_target": win_by_target,
        "win_count_by_kernel_sum5": win_count_sum5,
        "pairwise_kernel_target": pairwise,
        "note": "winner = 更低 RMSE；证明三专家在不同指标/不同核上各有优势",
    }

    with open(join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(join(out_dir, "win_by_target.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"counts": win_by_target, "kernels": win_by_target_detail},
            f,
            indent=2,
            ensure_ascii=False,
        )
    with open(join(out_dir, "win_by_kernel.json"), "w", encoding="utf-8") as f:
        json.dump(win_by_kernel, f, indent=2, ensure_ascii=False)

    # ---- human report ----
    lines = []
    lines.append("=" * 72)
    lines.append(f"Expert specialization  split={split_name}  n_kernels={len(kernel_names)}")
    lines.append("=" * 72)
    lines.append("\n[1] Overall RMSE (all samples in split)")
    header = f"{'mode':12s}" + "".join(f"{t:>14s}" for t in target_list) + f"{'sum5':>14s}"
    lines.append(header)
    for m in MODES:
        row = f"{m:12s}"
        for t in target_list:
            v = overall[m].get(t, float("nan"))
            row += f"{v:14.4f}"
        row += f"{overall[m]['rmse_sum5']:14.4f}"
        lines.append(row)

    lines.append("\n[2] Win count by target (#kernels where mode has lowest RMSE)")
    lines.append(f"{'target':16s}" + "".join(f"{m:>14s}" for m in MODES))
    for t in target_list:
        lines.append(
            f"{t:16s}" + "".join(f"{win_by_target[t][m]:14d}" for m in MODES)
        )

    lines.append("\n[3] Win count by kernel (lowest sum5 across targets)")
    lines.append("  " + "  ".join(f"{m}={win_count_sum5[m]}" for m in MODES))

    lines.append("\n[4] Pairwise (kernel×target cells)")
    for key, d in pairwise.items():
        lines.append(f"  {key}: {d}")

    lines.append("\n[5] Per-kernel winner (sum5)")
    for k in kernel_names:
        info = win_by_kernel[k]
        s = info["sum5"]
        lines.append(
            f"  {k:20s} winner={info['winner']:10s} "
            + " ".join(f"{m}={s[m]:.4f}" for m in MODES)
        )

    lines.append("\n[6] 结论提示")
    # 若每个指标的赢家不全是同一模态 → 存在指标级专长
    winners_per_target = []
    for t in target_list:
        c = win_by_target[t]
        winners_per_target.append(max(c, key=c.get))
    if len(set(winners_per_target)) >= 2:
        lines.append(
            "  ✓ 不同指标上主导专家不同: "
            + ", ".join(f"{t}->{w}" for t, w in zip(target_list, winners_per_target))
        )
    else:
        lines.append(
            f"  · 各指标 kernel 胜场多由同一专家主导 ({winners_per_target[0]})；"
            "仍可看 pairwise / 单核表是否有交叉优势"
        )
    if len(set(win_by_kernel[k]["winner"] for k in kernel_names)) >= 2:
        lines.append("  ✓ 不同 kernel 上 sum5 最优专家不同 → 存在核级专长（适合 MoE）")
    else:
        lines.append("  · 多数核由同一专家在 sum5 上领先")

    report = "\n".join(lines) + "\n"
    report_path = join(out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report, flush=True)
    print(f"[compare] wrote:\n  {csv_path}\n  {out_dir}/summary.json\n  {report_path}", flush=True)


if __name__ == "__main__":
    main()
