import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm


from os.path import abspath, dirname, join
_CODE_DIR = dirname(abspath(__file__))
_MM_ROOT = dirname(dirname(_CODE_DIR))
BASE_DIR = _MM_ROOT
AST_FINAL_ROOT = join(_CODE_DIR, "ast_final_dataset_1")
AST_JSON_ROOT = join(_CODE_DIR, "ast_output")
REPORT_PATH = join(_CODE_DIR, "ast_feature_effectiveness_report.json")

# 开关：是否额外做 9 维（algorithm_utils）分析
INCLUDE_BASE9 = True

# 开关：是否额外做 25 维 meta（9+16）分析
# 注意：ast_final_dataset_1 本体只有 16 维硬件特征（附在 CodeBERT 后）。
# 25 维中的前 9 维来自 algorithm_utils.py，需要额外从 ast_output/*.ast.json 计算。
# 因此默认关闭，避免与你当前“只看16维”需求冲突。
INCLUDE_META25 = False

sys.path.append(BASE_DIR)
try:
    from algorithm_select.algorithm_utils import extract_ast_meta_features  # noqa: E402
except ImportError:
    extract_ast_meta_features = None
    INCLUDE_BASE9 = False
    INCLUDE_META25 = False
    print("[warn] algorithm_select 不在本仓库，已关闭 BASE9/META25 分析")


TARGET_FIELDS = {
    "perf": "perf",
    "lut": "util_LUT",
    "ff": "util_FF",
    "dsp": "util_DSP",
    "bram": "util_BRAM",
}

HW16_NAMES = [
    "is_pragma_or_accel",
    "pipeline_ii",
    "parallel_or_unroll_factor",
    "tile_factor",
    "pipeline_off_penalty",
    "has_pipeline",
    "has_parallel_or_unroll",
    "has_reduction",
    "has_flatten",
    "has_scop",
    "has_partition",
    "has_interface_or_bus",
    "is_loop_node",
    "is_branch_node",
    "norm_depth",
    "norm_text_len",
]

BASE9_NAMES = [
    "for_loops",
    "if_branches",
    "array_decls",
    "time_pragmas",
    "resource_pragmas",
    "max_loop_depth",
    "op_complexity",
    "bit_width_score",
    "total_nodes",
]

META25_NAMES = [
    "meta_for_loops",
    "meta_if_branches",
    "meta_array_decls",
    "meta_time_pragmas",
    "meta_resource_pragmas",
    "meta_max_loop_depth",
    "meta_op_complexity",
    "meta_bit_width_score",
    "meta_total_nodes",
    "meta_adv_is_pragma",
    "meta_adv_pipeline_ii",
    "meta_adv_parallel_factor",
    "meta_adv_tile_factor",
    "meta_adv_pipeline_off_penalty",
    "meta_adv_has_pipeline",
    "meta_adv_has_parallel_or_unroll",
    "meta_adv_has_reduction",
    "meta_adv_has_flatten",
    "meta_adv_has_scop",
    "meta_adv_has_partition",
    "meta_adv_has_interface_or_bus",
    "meta_adv_has_false_dependence",
    "meta_adv_is_loop_node",
    "meta_adv_is_branch_node",
    "meta_adv_norm_depth",
    "meta_adv_norm_text_len",
]


def rankdata_average(x: np.ndarray) -> np.ndarray:
    """
    简易平均秩实现（处理并列值）。
    """
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]

    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    xr = rankdata_average(x)
    yr = rankdata_average(y)
    x_std = np.std(xr)
    y_std = np.std(yr)
    if x_std < 1e-12 or y_std < 1e-12:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def collect_ast_files(root: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".ast.pt"):
                files.append(os.path.join(dirpath, f))
    return files


def infer_ast_json_path(ast_pt_path: str) -> str:
    rel = os.path.relpath(ast_pt_path, AST_FINAL_ROOT)
    # .../machsuite/nw/15.ast.pt -> .../machsuite/nw/15.ast.json
    return os.path.join(AST_JSON_ROOT, rel.replace(".ast.pt", ".ast.json"))


def extract_meta25_from_paths(ast_pt_path: str, hw16_mean: np.ndarray) -> np.ndarray:
    base9 = np.zeros(9, dtype=np.float32)
    ast_json_path = infer_ast_json_path(ast_pt_path)
    if os.path.exists(ast_json_path):
        try:
            with open(ast_json_path, "r", encoding="utf-8") as f:
                ast_json = json.load(f)
            base9_t = extract_ast_meta_features(ast_json)
            base9 = torch.as_tensor(base9_t).flatten().cpu().numpy().astype(np.float32)
            if base9.shape[0] >= 9:
                base9 = base9[:9]
            else:
                base9 = np.pad(base9, (0, 9 - base9.shape[0]), mode="constant")
        except Exception:
            pass
    return np.concatenate([base9, hw16_mean.astype(np.float32)], axis=0)


def stats_and_corr(
    feature_mat: np.ndarray,
    feature_names: List[str],
    targets: Dict[str, np.ndarray],
) -> Tuple[List[Dict], List[str]]:
    rows: List[Dict] = []
    warnings: List[str] = []

    for i, name in enumerate(feature_names):
        col = feature_mat[:, i]
        non_zero_rate = float(np.mean(np.abs(col) > 1e-12))
        mean = float(np.mean(col))
        std = float(np.std(col))
        var = float(np.var(col))
        min_v = float(np.min(col))
        max_v = float(np.max(col))

        corr = {f"spearman_{k}": spearman_corr(col, v) for k, v in targets.items()}
        row = {
            "feature": name,
            "non_zero_rate": non_zero_rate,
            "mean": mean,
            "std": std,
            "var": var,
            "min": min_v,
            "max": max_v,
            **corr,
        }
        rows.append(row)

        if non_zero_rate < 0.01:
            warnings.append(f"{name}: 非零率过低 ({non_zero_rate:.2%})")
        if var < 1e-10:
            warnings.append(f"{name}: 方差过低 ({var:.3e})")

    return rows, warnings


def print_table(title: str, rows: List[Dict], top_n: int = 30):
    print(f"\n===== {title} =====")
    print(
        f"{'feature':<34} {'nz_rate':>8} {'var':>12} {'sp_perf':>9} {'sp_lut':>9} "
        f"{'sp_ff':>9} {'sp_dsp':>9} {'sp_bram':>9}"
    )
    for r in rows[:top_n]:
        print(
            f"{r['feature']:<34} {r['non_zero_rate']:>8.2%} {r['var']:>12.3e} "
            f"{r['spearman_perf']:>9.3f} {r['spearman_lut']:>9.3f} "
            f"{r['spearman_ff']:>9.3f} {r['spearman_dsp']:>9.3f} {r['spearman_bram']:>9.3f}"
        )


def main():
    ast_files = collect_ast_files(AST_FINAL_ROOT)
    if not ast_files:
        print(f"❌ 未在 {AST_FINAL_ROOT} 下找到 .ast.pt 文件")
        return

    hw16_list = []
    base9_list = []
    meta25_list = []
    target_dict = {k: [] for k in TARGET_FIELDS.keys()}

    skipped = 0
    for p in tqdm(ast_files, desc="Scanning AST PT"):
        try:
            data = torch.load(p, map_location="cpu")
            if not hasattr(data, "x"):
                skipped += 1
                continue
            x = data.x
            if x.dim() != 2 or x.shape[1] < 16:
                skipped += 1
                continue
            hw16 = x[:, -16:].mean(dim=0).cpu().numpy().astype(np.float64)

            # 目标字段必须存在
            missing_target = False
            curr_targets = {}
            for short_k, field in TARGET_FIELDS.items():
                if not hasattr(data, field):
                    missing_target = True
                    break
                val = getattr(data, field)
                curr_targets[short_k] = float(val.item() if torch.is_tensor(val) and val.numel() == 1 else torch.as_tensor(val).view(-1)[0].item())
            if missing_target:
                skipped += 1
                continue

            hw16_list.append(hw16)
            for k, v in curr_targets.items():
                target_dict[k].append(v)

            if INCLUDE_BASE9:
                ast_json_path = infer_ast_json_path(p)
                if os.path.exists(ast_json_path):
                    try:
                        with open(ast_json_path, "r", encoding="utf-8") as f:
                            ast_json = json.load(f)
                        base9 = torch.as_tensor(extract_ast_meta_features(ast_json)).flatten().cpu().numpy().astype(np.float64)
                        if base9.shape[0] >= 9:
                            base9 = base9[:9]
                        else:
                            base9 = np.pad(base9, (0, 9 - base9.shape[0]), mode="constant")
                        base9_list.append(base9)
                    except Exception:
                        base9_list.append(np.zeros(9, dtype=np.float64))
                else:
                    base9_list.append(np.zeros(9, dtype=np.float64))

            if INCLUDE_META25:
                meta25 = extract_meta25_from_paths(p, hw16.astype(np.float32))
                if meta25.shape[0] != 25:
                    meta25 = np.pad(meta25[:25], (0, max(0, 25 - len(meta25[:25]))), mode="constant")
                meta25_list.append(meta25.astype(np.float64))
        except Exception:
            skipped += 1
            continue

    if not hw16_list:
        print("❌ 没有可用样本用于分析。")
        return

    hw16_mat = np.stack(hw16_list, axis=0)
    targets_np = {k: np.array(v, dtype=np.float64) for k, v in target_dict.items()}

    hw_rows, hw_warn = stats_and_corr(hw16_mat, HW16_NAMES, targets_np)
    print(f"\n✅ 可用样本: {hw16_mat.shape[0]} | 跳过样本: {skipped}")
    print_table("HW16 Feature Health", hw_rows, top_n=16)

    base9_rows, base9_warn = [], []
    if INCLUDE_BASE9 and base9_list:
        base9_mat = np.stack(base9_list, axis=0)
        base9_rows, base9_warn = stats_and_corr(base9_mat, BASE9_NAMES, targets_np)
        print_table("BASE9 Feature Health", base9_rows, top_n=9)

    meta_rows, meta_warn = [], []
    if INCLUDE_META25 and meta25_list:
        meta25_mat = np.stack(meta25_list, axis=0)
        meta_rows, meta_warn = stats_and_corr(meta25_mat, META25_NAMES, targets_np)
        print_table("META25 Feature Health", meta_rows, top_n=25)

    all_warn = hw_warn + base9_warn + meta_warn
    print("\n===== Warnings =====")
    if not all_warn:
        print("无明显异常（低非零率/低方差）")
    else:
        for w in all_warn:
            print(f"- {w}")

    report = {
        "num_total_ast_files": len(ast_files),
        "num_used_samples": int(hw16_mat.shape[0]),
        "num_skipped_samples": int(skipped),
        "hw16_stats": hw_rows,
        "base9_stats": base9_rows,
        "meta25_stats": meta_rows,
        "warnings": all_warn,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n📄 报告已保存: {REPORT_PATH}")


if __name__ == "__main__":
    main()
