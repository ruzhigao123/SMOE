import os
import random
import re
from typing import Dict, List, Tuple

import torch


# ===== 路径配置（按你的项目结构）=====
from os.path import abspath, dirname, join
_CODE_DIR = dirname(abspath(__file__))
AST_ROOT = join(_CODE_DIR, 'ast_final_dataset_1')
GRAPH_ROOT = join(dirname(_CODE_DIR), 'graph')

# 抽样数量
SAMPLE_SIZE = 20

# 需要比对的标签字段
TARGET_FIELDS = [
    "perf",
    "actual_perf",
    "quality",
    "util_BRAM",
    "util_DSP",
    "util_LUT",
    "util_FF",
    "total_BRAM",
    "total_DSP",
    "total_LUT",
    "total_FF",
]


def _safe_float(x):
    if torch.is_tensor(x):
        if x.numel() == 1:
            return float(x.item())
        return float(x.view(-1)[0].item())
    return float(x)


def _extract_idx_from_ast_name(ast_file_name: str) -> int:
    """
    兼容两种命名：
    - 15.ast.pt
    - data_15.ast.pt
    """
    stem = ast_file_name.replace(".ast.pt", "")
    m = re.search(r"(\d+)$", stem)
    if not m:
        raise ValueError(f"无法从文件名提取配置编号: {ast_file_name}")
    return int(m.group(1))


def _collect_ast_files(ast_root: str) -> List[str]:
    ast_files = []
    for dirpath, _, filenames in os.walk(ast_root):
        for f in filenames:
            if f.endswith(".ast.pt"):
                ast_files.append(os.path.join(dirpath, f))
    return ast_files


def _sample_files(ast_files: List[str], sample_size: int) -> List[str]:
    if len(ast_files) <= sample_size:
        return ast_files
    return random.sample(ast_files, sample_size)


def _check_one(ast_path: str) -> Dict:
    result = {
        "ast_path": ast_path,
        "status": "PASS",
        "reasons": [],
        "graph_path": "",
    }

    rel_path = os.path.relpath(ast_path, AST_ROOT)
    parts = rel_path.split(os.sep)
    if len(parts) < 3:
        result["status"] = "FAIL"
        result["reasons"].append(f"AST 相对路径层级异常: {rel_path}")
        return result

    benchmark = parts[0]
    kernel = parts[1]
    ast_file_name = parts[2]

    try:
        idx = _extract_idx_from_ast_name(ast_file_name)
    except Exception as e:
        result["status"] = "FAIL"
        result["reasons"].append(str(e))
        return result

    graph_path = os.path.join(GRAPH_ROOT, benchmark, kernel, f"data_{idx}.pt")
    result["graph_path"] = graph_path

    if not os.path.exists(graph_path):
        result["status"] = "FAIL"
        result["reasons"].append(f"找不到对应 graph 文件: {graph_path}")
        return result

    try:
        ast_data = torch.load(ast_path, map_location="cpu")
        graph_data = torch.load(graph_path, map_location="cpu")
    except Exception as e:
        result["status"] = "FAIL"
        result["reasons"].append(f"加载 pt 失败: {e}")
        return result

    # 1) 元信息检查（merge_ast_dataset.py 注入）
    if hasattr(ast_data, "kernel"):
        ast_kernel = str(ast_data.kernel)
        if ast_kernel != kernel:
            result["status"] = "FAIL"
            result["reasons"].append(f"kernel 不一致: ast={ast_kernel}, path={kernel}")
    else:
        result["reasons"].append("AST 中缺少 kernel 字段（不一定致命）")

    if hasattr(ast_data, "benchmark"):
        ast_bench = str(ast_data.benchmark)
        if ast_bench != benchmark:
            result["status"] = "FAIL"
            result["reasons"].append(f"benchmark 不一致: ast={ast_bench}, path={benchmark}")
    else:
        result["reasons"].append("AST 中缺少 benchmark 字段（不一定致命）")

    # 2) 标签值逐项比对
    for field in TARGET_FIELDS:
        if not hasattr(graph_data, field):
            result["status"] = "FAIL"
            result["reasons"].append(f"graph 缺少字段: {field}")
            continue
        if not hasattr(ast_data, field):
            result["status"] = "FAIL"
            result["reasons"].append(f"ast 缺少字段: {field}")
            continue

        g_val = _safe_float(getattr(graph_data, field))
        a_val = _safe_float(getattr(ast_data, field))
        if abs(g_val - a_val) > 1e-8:
            result["status"] = "FAIL"
            result["reasons"].append(f"{field} 值不一致: ast={a_val}, graph={g_val}")

    return result


def main():
    random.seed(42)
    ast_files = _collect_ast_files(AST_ROOT)
    if not ast_files:
        print(f"❌ 在 {AST_ROOT} 下没有找到 .ast.pt 文件")
        return

    chosen = _sample_files(ast_files, SAMPLE_SIZE)
    print(f"🔎 共发现 AST 文件: {len(ast_files)}，本次抽样检查: {len(chosen)}")

    pass_cnt = 0
    fail_cnt = 0
    detail_rows: List[Tuple[int, Dict]] = []

    for i, ast_path in enumerate(chosen, 1):
        ret = _check_one(ast_path)
        detail_rows.append((i, ret))
        if ret["status"] == "PASS":
            pass_cnt += 1
        else:
            fail_cnt += 1

    print("\n================= 抽样检查结果 =================")
    print(f"PASS: {pass_cnt} | FAIL: {fail_cnt} | TOTAL: {len(chosen)}")
    print("===============================================")

    for i, row in detail_rows:
        print(f"\n[{i:02d}] {row['status']}")
        print(f"AST   : {row['ast_path']}")
        print(f"GRAPH : {row['graph_path']}")
        if row["reasons"]:
            for r in row["reasons"]:
                print(f"  - {r}")
        else:
            print("  - 映射与标签检查全部通过")

    print("\n判定建议：")
    if fail_cnt == 0:
        print("✅ 这批抽样映射正确，可认为 AST 与 graph 配对逻辑基本可靠。")
    else:
        print("⚠️ 存在不匹配样本，请优先检查 merge_ast_dataset.py 的 idx 解析和路径拼接。")


if __name__ == "__main__":
    main()
