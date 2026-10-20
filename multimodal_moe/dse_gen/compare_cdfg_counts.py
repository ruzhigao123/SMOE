#!/usr/bin/env python3
"""
比较两套 CDFG（graph/data_*.pt）数量是否一致：

1) new_kernel/dse_database/gen_dataset.py
   → new_kernel/two_tower_dataset/graph/{machsuite|poly}/{kernel}/data_*.pt

2) multimodal_moe/dse_gen/gen_dataset.py
   → multimodal_moe/two_tower_datasets/graph/{machsuite|poly}/{kernel}/data_*.pt

用法:
  python compare_cdfg_counts.py
  python compare_cdfg_counts.py --csv /tmp/cdfg_compare.csv
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from os.path import abspath, dirname, join
from pathlib import Path


_HERE = dirname(abspath(__file__))
_MM_ROOT = dirname(_HERE)  # multimodal_moe
_HMOE_ROOT = dirname(_MM_ROOT)
_NEW_KERNEL_GRAPH = Path(_HMOE_ROOT) / "new_kernel" / "two_tower_dataset" / "graph"
_MM_GRAPH = Path(_MM_ROOT) / "two_tower_datasets" / "graph"

SUITES = ("machsuite", "poly")


def count_kernel_graphs(graph_root: Path) -> dict:
    """返回 {kernel: count}；同时记录 suite。"""
    counts = {}
    suite_of = {}
    if not graph_root.is_dir():
        return counts, suite_of
    for suite in SUITES:
        suite_dir = graph_root / suite
        if not suite_dir.is_dir():
            continue
        for kdir in sorted(suite_dir.iterdir()):
            if not kdir.is_dir():
                continue
            n = len(list(kdir.glob("data_*.pt")))
            counts[kdir.name] = n
            suite_of[kdir.name] = suite
    return counts, suite_of


def main():
    ap = argparse.ArgumentParser(description="Compare CDFG counts: new_kernel vs multimodal_moe")
    ap.add_argument("--new-kernel-graph", type=str, default=str(_NEW_KERNEL_GRAPH))
    ap.add_argument("--mm-graph", type=str, default=str(_MM_GRAPH))
    ap.add_argument("--csv", type=str, default="")
    args = ap.parse_args()

    nk_root = Path(args.new_kernel_graph)
    mm_root = Path(args.mm_graph)

    nk, nk_suite = count_kernel_graphs(nk_root)
    mm, mm_suite = count_kernel_graphs(mm_root)

    all_kernels = sorted(set(nk) | set(mm))

    print(f"new_kernel graph : {nk_root}  (exists={nk_root.is_dir()})")
    print(f"multimodal graph : {mm_root}  (exists={mm_root.is_dir()})")
    print()
    hdr = f"{'kernel':28} {'suite':10} {'new_kernel':>10} {'multimodal':>10} {'diff':>8}  status"
    print(hdr)
    print("-" * len(hdr))

    tot_nk = tot_mm = 0
    n_match = n_mismatch = n_only_nk = n_only_mm = 0
    rows = []

    for k in all_kernels:
        a = nk.get(k)
        b = mm.get(k)
        suite = nk_suite.get(k) or mm_suite.get(k) or "?"
        if a is None:
            status = "ONLY_multimodal"
            n_only_mm += 1
            diff = "NA"
            a_s, b_s = "-", str(b)
            tot_mm += b
        elif b is None:
            status = "ONLY_new_kernel"
            n_only_nk += 1
            diff = "NA"
            a_s, b_s = str(a), "-"
            tot_nk += a
        else:
            d = a - b
            tot_nk += a
            tot_mm += b
            if d == 0:
                status = "MATCH"
                n_match += 1
            else:
                status = "MISMATCH"
                n_mismatch += 1
            diff = str(d)
            a_s, b_s = str(a), str(b)
        print(f"{k:28} {suite:10} {a_s:>10} {b_s:>10} {diff:>8}  {status}")
        rows.append((k, suite, a if a is not None else "", b if b is not None else "", diff, status))

    print("-" * len(hdr))
    print(f"{'TOTAL':28} {'':10} {tot_nk:>10} {tot_mm:>10} {tot_nk - tot_mm:>8}")
    print()
    print(f"kernels: new_kernel={len(nk)}  multimodal={len(mm)}  union={len(all_kernels)}")
    print(f"MATCH={n_match}  MISMATCH={n_mismatch}  ONLY_new_kernel={n_only_nk}  ONLY_multimodal={n_only_mm}")
    consistent = (
        n_mismatch == 0
        and n_only_nk == 0
        and n_only_mm == 0
        and tot_nk == tot_mm
        and len(nk) > 0
        and len(mm) > 0
    )
    print()
    if not nk_root.is_dir() or not mm_root.is_dir():
        print("RESULT: INCOMPLETE (至少一侧 graph 目录不存在，请先跑完对应 gen_dataset.py)")
    elif consistent:
        print("RESULT: CONSISTENT — 两侧 CDFG 按 kernel 数量完全一致")
    else:
        print("RESULT: INCONSISTENT — 两侧 CDFG 数量不一致（见上表）")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("kernel,suite,new_kernel,multimodal,diff,status\n")
            for row in rows:
                f.write(",".join(map(str, row)) + "\n")
            f.write(f"TOTAL,,{tot_nk},{tot_mm},{tot_nk - tot_mm},\n")
        print(f"CSV written: {out}")


if __name__ == "__main__":
    main()
