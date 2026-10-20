#!/usr/bin/env python3
"""
统计 new_kernel/dse_database 下 42 kernel 的 .db 配置数量。

用法:
  python count_hlsyn_configs.py
  python count_hlsyn_configs.py --check-generated
  python count_hlsyn_configs.py --csv /tmp/new_kernel_counts.csv
"""
from __future__ import annotations

import argparse
from os.path import abspath, dirname, join
from pathlib import Path
import sys

_DSE_GEN = dirname(abspath(__file__))
if _DSE_GEN not in sys.path:
    sys.path.insert(0, _DSE_GEN)

from hlsyn_io import (  # noqa: E402
    DSE_DB_ROOT,
    load_kernel_database,
    resolve_config,
    resolve_db_paths,
    resolve_gexf,
    resolve_source,
    suite_of,
)

MACHSUITE_KERNEL = [
    "aes",
    "gemm-blocked",
    "gemm-ncubed",
    "spmv-crs",
    "spmv-ellpack",
    "stencil",
    "nw",
    "md",
    "stencil-3d",
]
POLY_KERNEL = [
    "2mm",
    "3mm",
    "adi",
    "atax",
    "bicg",
    "bicg-large",
    "covariance",
    "doitgen",
    "doitgen-red",
    "fdtd-2d",
    "fdtd-2d-large",
    "gemm-p",
    "gemm-p-large",
    "gemver",
    "gesummv",
    "heat-3d",
    "jacobi-1d",
    "jacobi-2d",
    "mvt",
    "seidel-2d",
    "symm",
    "symm-opt",
    "syrk",
    "syr2k",
    "trmm",
    "trmm-opt",
    "mvt-medium",
    "correlation",
    "atax-medium",
    "bicg-medium",
    "gesummv-medium",
    "symm-opt-medium",
    "gemver-medium",
]
ALL_KERNEL = MACHSUITE_KERNEL + POLY_KERNEL
_MM_ROOT = dirname(_DSE_GEN)
_GEN_C_ROOT = join(_MM_ROOT, "two_tower_datasets_c")


def count_db(kernel: str):
    try:
        db, path, tag = load_kernel_database(kernel)
    except Exception:
        return None
    n_all = n_pos = n_lv2 = 0
    for key, obj in db.items():
        perf = float(getattr(obj, "perf", 0) or 0)
        n_all += 1
        if perf > 0:
            n_pos += 1
        if str(key).startswith("lv2"):
            n_lv2 += 1
    return path, tag, n_all, n_pos, n_lv2


def count_generated(kernel: str):
    suite = suite_of(kernel, MACHSUITE_KERNEL)
    c_dir = Path(_GEN_C_ROOT) / "code" / suite / kernel
    g_dir = Path(_GEN_C_ROOT) / "graph" / suite / kernel
    n_c = len(list(c_dir.glob("*.c"))) if c_dir.is_dir() else 0
    n_g = len(list(g_dir.glob("data_*.pt"))) if g_dir.is_dir() else 0
    return n_c, n_g


def main():
    ap = argparse.ArgumentParser(description="Count new_kernel .db design configs")
    ap.add_argument("--check-generated", action="store_true")
    ap.add_argument("--csv", type=str, default="")
    args = ap.parse_args()

    print(f"new_kernel root: {DSE_DB_ROOT}")
    if args.check_generated:
        print(f"generated: {_GEN_C_ROOT}")
    print()

    hdr = f"{'kernel':28} {'all':>6} {'perf>0':>7} {'lv2':>6}"
    if args.check_generated:
        hdr += f" {'#.c':>6} {'#graph':>7}"
    hdr += "  assets"
    print(hdr)
    print("-" * len(hdr))

    tot_all = tot_pos = tot_lv2 = tot_c = tot_g = 0
    n_ok = 0
    csv_rows = []

    for k in ALL_KERNEL:
        assets = (
            f"cfg={'Y' if resolve_config(k) else 'N'} "
            f"db={'Y' if resolve_db_paths(k) else 'N'} "
            f"gexf={'Y' if resolve_gexf(k) else 'N'} "
            f"src={'Y' if resolve_source(k) else 'N'}"
        )
        info = count_db(k)
        nc = ng = 0
        if info is None:
            line = f"{k:28} {0:6d} {0:7d} {0:6d}"
            if args.check_generated:
                nc, ng = count_generated(k)
                line += f" {nc:6d} {ng:7d}"
                tot_c += nc
                tot_g += ng
            line += f"  MISSING {assets}"
            print(line)
            csv_rows.append((k, 0, 0, 0, nc, ng, "MISSING"))
            continue

        _path, _tag, n_all, n_pos, n_lv2 = info
        tot_all += n_all
        tot_pos += n_pos
        tot_lv2 += n_lv2
        n_ok += 1
        line = f"{k:28} {n_all:6d} {n_pos:7d} {n_lv2:6d}"
        if args.check_generated:
            nc, ng = count_generated(k)
            tot_c += nc
            tot_g += ng
            line += f" {nc:6d} {ng:7d}"
        line += f"  {assets}"
        print(line)
        csv_rows.append((k, n_all, n_pos, n_lv2, nc, ng, "ok"))

    print("-" * len(hdr))
    tot_line = f"{'TOTAL':28} {tot_all:6d} {tot_pos:7d} {tot_lv2:6d}"
    if args.check_generated:
        tot_line += f" {tot_c:6d} {tot_g:7d}"
    print(tot_line)
    print()
    print(f"kernels with .db: {n_ok} / {len(ALL_KERNEL)}")
    print("口径: all=可反序列化 Result; perf>0=有效 latency; lv2=key 以 lv2 开头")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("kernel,all,perf_gt0,lv2,n_c,n_graph,status\n")
            for row in csv_rows:
                f.write(",".join(map(str, row)) + "\n")
            f.write(f"TOTAL,{tot_all},{tot_pos},{tot_lv2},{tot_c},{tot_g},\n")
        print(f"CSV written: {out}")


if __name__ == "__main__":
    main()
