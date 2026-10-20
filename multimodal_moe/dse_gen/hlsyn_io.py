"""
new_kernel/dse_database 数据源适配。

布局（相对 HierarchicalMoE-master）:
  new_kernel/dse_database/
    {machsuite|poly}/
      config/{kernel}_ds_config.json     # 设计空间定义（非探索结果）
      databases/**/{kernel}_result_*.db  # 探索得到的 design points
      sources/**/{kernel}_kernel.c
    programl/{machsuite|poly}/
      processed/{kernel}_processed_result.gexf
      {kernel}/{kernel}.c                # 带 auto{PRAGMA} 的模板（优先）

对外 API 与原先 HLSyn 适配层兼容，便于 gen_dataset*.py 切换。
"""
from __future__ import annotations

import pickle
import sys
from glob import iglob
from os.path import abspath, basename, dirname, exists, join
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

_DSE_GEN_DIR = dirname(abspath(__file__))
_MM_ROOT = dirname(_DSE_GEN_DIR)
_HMOE_ROOT = dirname(_MM_ROOT)

DSE_DB_ROOT = join(_HMOE_ROOT, "new_kernel", "dse_database")
# 兼容旧 import 名
HLSYN_DATA_ROOT = DSE_DB_ROOT
GRAPHS_DIR = join(DSE_DB_ROOT, "programl")

MACHSUITE_KERNELS = {
    "aes",
    "gemm-blocked",
    "gemm-ncubed",
    "spmv-crs",
    "spmv-ellpack",
    "stencil",
    "nw",
    "md",
    "stencil-3d",
}

# sources 文件名别名：kernel 名 -> 实际 source stem（不含 _kernel.c）
SOURCE_FILE_ALIAS = {
    "spmv-crs": "spmv-src",  # machsuite/sources/original-size/spmv-src_kernel.c
}


def suite_of(kernel: str, machsuite: Optional[Iterable[str]] = None) -> str:
    ms = set(machsuite) if machsuite is not None else MACHSUITE_KERNELS
    return "machsuite" if kernel in ms else "poly"


def _ensure_autodse_on_path() -> None:
    """MerlinResult pickle 需要 import autodse.result。"""
    if DSE_DB_ROOT not in sys.path:
        sys.path.insert(0, DSE_DB_ROOT)


def resolve_gexf(kernel: str) -> Optional[str]:
    suite = suite_of(kernel)
    p = join(DSE_DB_ROOT, "programl", suite, "processed", f"{kernel}_processed_result.gexf")
    return p if exists(p) else None


def resolve_config(kernel: str) -> Optional[str]:
    suite = suite_of(kernel)
    p = join(DSE_DB_ROOT, suite, "config", f"{kernel}_ds_config.json")
    return p if exists(p) else None


def resolve_db_paths(kernel: str) -> List[str]:
    """精确匹配 {kernel}_result*.db，避免 gemm 误匹配 gemm-p。"""
    suite = suite_of(kernel)
    pattern = join(DSE_DB_ROOT, suite, "databases", "**", "*")
    out = []
    for f in iglob(pattern, recursive=True):
        if not f.endswith(".db"):
            continue
        name = basename(f)
        if name.startswith(f"{kernel}_result") and name.endswith(".db"):
            out.append(f)
    return sorted(out)


def resolve_source(kernel: str) -> Optional[str]:
    suite = suite_of(kernel)
    stem = SOURCE_FILE_ALIAS.get(kernel, kernel)
    candidates = [
        join(DSE_DB_ROOT, "programl", suite, kernel, f"{kernel}.c"),
        join(DSE_DB_ROOT, "programl", suite, kernel, f"{kernel}.cpp"),
        join(DSE_DB_ROOT, suite, "sources", f"{stem}_kernel.c"),
        join(DSE_DB_ROOT, suite, "sources", f"{stem}_kernel.cpp"),
        join(DSE_DB_ROOT, suite, "sources", "original-size", f"{stem}_kernel.c"),
        join(DSE_DB_ROOT, suite, "sources", "original-size", f"{stem}_kernel.cpp"),
        join(DSE_DB_ROOT, suite, "sources", f"{kernel}_kernel.c"),
        join(DSE_DB_ROOT, suite, "sources", "original-size", f"{kernel}_kernel.c"),
    ]
    for p in candidates:
        if exists(p):
            return p
    # 目录扫描兜底
    src_root = join(DSE_DB_ROOT, suite, "sources")
    if exists(src_root):
        hits = [
            f
            for f in iglob(join(src_root, "**", "*"), recursive=True)
            if f.endswith((".c", ".cpp")) and (kernel in basename(f) or stem in basename(f))
        ]
        if hits:
            return sorted(hits)[0]
    return None


def _unpickle_result(raw):
    if isinstance(raw, (int, float)):
        return None
    if isinstance(raw, dict) and "perf" in raw:
        return SimpleNamespace(
            perf=float(raw.get("perf") or 0.0),
            point=dict(raw.get("point") or {}),
            res_util=dict(raw.get("res_util") or {}),
            valid=bool(raw.get("valid", True)),
        )
    if isinstance(raw, bytes):
        _ensure_autodse_on_path()
        blob = raw.replace(b"localdse", b"autodse")
        try:
            return pickle.loads(blob)
        except Exception:
            try:
                return pickle.loads(
                    blob.replace(b"autodse", b"dse_database.autodse")
                )
            except Exception:
                return None
    if hasattr(raw, "perf"):
        return raw
    return None


def load_kernel_database(kernel: str) -> Tuple[Dict[str, object], str, str]:
    """
    加载某 kernel 的全部 design points（来自 .db）。
    Returns: (database, primary_db_path, tag)
    """
    db_paths = resolve_db_paths(kernel)
    if not db_paths:
        raise FileNotFoundError(f"No .db for kernel={kernel!r} under {DSE_DB_ROOT}")

    _ensure_autodse_on_path()
    database: Dict[str, object] = {}
    for path in db_paths:
        with open(path, "rb") as f:
            raw = pickle.load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Unexpected .db root type in {path}: {type(raw)}")
        for key, val in raw.items():
            k_str = key.decode("utf-8") if isinstance(key, bytes) else str(key)
            obj = _unpickle_result(val)
            if obj is None:
                continue
            database[k_str] = obj
    return database, db_paths[0], "db"


def apply_point_to_source(source_text: str, point: Optional[dict], key: str = "") -> str:
    """用 design point 替换 auto{PRAGMA}；无 point 时回退解析 key（lv1:/lv2: 前缀）。"""
    fcc = source_text
    if isinstance(point, dict) and point:
        for k, v in point.items():
            if v is None or v == "NA":
                val = ""
            else:
                val = str(v)
            fcc = fcc.replace("auto{" + str(k) + "}", val)
        return fcc

    body = key
    if body.startswith("lv1") or body.startswith("lv2"):
        # lv1:xxx 或 lv2-xxx / lv2:xxx
        if len(body) > 4 and body[3] in ":-_":
            body = body[4:]
        else:
            body = body[3:]
    for part in body.split("."):
        if "-" not in part:
            continue
        name, _, rest = part.partition("-")
        val = "" if rest == "NA" else rest
        fcc = fcc.replace("auto{" + name + "}", val)
    return fcc


def list_kernels_with_assets(kernels: Iterable[str]) -> List[str]:
    out = []
    for k in kernels:
        if resolve_gexf(k) and resolve_db_paths(k) and resolve_source(k):
            out.append(k)
    return out
