"""
从 points_list.pkl 或 .c 源文件解析 DesignPoint（pragma 配置）。

不同 kernel 的 pragma key 数目不同（约 3–21），编码侧用 PragmaFeatureEncoder
做 union-key 固定维向量，缺失 key 填 0/UNK。
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from os.path import abspath, dirname, exists, join
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from .pragma_features import DesignPoint, PragmaFeatureEncoder

_MOE = dirname(abspath(__file__))
_MM = dirname(_MOE)
_MPM = join(dirname(dirname(_MM)), "MPM-Innovation-master", "MPM-Innovation-master")


def default_points_roots(dataset_mode: Optional[str] = None) -> List[str]:
    try:
        from multimodal_moe.moe_system.utils_new import default_data_paths
    except Exception:
        from utils_new import default_data_paths  # type: ignore
    paths = default_data_paths(dataset_mode)
    roots = [paths["points_root"], paths.get("c_points_root")]
    # MPM 回退
    roots.append(join(_MPM, "two_tower_datasets", "points"))
    if paths["dataset_mode"] == "not_0":
        roots.insert(0, join(_MM, "two_tower_datasets_not_0", "points"))
        roots.insert(1, join(_MM, "two_tower_datasets_c_not_0", "points"))
    out: List[str] = []
    for r in roots:
        if r and r not in out:
            out.append(r)
    return out


def default_c_src_roots(dataset_mode: Optional[str] = None) -> List[str]:
    try:
        from multimodal_moe.moe_system.utils_new import default_data_paths
    except Exception:
        from utils_new import default_data_paths  # type: ignore
    paths = default_data_paths(dataset_mode)
    roots = [paths["c_src_root"], join(_MPM, "two_tower_datasets_c", "code")]
    if paths["dataset_mode"] == "not_0":
        roots.insert(0, join(_MM, "two_tower_datasets_c_not_0", "code"))
    out: List[str] = []
    for r in roots:
        if r and r not in out:
            out.append(r)
    return out


# ---- minimal .c pragma parser (ACCEL style) ----

PRAGMA_RE = re.compile(
    r"^\s*#\s*pragma\s+ACCEL\s+(?P<body>.+?)\s*$",
    re.IGNORECASE,
)
LOOP_LABEL_SAME_LINE_RE = re.compile(
    r"^\s*(?P<label>[A-Za-z_][\w]*)\s*:\s*(?:for|while)\b"
)
LOOP_LABEL_ONLY_RE = re.compile(r"^\s*(?P<label>[A-Za-z_][\w]*)\s*:\s*$")
FOR_WHILE_RE = re.compile(r"^\s*(?:for|while)\b")
FACTOR_RE = re.compile(r"FACTOR\s*=\s*(?P<v>-?\d+)", re.IGNORECASE)
REDUCTION_RE = re.compile(r"reduction\s*=\s*(?P<v>\w+)", re.IGNORECASE)


@dataclass
class ParsedPragma:
    kind: str
    value: Union[str, int, None]
    reduction: Optional[str]
    loop_label: Optional[str]
    loop_index: int
    raw: str


def _classify_body(body: str) -> Optional[ParsedPragma]:
    body = body.strip()
    if not body or body.lower() == "kernel":
        return None
    upper = body.upper()
    reduction = None
    m_red = REDUCTION_RE.search(body)
    if m_red:
        reduction = m_red.group("v")
    factor = None
    m_fac = FACTOR_RE.search(body)
    if m_fac:
        factor = int(m_fac.group("v"))

    if upper.startswith("PIPELINE"):
        rest = body[len("PIPELINE") :].strip()
        if rest.lower().startswith("factor"):
            value: Union[str, int, None] = "on"
        elif rest:
            value = rest.split()[0].lower()
        else:
            value = "on"
        return ParsedPragma("PIPELINE", value, reduction, None, -1, body)
    if upper.startswith("TILE") or upper.startswith("TILING"):
        return ParsedPragma(
            "TILE", factor if factor is not None else 1, reduction, None, -1, body
        )
    if upper.startswith("PARALLEL") or upper.startswith("UNROLL"):
        kind = "PARALLEL" if upper.startswith("PARALLEL") else "UNROLL"
        return ParsedPragma(
            kind, factor if factor is not None else 1, reduction, None, -1, body
        )
    tok = body.split()[0].upper()
    return ParsedPragma(tok, factor if factor is not None else body, reduction, None, -1, body)


def parse_accel_pragmas_from_c(source: Union[str, Path]) -> List[ParsedPragma]:
    if isinstance(source, Path) or (isinstance(source, str) and Path(source).is_file()):
        text = Path(source).read_text(encoding="utf-8", errors="ignore")
    else:
        text = str(source)
    lines = text.splitlines()
    pending: List[ParsedPragma] = []
    results: List[ParsedPragma] = []
    loop_index = 0
    pending_label: Optional[str] = None

    def _flush_loop(label: str) -> None:
        nonlocal pending, loop_index
        for p in pending:
            p.loop_label = label
            p.loop_index = loop_index
            results.append(p)
        pending = []
        loop_index += 1

    for line in lines:
        m = PRAGMA_RE.match(line)
        if m:
            parsed = _classify_body(m.group("body"))
            if parsed is not None:
                pending.append(parsed)
            continue
        same = LOOP_LABEL_SAME_LINE_RE.match(line)
        if same:
            _flush_loop(same.group("label"))
            pending_label = None
            continue
        only = LOOP_LABEL_ONLY_RE.match(line)
        if only:
            pending_label = only.group("label")
            continue
        if FOR_WHILE_RE.match(line):
            label = pending_label if pending_label else f"L{loop_index}"
            _flush_loop(label)
            pending_label = None
    for p in pending:
        p.loop_label = "GLOBAL"
        p.loop_index = -1
        results.append(p)
    return results


def pragmas_to_design_point(pragmas: List[ParsedPragma]) -> DesignPoint:
    point: DesignPoint = {}
    for p in pragmas:
        scope = p.loop_label or f"L{p.loop_index}"
        if p.kind == "PIPELINE":
            point[f"__PIPE__{scope}"] = p.value if p.value is not None else "on"
        elif p.kind == "TILE":
            point[f"__TILE__{scope}"] = int(p.value) if p.value is not None else 1
        elif p.kind in ("PARALLEL", "UNROLL"):
            point[f"__PARA__{scope}"] = (
                int(p.value) if isinstance(p.value, int) else p.value  # type: ignore
            )
            if p.reduction:
                point[f"__REDUCTION__{scope}"] = p.reduction
        else:
            point[f"__{p.kind}__{scope}"] = p.value if p.value is not None else ""
    return point


def parse_c_file_to_point(path: Union[str, Path]) -> DesignPoint:
    return pragmas_to_design_point(parse_accel_pragmas_from_c(path))


_points_cache: Dict[Tuple[str, str, str], List[Optional[DesignPoint]]] = {}


def _load_points_list(pkl_path: str) -> List[Optional[DesignPoint]]:
    key = ("pkl", pkl_path, "")
    if key in _points_cache:
        return _points_cache[key]
    with open(pkl_path, "rb") as f:
        pts = pickle.load(f)
    if not isinstance(pts, list):
        pts = list(pts)
    _points_cache[key] = pts
    return pts


def resolve_design_point(
    suite: str,
    kernel: str,
    cfg_id: int,
    points_roots: Optional[Sequence[str]] = None,
    c_src_roots: Optional[Sequence[str]] = None,
) -> Optional[DesignPoint]:
    """
    优先 points_list.pkl[cfg_id]，否则解析 {cfg_id}.c。
    """
    points_roots = list(points_roots or default_points_roots())
    c_src_roots = list(c_src_roots or default_c_src_roots())

    for root in points_roots:
        pkl = join(root, suite, kernel, "points_list.pkl")
        if not exists(pkl):
            continue
        pts = _load_points_list(pkl)
        if 0 <= int(cfg_id) < len(pts) and pts[int(cfg_id)] is not None:
            pt = pts[int(cfg_id)]
            if isinstance(pt, dict) and len(pt) > 0:
                return dict(pt)

    for root in c_src_roots:
        c_path = join(root, suite, kernel, f"{int(cfg_id)}.c")
        if exists(c_path):
            pt = parse_c_file_to_point(c_path)
            if pt:
                return pt
    return None


def collect_points_for_samples(
    data_list: Sequence[Any],
    points_roots: Optional[Sequence[str]] = None,
    c_src_roots: Optional[Sequence[str]] = None,
) -> List[Optional[DesignPoint]]:
    out: List[Optional[DesignPoint]] = []
    for d in data_list:
        suite = str(getattr(d, "suite", "") or "")
        kernel = str(getattr(d, "kernel", "") or "")
        cfg_id = int(getattr(d, "cfg_id", -1))
        out.append(
            resolve_design_point(
                suite, kernel, cfg_id, points_roots=points_roots, c_src_roots=c_src_roots
            )
        )
    return out


def attach_pragma_features(
    data_list: Sequence[Any],
    encoder: PragmaFeatureEncoder,
    points_roots: Optional[Sequence[str]] = None,
    c_src_roots: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    """给每个 Data 写入 pragma_feat [D]；返回统计。"""
    found = 0
    missing = 0
    for d in data_list:
        suite = str(getattr(d, "suite", "") or "")
        kernel = str(getattr(d, "kernel", "") or "")
        cfg_id = int(getattr(d, "cfg_id", -1))
        pt = resolve_design_point(
            suite, kernel, cfg_id, points_roots=points_roots, c_src_roots=c_src_roots
        )
        if pt:
            found += 1
        else:
            missing += 1
        feat = encoder.transform_one(pt)
        d.pragma_feat = torch.from_numpy(feat.astype("float32"))
    return {"found": found, "missing": missing, "dim": int(encoder.dim_)}


def fit_pragma_encoder_from_train(
    train_list: Sequence[Any],
    points_roots: Optional[Sequence[str]] = None,
    c_src_roots: Optional[Sequence[str]] = None,
) -> Tuple[PragmaFeatureEncoder, Dict[str, int]]:
    points = collect_points_for_samples(
        train_list, points_roots=points_roots, c_src_roots=c_src_roots
    )
    non_empty = [p for p in points if p]
    enc = PragmaFeatureEncoder()
    if not non_empty:
        # 空语料：仅 summary 6 维占位，保证可跑
        enc.fit([{"__PIPE__L0": "off", "__TILE__L0": 1, "__PARA__L0": 1}])
        stats = {"train_with_pragma": 0, "train_total": len(train_list), "dim": enc.dim_}
        return enc, stats
    enc.fit(non_empty)
    stats = {
        "train_with_pragma": len(non_empty),
        "train_total": len(train_list),
        "dim": enc.dim_,
        "n_keys": len(enc.keys_),
    }
    return enc, stats
