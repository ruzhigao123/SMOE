"""
固定长度 pragma / DesignPoint 编码器（适配不同 kernel 的 key 数目差异）。

对训练集做 union-of-keys fit；缺失 key 填 0 / UNK。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

DesignPoint = Dict[str, Union[str, int, float]]


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class PragmaFeatureEncoder:
    """
    Fit on a corpus of design points, then transform each point to a float vector.

    Encoding:
      - for each observed pragma key:
          * numeric value  -> log2(max(v,1)) normalized by corpus max
          * categorical    -> one-hot over seen categories (+ UNK)
      - summary stats: num_pragmas, mean_log_tile, mean_log_para, pipe on/flatten/off
    """

    def __init__(self):
        self.keys_: List[str] = []
        self.cat_vocab_: Dict[str, List[str]] = {}
        self.num_max_: Dict[str, float] = {}
        self.dim_: int = 0
        self.feature_names_: List[str] = []
        self._fitted = False

    def fit(self, points: Sequence[DesignPoint]) -> "PragmaFeatureEncoder":
        key_set = set()
        cat_values: Dict[str, set] = {}
        num_max: Dict[str, float] = {}

        for pt in points:
            if not pt:
                continue
            for k, v in pt.items():
                key_set.add(k)
                if _is_number(v):
                    num_max[k] = max(num_max.get(k, 1.0), float(abs(v)) + 1e-8)
                else:
                    cat_values.setdefault(k, set()).add(str(v).lower())

        self.keys_ = sorted(key_set)
        self.cat_vocab_ = {k: sorted(vs) + ["__UNK__"] for k, vs in cat_values.items()}
        self.num_max_ = num_max

        names: List[str] = []
        dim = 0
        for k in self.keys_:
            if k in self.cat_vocab_:
                for cat in self.cat_vocab_[k]:
                    names.append(f"{k}={cat}")
                    dim += 1
            else:
                names.append(f"{k}#lognorm")
                dim += 1

        for s in [
            "summary#num_pragmas",
            "summary#mean_log_tile",
            "summary#mean_log_para",
            "summary#num_pipe_on",
            "summary#num_pipe_flatten",
            "summary#num_pipe_off",
        ]:
            names.append(s)
            dim += 1

        self.feature_names_ = names
        self.dim_ = dim
        self._fitted = True
        return self

    def transform_one(self, point: Optional[DesignPoint]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PragmaFeatureEncoder must be fit() before transform")
        point = point or {}
        vec = np.zeros(self.dim_, dtype=np.float32)
        if self.dim_ <= 0:
            return vec
        idx = 0
        for k in self.keys_:
            if k in self.cat_vocab_:
                vocab = self.cat_vocab_[k]
                raw = point.get(k, None)
                token = str(raw).lower() if raw is not None else "__UNK__"
                if token not in vocab:
                    token = "__UNK__"
                one_hot_i = vocab.index(token)
                vec[idx + one_hot_i] = 1.0
                idx += len(vocab)
            else:
                v = point.get(k, 0)
                try:
                    fv = float(v)
                except Exception:
                    fv = 0.0
                denom = self.num_max_.get(k, 1.0)
                vec[idx] = math.log2(max(abs(fv), 1.0)) / max(math.log2(denom), 1e-6)
                idx += 1

        tiles = [float(v) for kk, v in point.items() if "TILE" in kk and _is_number(v)]
        paras = [float(v) for kk, v in point.items() if "PARA" in kk and _is_number(v)]
        pipes = [str(v).lower() for kk, v in point.items() if "PIPE" in kk]

        vec[idx] = float(len(point))
        idx += 1
        vec[idx] = (
            float(np.mean([math.log2(max(t, 1.0)) for t in tiles])) if tiles else 0.0
        )
        idx += 1
        vec[idx] = (
            float(np.mean([math.log2(max(p, 1.0)) for p in paras])) if paras else 0.0
        )
        idx += 1
        vec[idx] = float(sum(1 for p in pipes if p in ("on", "")))
        idx += 1
        vec[idx] = float(sum(1 for p in pipes if p == "flatten"))
        idx += 1
        vec[idx] = float(sum(1 for p in pipes if p == "off"))
        return vec

    def transform(self, points: Sequence[Optional[DesignPoint]]) -> np.ndarray:
        return np.stack([self.transform_one(p) for p in points], axis=0)

    def fit_transform(self, points: Sequence[DesignPoint]) -> np.ndarray:
        return self.fit(points).transform(points)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        payload = {
            "keys": self.keys_,
            "cat_vocab": self.cat_vocab_,
            "num_max": self.num_max_,
            "dim": self.dim_,
            "feature_names": self.feature_names_,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PragmaFeatureEncoder":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls()
        obj.keys_ = payload["keys"]
        obj.cat_vocab_ = payload["cat_vocab"]
        obj.num_max_ = {k: float(v) for k, v in payload["num_max"].items()}
        obj.dim_ = int(payload["dim"])
        obj.feature_names_ = payload.get("feature_names", [])
        obj._fitted = True
        return obj
