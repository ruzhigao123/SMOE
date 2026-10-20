"""
模态专家池（Low-level experts）。

默认 depth=2 的轻量残差专家；输出会进入 soft_mix，不再是死参数。
"""
from __future__ import annotations

import torch
import torch.nn as nn


EXPERT_MODE_NAMES = [
    "all",
    "ast_cdfg",
    "ast_text",
    "cdfg_text",
    "cdfg_only",
    "ast_only",
    "LM",
]

# 单模态方案：门控默认可禁用，防止错误塌缩
SINGLE_MODE_NAMES = frozenset({"cdfg_only", "ast_only", "LM"})

MODE_TO_MASK = {
    "all":       (1, 1, 1),
    "ast_cdfg":  (1, 0, 1),
    "ast_text":  (0, 1, 1),
    "cdfg_text": (1, 1, 0),
    "cdfg_only": (1, 0, 0),
    "ast_only":  (0, 0, 1),
    "LM":        (0, 1, 0),
}


class ResidualExpertMLP(nn.Module):
    """in → expert_dim，再叠 depth 个残差 FFN。"""

    def __init__(self, in_dim: int, expert_dim: int = 256, depth: int = 2, dropout: float = 0.05):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, expert_dim),
            nn.LayerNorm(expert_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(expert_dim),
                    nn.Linear(expert_dim, expert_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_dim * 2, expert_dim),
                )
                for _ in range(max(depth, 1))
            ]
        )
        self.out_norm = nn.LayerNorm(expert_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out_norm(h)


class ModalExperts(nn.Module):
    def __init__(
        self,
        cdfg_dim=128,
        text_dim=768,
        ast_dim=128,
        expert_dim=256,
        depth: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.cdfg_expert = ResidualExpertMLP(cdfg_dim, expert_dim, depth=depth, dropout=dropout)
        self.text_expert = ResidualExpertMLP(text_dim, expert_dim, depth=depth, dropout=dropout)
        self.ast_expert = ResidualExpertMLP(ast_dim, expert_dim, depth=depth, dropout=dropout)

    def forward(self, h_g, h_s, h_a):
        return torch.stack(
            [self.cdfg_expert(h_g), self.text_expert(h_s), self.ast_expert(h_a)],
            dim=1,
        )
