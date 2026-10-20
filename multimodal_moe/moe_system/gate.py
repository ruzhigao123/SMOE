"""
典型 Hierarchical Modal MoE 门控：按任务对 3 个模态专家做 soft routing。

防极化对齐 AAAI'25 HierarchicalMoE：
  - 全程 soft softmax（不用 hard Gumbel）
  - CV(importance) 负载均衡正则（Shazeer / 论文 LR）
  - 高层次控初始化为均匀权重
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experts import EXPERT_MODE_NAMES, MODE_TO_MASK, SINGLE_MODE_NAMES


class HierarchicalExpertGate(nn.Module):
    """
    High-level gate：每个任务独立线性层 → 3 维 logits → softmax。
    论文低层门控用 linear；此处按任务分头，适配多指标。
    """

    EXPERT_NAMES = ("CDFG", "Text", "AST")

    def __init__(
        self,
        gate_in_dim: int = 384,
        num_tasks: int = 5,
        num_experts: int = 3,
        hidden: int = 128,
        soft_epochs: int = 20,
    ):
        super().__init__()
        del soft_epochs  # 保留参数兼容旧调用；不再做 hard 切换
        self.num_experts = int(num_experts)
        # 论文：门控用线性层即可；我们保留一层非线性以稳住多模态尺度差
        self.gate_layer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(gate_in_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, self.num_experts),
                )
                for _ in range(num_tasks)
            ]
        )
        self._init_uniform_gate()

    def _init_uniform_gate(self):
        """对齐论文：高层门控初始化为均匀分配，减轻早期极化。"""
        for seq in self.gate_layer:
            last = seq[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, gate_feat, task_idx, epoch=None):
        del epoch
        if gate_feat.dim() == 1:
            gate_feat = gate_feat.view(1, -1)
        logits = self.gate_layer[task_idx](gate_feat)
        # 论文：全程 soft softmax，不做 hard one-hot
        weights = F.softmax(logits, dim=-1)
        expert_idx = weights.argmax(dim=-1)
        return weights, expert_idx, logits

    @staticmethod
    def cv_squared(x: torch.Tensor) -> torch.Tensor:
        """与 HierarchicalMoE-master 一致：CV = std / mean。"""
        if x.numel() <= 1:
            return x.new_zeros(())
        mean = x.float().mean().clamp_min(1e-10)
        return x.float().std(unbiased=False) / mean

    @staticmethod
    def compute_balance_loss(weights: torch.Tensor) -> torch.Tensor:
        """
        论文式 (4)：LR = CV(I(WG))，I = sum_i softmax(...) 或 batch 上 sum。
        weights: [B, E]
        """
        importance = weights.sum(dim=0)  # [E]
        return HierarchicalExpertGate.cv_squared(importance)


# ---------- 保留旧 7 方案门控（兼容旧实验，典型 MoE 不再使用）----------
class TaskModalGate(nn.Module):
    def __init__(
        self,
        gate_in_dim: int = 128 + 128 + 128,
        num_tasks: int = 5,
        num_experts: int = 3,
        hidden: int = 128,
        warmup_epochs: int = 0,
        top_k: int = 2,
        soft_epochs: int = 20,
        allow_single_modal: bool = True,
    ):
        super().__init__()
        del num_experts, warmup_epochs, top_k
        self.soft_epochs = int(soft_epochs)
        self.allow_single_modal = bool(allow_single_modal)
        self.mode_names = list(EXPERT_MODE_NAMES)
        masks = [MODE_TO_MASK[n] for n in self.mode_names]
        self.register_buffer("mode_masks", torch.tensor(masks, dtype=torch.float32))
        single_flags = [n in SINGLE_MODE_NAMES for n in self.mode_names]
        self.register_buffer("single_mode_flags", torch.tensor(single_flags, dtype=torch.bool))
        n_modes = len(self.mode_names)
        self.gate_layer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(gate_in_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, n_modes),
                )
                for _ in range(num_tasks)
            ]
        )

    def forward(self, gate_feat, task_idx, epoch=None, threshold=None):
        del threshold
        if gate_feat.dim() == 1:
            gate_feat = gate_feat.view(1, -1)
        logits = self.gate_layer[task_idx](gate_feat)
        if not self.allow_single_modal:
            logits = logits.masked_fill(self.single_mode_flags.unsqueeze(0), -1e9)
        ep = 0 if epoch is None else int(epoch)
        tau = max(0.5, 2.0 * (0.98 ** ep))
        if self.training:
            scheme = F.gumbel_softmax(logits, tau=tau, hard=(ep >= self.soft_epochs), dim=-1)
        else:
            scheme = F.one_hot(logits.argmax(dim=-1), num_classes=logits.size(-1)).float()
        modal_mask = scheme @ self.mode_masks
        scheme_idx = scheme.argmax(dim=-1)
        return modal_mask, scheme, scheme_idx, logits

    @staticmethod
    def compute_balance_loss(scheme: torch.Tensor) -> torch.Tensor:
        importance = scheme.sum(dim=0)
        mean = importance.float().mean().clamp_min(1e-10)
        return importance.float().std(unbiased=False) / mean


class SoftModalGate(nn.Module):
    def __init__(self, gate_in_dim=384, num_tasks=5, num_modes=7, hidden=64):
        super().__init__()
        self.num_modes = num_modes
        self.gate_layer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(gate_in_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, num_modes),
                )
                for _ in range(num_tasks)
            ]
        )
        masks = [MODE_TO_MASK[n] for n in EXPERT_MODE_NAMES]
        self.register_buffer("mode_masks", torch.tensor(masks, dtype=torch.float32))

    def forward(self, gate_feat, task_idx, epoch=None):
        del epoch
        if gate_feat.dim() == 1:
            gate_feat = gate_feat.view(1, -1)
        logits = self.gate_layer[task_idx](gate_feat)
        mode_probs = F.softmax(logits, dim=-1)
        modal_weights = mode_probs @ self.mode_masks
        return modal_weights / (modal_weights.sum(dim=-1, keepdim=True) + 1e-8), mode_probs
