"""Selector-inspired residual-risk modelling and first-order MLDG for soft MoE.

Adapt the graph/pragma selector's candidate features and supervision to the
existing frozen multimodal encodings. No expert is selected or discarded here.
Only training/support labels fit the feature state or enter the losses.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Dict, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_SELECTOR_CONFIG = {
    "temperature": 0.75,
    "aux_weight": 1.0,
    "supervised_epochs": 20,
    "mldg_steps": 32,
    "inner_lr": 2e-5,
    "meta_lr": 5e-5,
    "points_per_kernel": 24,
    "support_families": 4,
    "meta_families": 2,
}


def selector_config_from_args(args) -> dict:
    return {
        name: getattr(args, "selector_" + name, default)
        for name, default in DEFAULT_SELECTOR_CONFIG.items()
    }


def validate_selector_config(config: dict) -> None:
    for name in ("temperature", "inner_lr", "meta_lr"):
        if not 0 < float(config[name]) < float("inf"):
            raise ValueError(f"selector_{name} 必须为有限正数")
    if not 0 <= float(config["aux_weight"]) < float("inf"):
        raise ValueError("selector_aux_weight 必须为有限非负数")
    if int(config["supervised_epochs"]) < 0:
        raise ValueError("selector_supervised_epochs 必须非负")
    for name in ("mldg_steps", "points_per_kernel", "support_families", "meta_families"):
        if int(config[name]) < 1:
            raise ValueError(f"selector_{name} 必须为正整数")


def kernel_family(kernel: str) -> str:
    # Same family grouping used by the supplied selector.py.
    for prefix in ("gemm", "gemver", "gesummv", "mvt", "symm", "bicg",
                   "trmm", "fdtd", "doitgen", "stencil"):
        if kernel.startswith(prefix):
            return prefix
    return "syr" if kernel in {"syrk", "syr2k"} else kernel


def split_family_kernels(kernels: Sequence[str], ratio: float, seed: int):
    if not 0 < ratio < 1:
        raise ValueError("train_ratio 必须在 (0,1) 内")
    families = defaultdict(list)
    for kernel in dict.fromkeys(kernels):
        families[kernel_family(kernel)].append(kernel)
    names = sorted(families)
    if len(names) < 2:
        raise ValueError("selector 源域划分至少需要两个程序家族")
    random.Random(seed).shuffle(names)
    wanted = max(1, int(sum(map(len, families.values())) * ratio))
    train, val = [], []
    for index, name in enumerate(names):
        # Always reserve a whole final family for validation.
        destination = train if len(train) < wanted and index < len(names) - 1 else val
        destination.extend(families[name])
    return train, val


def embedding_summary(values: torch.Tensor) -> torch.Tensor:
    """Ten statistics per candidate, matching selector.py's latent summaries."""
    quantiles = torch.quantile(
        values.float(), values.new_tensor([0.25, 0.5, 0.75]), dim=-1
    ).movedim(0, -1)
    return torch.cat([
        values.mean(-1, keepdim=True),
        values.std(-1, unbiased=False, keepdim=True),
        values.amin(-1, keepdim=True), values.amax(-1, keepdim=True),
        values.abs().mean(-1, keepdim=True),
        values.square().mean(-1, keepdim=True).sqrt(),
        values.norm(dim=-1, keepdim=True) / values.shape[-1] ** 0.5,
        quantiles,
    ], dim=-1)


def sample_family_episode(kernels: Sequence[str], support_count: int,
                          meta_count: int, rng: random.Random):
    families = defaultdict(list)
    for kernel in dict.fromkeys(kernels):
        families[kernel_family(kernel)].append(kernel)
    if len(families) < 2:
        raise ValueError("MLDG 至少需要两个训练家族")
    meta_count = min(meta_count, max(1, len(families) // 3))
    support_count = min(support_count, len(families) - meta_count)
    selected = rng.sample(sorted(families), support_count + meta_count)
    support = [rng.choice(families[family]) for family in selected[:support_count]]
    meta = [rng.choice(families[family]) for family in selected[support_count:]]
    return support, meta


class ResidualRiskScorer(nn.Module):
    """Predict residual moments [B,T,3] from observable candidate evidence.

Candidate queries attend to frozen CDFG/Text/AST tokens and an optional pragma
token. This is the existing dataset's analogue of graph/pragma cross-attention;
it does not require the external selector's static-graph export schema.
"""

    def __init__(self, emb_dim: int, modal_dim: int, n_targets: int,
                 temperature: float = 0.75, hidden: int = 96):
        super().__init__()
        self.modal_dim = modal_dim
        self.n_targets = n_targets
        self.temperature = float(temperature)
        self.register_buffer("prediction_mean", torch.zeros(n_targets))
        self.register_buffer("prediction_std", torch.ones(n_targets))
        self.register_buffer("summary_mean", torch.zeros(3, 10))
        self.register_buffer("summary_std", torch.ones(3, 10))
        self.register_buffer("residual_scale", torch.ones(n_targets))
        self.register_buffer("statistics_fitted", torch.tensor(False))
        self.modal_encoder = nn.Sequential(nn.Linear(modal_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.modal_identity = nn.Embedding(3, hidden)
        extra = emb_dim - 3 * modal_dim
        self.pragma_encoder = nn.Linear(extra, hidden) if extra > 0 else None
        self.context_encoder = nn.Sequential(nn.Linear(emb_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.candidate_identity = nn.Embedding(3, hidden)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(3 * n_targets + 2 + 23, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(hidden, 4, dropout=0.15, batch_first=True) for _ in range(2)
        ])
        self.attention_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])
        self.feedforwards = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Dropout(0.15),
                          nn.Linear(4 * hidden, hidden)) for _ in range(2)
        ])
        self.feedforward_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])
        self.target_identity = nn.Embedding(n_targets, hidden)
        self.scorer = nn.Sequential(
            nn.Linear(8 * hidden, 192), nn.GELU(), nn.Dropout(0.15), nn.LayerNorm(192),
            nn.Linear(192, 192), nn.GELU(), nn.Dropout(0.15), nn.Linear(192, 2),
        )
        with torch.no_grad():
            self.scorer[-1].weight.mul_(0.05)
            self.scorer[-1].bias[0].zero_()
            self.scorer[-1].bias[1].fill_(-1.25)

    def candidate_summaries(self, gate_feat: torch.Tensor) -> torch.Tensor:
        modal = gate_feat[:, :3 * self.modal_dim].reshape(-1, 3, self.modal_dim)
        # Encoding order is CDFG/Text/AST; expert prediction order is CDFG/AST/Text.
        return embedding_summary(modal[:, [0, 2, 1]])

    @torch.no_grad()
    def fit_statistics(self, predictions: torch.Tensor, truth: torch.Tensor,
                       summaries: torch.Tensor) -> None:
        if predictions.ndim != 3 or tuple(predictions.shape[1:]) != (3, self.n_targets):
            raise ValueError("selector predictions 必须为 [N,3,T]")
        if tuple(truth.shape) != (len(predictions), self.n_targets):
            raise ValueError("selector truth 必须为 [N,T]")
        if tuple(summaries.shape) != (len(predictions), 3, 10) or not len(predictions):
            raise ValueError("selector summaries 必须为非空 [N,3,10]")
        if not all(torch.isfinite(value).all() for value in (predictions, truth, summaries)):
            raise ValueError("selector 训练统计包含非有限数值")
        flattened = predictions.float().reshape(-1, self.n_targets)
        mean, std = flattened.mean(0), flattened.std(0, unbiased=False)
        summary_mean, summary_std = summaries.mean(0), summaries.std(0, unbiased=False)
        residual = (predictions - truth[:, None]).reshape(-1, self.n_targets)
        median = residual.median(0).values
        mad = (residual - median).abs().median(0).values * 1.4826
        rms = residual.square().mean(0).sqrt()
        scale = torch.where(mad > 1e-4, mad, rms)
        for destination, value in (
            (self.prediction_mean, mean),
            (self.prediction_std, torch.where(std < 1e-4, torch.ones_like(std), std)),
            (self.summary_mean, summary_mean),
            (self.summary_std, torch.where(summary_std < 1e-4, torch.ones_like(summary_std), summary_std)),
            (self.residual_scale, torch.where(scale > 1e-4, scale, torch.ones_like(scale))),
        ):
            destination.copy_(value.to(destination))
        self.statistics_fitted.fill_(True)

    def forward(self, emb: torch.Tensor, predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not bool(self.statistics_fitted):
            raise RuntimeError("selector 门控必须先用训练集拟合统计量，或加载 selector checkpoint")
        if tuple(predictions.shape) != (len(emb), 3, self.n_targets):
            raise ValueError("expert_preds 必须为 [B,3,T]")
        standardized = ((predictions - self.prediction_mean) / self.prediction_std).clamp(-8, 8)
        center = predictions.mean(1, keepdim=True)
        spread = predictions.std(1, unbiased=False, keepdim=True).clamp_min(1e-5)
        disagreement = ((predictions - center) / spread).clamp(-8, 8)
        absolute = disagreement.abs()
        prediction_features = torch.cat([
            standardized, disagreement, absolute,
            absolute.mean(2, keepdim=True), absolute.amax(2, keepdim=True),
        ], dim=-1)
        z = ((self.candidate_summaries(emb) - self.summary_mean) / self.summary_std).clamp(-8, 8)
        summary_features = torch.cat([
            z, z.abs(), z.square().mean(-1, keepdim=True).sqrt(),
            z.abs().amax(-1, keepdim=True), torch.ones_like(z[..., :1]),
        ], dim=-1)
        candidate = self.candidate_encoder(torch.cat([prediction_features, summary_features], dim=-1))
        candidate = candidate + self.candidate_identity.weight[None]
        context = self.context_encoder(emb)
        tokens = self.modal_encoder(emb[:, :3 * self.modal_dim].reshape(-1, 3, self.modal_dim))
        tokens = tokens + self.modal_identity.weight[None]
        if self.pragma_encoder is not None:
            tokens = torch.cat([tokens, self.pragma_encoder(emb[:, 3 * self.modal_dim:])[:, None]], dim=1)
        evidence = candidate + context[:, None]
        for attention, norm, feedforward, ff_norm in zip(
            self.attention, self.attention_norms, self.feedforwards, self.feedforward_norms
        ):
            attended, _ = attention(evidence, tokens, tokens, need_weights=False)
            evidence = norm(evidence + attended)
            evidence = ff_norm(evidence + feedforward(evidence))
        context = context[:, None].expand(-1, 3, -1)
        base = torch.cat([context, candidate, evidence, context * candidate, context * evidence], dim=-1)
        base = base[:, None].expand(-1, self.n_targets, -1, -1)
        target = self.target_identity.weight[None, :, None].expand(len(emb), -1, 3, -1)
        features = torch.cat([base, target, candidate[:, None] * target, evidence[:, None] * target], dim=-1)
        moments = self.scorer(features)
        mean = moments[..., 0]
        variance = F.softplus(moments[..., 1]) + 1e-4
        risk = mean.square() + variance
        return {"mean": mean, "variance": variance, "risk": risk,
                "logits": -risk / self.temperature}


def residual_risk_loss(output: dict, predictions: torch.Tensor, truth: torch.Tensor,
                       scale: torch.Tensor, epoch: int, meta: bool = False):
    """NLL + risk calibration + listwise ranking + margin-weighted winner CE."""
    errors = (predictions.detach() - truth[:, None]).permute(0, 2, 1)
    residual = (errors / scale[None, :, None]).clamp(-8, 8)
    observed_risk = residual.square()
    nll = 0.5 * ((residual - output["mean"]).square() / output["variance"]
                 + output["variance"].log()).mean(dim=(0, 2))
    calibration = F.smooth_l1_loss(torch.log1p(output["risk"]), torch.log1p(observed_risk),
                                   beta=0.25, reduction="none").mean(dim=(0, 2))
    soft_target = F.softmax(-observed_risk / 0.75, dim=-1)
    ranking = -(soft_target * F.log_softmax(output["logits"], dim=-1)).sum(-1).mean(0)
    raw_errors = errors.square()
    winner = raw_errors.argmin(-1)
    best_two = raw_errors.sort(-1).values[..., :2]
    margin = (best_two[..., 1] - best_two[..., 0]) / best_two[..., 1].clamp_min(1e-8)
    # Gradually include ambiguous points; clear winners retain their weight.
    progress = min(max(float(epoch) / 20.0, 0.0), 1.0)
    weight = 0.2 + 0.8 * torch.maximum((margin / 0.2).clamp(0, 1), margin.new_tensor(progress))
    ce = F.cross_entropy(output["logits"].reshape(-1, 3), winner.reshape(-1),
                         label_smoothing=0.02, reduction="none").view_as(weight)
    classification = (ce * weight).sum(0) / weight.sum(0).clamp_min(1.0)
    coefficients = (0.15, 0.1, 0.2, 1.0) if meta else (1.0, 0.25, 0.4, 0.2)
    parts = {"selector_nll": nll, "selector_calibration": calibration,
             "selector_ranking": ranking, "selector_classification": classification}
    losses = sum(coefficient * value for coefficient, value in zip(coefficients, parts.values()))
    return losses, parts


def first_order_mldg_step(model: nn.Module, optimizer: torch.optim.Optimizer,
                          support_objective: Callable, meta_objective: Callable,
                          inner_lr: float) -> dict:
    """Temporary support update, target-balanced meta gradients, exact restore.

The callbacks return one loss per target. No validation/query data belongs in
either callback. Frozen experts never enter the parameter list.
"""
    parameters = list(model.trainable_parameters())
    optimizer.zero_grad(set_to_none=True)
    support_targets = support_objective()
    support_loss = support_targets.mean()
    support_gradients = torch.autograd.grad(support_loss, parameters, allow_unused=True)
    originals = [parameter.detach().clone() for parameter in parameters]
    with torch.no_grad():
        for parameter, gradient in zip(parameters, support_gradients):
            if gradient is not None:
                parameter.add_(gradient, alpha=-inner_lr)
    try:
        meta_targets = meta_objective()
        target_gradients = [
            torch.autograd.grad(meta_targets[index], parameters,
                                retain_graph=index + 1 < len(meta_targets), allow_unused=True)
            for index in range(len(meta_targets))
        ]
        norms = torch.stack([
            sum((g.square().sum() for g in gradients if g is not None),
                meta_targets.new_zeros(())).clamp_min(1e-16).sqrt()
            for gradients in target_gradients
        ])
        scales = norms.detach().median().clamp_min(1e-8) / norms.detach().clamp_min(1e-8)
        meta_gradients = []
        for index in range(len(parameters)):
            pieces = [gradients[index] * scales[target] for target, gradients in enumerate(target_gradients)
                      if gradients[index] is not None]
            meta_gradients.append(torch.stack(pieces).mean(0) if pieces else None)
    finally:
        with torch.no_grad():
            for parameter, original in zip(parameters, originals):
                parameter.copy_(original)
    for parameter, support, meta_gradient in zip(parameters, support_gradients, meta_gradients):
        pieces = []
        if support is not None:
            pieces.append(0.25 * support)
        if meta_gradient is not None:
            pieces.append(0.75 * meta_gradient)
        parameter.grad = sum(pieces) if pieces else None
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    if not torch.isfinite(gradient_norm):
        raise RuntimeError("selector MLDG 产生非有限梯度")
    optimizer.step()
    return {"support_loss": float(support_loss.detach()), "meta_loss": float(meta_targets.mean().detach()),
            "target_gradient_cv": float(norms.std(unbiased=False) / norms.mean().clamp_min(1e-8)),
            "gradient_norm": float(gradient_norm)}
