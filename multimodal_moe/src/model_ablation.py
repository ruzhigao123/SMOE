"""
消融模型（最初单模态方案 + 可选多模态融合）。

CDFG 默认对齐旧 MPM：
  GCNConv × 5, D=128, dropout=0.2, ReLU, lr=1e-3（src/config.py）
可选 gnn_type='transformer' 对齐 HierarchicalMoE 论文骨干。

单模态：encode → MLP(D)
多模态：TripleModalFusion + 置零 → MLP(256)
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, global_add_pool, global_max_pool
from torch_geometric.nn.dense import Linear as PyGLinear

try:
    from .utils import MLP, _get_y_with_target
except ImportError:  # 直接运行本文件时无包上下文
    import sys
    from os.path import abspath, dirname

    _SRC = dirname(abspath(__file__))
    _HMOE = dirname(dirname(_SRC))
    if _HMOE not in sys.path:
        sys.path.insert(0, _HMOE)
    from multimodal_moe.src.utils import MLP, _get_y_with_target

# (use_cdfg, use_text, use_ast)
MODE_TO_MASK = {
    "all": (1, 1, 1),
    "ast_cdfg": (1, 0, 1),
    "ast_text": (0, 1, 1),
    "cdfg_text": (1, 1, 0),
    "cdfg_only": (1, 0, 0),
    "ast_only": (0, 0, 1),
    "LM": (0, 1, 0),
}

ABLATION_MODES = list(MODE_TO_MASK.keys())
MULTI_MODAL_MODES = {m for m, mask in MODE_TO_MASK.items() if sum(mask) >= 2}


class CDFGEncoder(nn.Module):
    """
    comparative_if=False（默认）:
      gnn_type=gcn → 旧 MPM GCNConv；transformer → HierarchicalMoE TransformerConv
    comparative_if=True:
      使用 src/comp_model 中 FLAGS.comparative_model（如 ironman）作为 SOTA 骨干。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 7,
        drop_out: float = 0.2,
        D: int = 128,
        gnn_type: str = "gcn",
        activation: str = "relu",
        comparative_if: bool = None,
        comparative_model: str = None,
    ):
        super().__init__()
        self.drop_out = float(drop_out)
        self.num_layers = int(num_layers)
        self.D = int(D)
        self.gnn_type = str(gnn_type).lower()
        self.activation = str(activation).lower()
        self.comp_backend = None

        # 未显式传入时读 FLAGS
        if comparative_if is None or comparative_model is None:
            try:
                from .config import FLAGS as _F
            except Exception:
                _F = None
            if comparative_if is None:
                comparative_if = bool(getattr(_F, "comparative_if", False)) if _F else False
            if comparative_model is None:
                comparative_model = (
                    str(getattr(_F, "comparative_model", "ironman")) if _F else "ironman"
                )

        self.comparative_if = bool(comparative_if)
        self.comparative_model = str(comparative_model).lower()

        if self.comparative_if:
            from .comp_model import build_comparative_gnn
            # Explicit constructor settings must not be overwritten by runtime FLAGS.
            import copy
            from .config import FLAGS
            backend_flags = copy.copy(FLAGS)
            backend_flags.comparative_model = self.comparative_model
            backend_flags.D = self.D
            backend_flags.hidden_num = hidden_channels
            backend_flags.gnn_layers = self.num_layers
            backend_flags.dropout = self.drop_out
            self.comp_backend = build_comparative_gnn(in_channels, flags=backend_flags)
            return

        if self.gnn_type not in ("transformer", "gcn"):
            raise ValueError(f"gnn_type 应为 transformer|gcn，得到 {gnn_type}")
        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            in_c = in_channels if i == 0 else hidden_channels
            if self.gnn_type == "transformer":
                self.convs.append(
                    TransformerConv(in_c, hidden_channels, dropout=self.drop_out)
                )
            else:
                self.convs.append(GCNConv(in_c, hidden_channels))
        self.first_MLP = PyGLinear(hidden_channels * 2, self.D)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "elu":
            return F.elu(x)
        return F.relu(x)

    def forward(self, data) -> torch.Tensor:
        if self.comp_backend is not None:
            # SOTA 骨干：直接返回图级 [B, D]
            if hasattr(data, "x") and data.x is not None and data.x.dtype != torch.float32:
                data.x = data.x.float()
            return self.comp_backend(data)
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self._act(x)
            x = F.dropout(x, p=self.drop_out, training=self.training)
        x = torch.cat([global_max_pool(x, batch), global_add_pool(x, batch)], dim=1)
        return self.first_MLP(x)


class TripleModalFusionBlock(nn.Module):
    """MPM TripleModalFusionBlock（仅多模态使用）。"""

    def __init__(
        self,
        g_dim: int = 128,
        s_dim: int = 768,
        a_dim: int = 784,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_g = nn.Linear(g_dim, embed_dim)
        self.proj_s = nn.Linear(s_dim, embed_dim)
        self.proj_a = nn.Linear(a_dim, embed_dim)
        self.attn_gs = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn_gsa = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.gate = nn.Sequential(nn.Linear(embed_dim * 2, 1), nn.Sigmoid())

    def forward(self, h_g, h_s, h_a):
        g_emb = self.proj_g(h_g).unsqueeze(1)
        s_emb = self.proj_s(h_s).unsqueeze(1)
        a_emb = self.proj_a(h_a).unsqueeze(1)
        h_gs, _ = self.attn_gs(query=g_emb, key=s_emb, value=s_emb)
        h_gs = self.norm1(h_gs + g_emb)
        h_fuse = self.norm2(h_fuse + h_gs).squeeze(1)
        g_proj = g_emb.squeeze(1)
        gate_val = self.gate(torch.cat([g_proj, h_fuse], dim=-1))
        return gate_val * g_proj + (1.0 - gate_val) * h_fuse


class AblationNet(nn.Module):
    """
    单模态（最初方案）: encode → MLP(D)
    多模态: TripleModalFusion + 置零 → MLP(256)
    CDFG 默认 GCNConv（旧 MPM）。
    """

    def __init__(
        self,
        in_cdfg: int,
        target_list,
        ablation_mode: str = "cdfg_only",
        D: int = 128,
        hidden_num: int = 128,
        gnn_layers: int = 5,
        dropout: float = 0.2,
        out_dim: int = 1,
        fusion_embed_dim: int = 256,
        fusion_heads: int = 8,
        gnn_type: str = "gcn",
        gnn_activation: str = "relu",
        comparative_if: bool = None,
        comparative_model: str = None,
    ):
        super().__init__()
        if ablation_mode not in MODE_TO_MASK:
            raise ValueError(
                f"Unknown ablation_mode={ablation_mode}, choose from {ABLATION_MODES}"
            )
        self.ablation_mode = ablation_mode
        self.mask = MODE_TO_MASK[ablation_mode]
        self.target_list = list(target_list)
        self.D = int(D)
        self.gnn_type = str(gnn_type).lower()
        self.use_fusion = ablation_mode in MULTI_MODAL_MODES
        self.fusion_embed_dim = int(fusion_embed_dim)

        self.cdfg_enc = CDFGEncoder(
            in_channels=in_cdfg,
            hidden_channels=hidden_num,
            num_layers=gnn_layers,
            drop_out=dropout,
            D=self.D,
            gnn_type=self.gnn_type,
            activation=gnn_activation,
            comparative_if=comparative_if,
            comparative_model=comparative_model,
        )
        self.text_proj = nn.Sequential(nn.Linear(768, self.D), nn.ReLU(), nn.Linear(self.D, self.D))
        self.ast_proj = nn.Sequential(nn.Linear(784, self.D), nn.ReLU(), nn.Linear(self.D, self.D))

        if self.use_fusion:
            self.fusion_block = TripleModalFusionBlock(
                g_dim=self.D,
                s_dim=768,
                a_dim=784,
                embed_dim=self.fusion_embed_dim,
                num_heads=fusion_heads,
                dropout=0.1,
            )
            head_in = self.fusion_embed_dim
        else:
            self.fusion_block = None
            n_active = sum(self.mask)
            if n_active <= 0:
                raise ValueError("ablation mask empty")
            head_in = self.D * n_active

        self.MLPs = nn.ModuleDict(
            {
                t: MLP(
                    input_dim=head_in,
                    output_dim=out_dim,
                    activation_type="relu",
                    num_hidden_lyr=2,
                    hidden_channels=[hidden_num, hidden_num],
                )
                for t in self.target_list
            }
        )
        self.loss_fn = nn.MSELoss()

    def _batch_size(self, data) -> int:
        if hasattr(data, "batch") and data.batch is not None:
            return int(data.batch.max().item()) + 1
        if hasattr(data, "text_feat") and data.text_feat is not None:
            t = data.text_feat
            return int(t.size(0)) if t.dim() > 1 else 1
        return 1

    def encode_single(self, data) -> torch.Tensor:
        """最初 only-* 路径。"""
        use_g, use_s, use_a = self.mask
        parts = []
        if use_g:
            parts.append(self.cdfg_enc(data))
        if use_s:
            parts.append(self.text_proj(data.text_feat.float()))
        if use_a:
            parts.append(self.ast_proj(data.ast_feat.float()))
        return torch.cat(parts, dim=-1)

    def encode_fusion(self, data) -> torch.Tensor:
        use_g, use_s, use_a = self.mask
        B = self._batch_size(data)
        device = data.x.device

        if use_g:
            h_g = self.cdfg_enc(data)
        else:
            h_g = torch.zeros(B, self.D, device=device, dtype=torch.float32)

        if use_s:
            h_s = data.text_feat.float()
            if h_s.dim() == 1:
                h_s = h_s.view(B, -1)
        else:
            h_s = torch.zeros(B, 768, device=device, dtype=torch.float32)

        if use_a:
            h_a = data.ast_feat.float()
            if h_a.dim() == 1:
                h_a = h_a.view(B, -1)
        else:
            h_a = torch.zeros(B, 784, device=device, dtype=torch.float32)

        if self.ablation_mode == "ast_text":
            h_g = torch.zeros_like(h_g)
        elif self.ablation_mode == "ast_cdfg":
            h_s = torch.zeros_like(h_s)
        elif self.ablation_mode == "cdfg_text":
            h_a = torch.zeros_like(h_a)

        return self.fusion_block(h_g=h_g, h_s=h_s, h_a=h_a)

    def encode(self, data) -> torch.Tensor:
        if self.use_fusion:
            return self.encode_fusion(data)
        return self.encode_single(data)

    def forward(self, data) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        h = self.encode(data)
        out_dict, loss_dict = {}, {}
        total = 0.0
        for tname in self.target_list:
            pred = self.MLPs[tname](h)
            out_dict[tname] = pred
            y = _get_y_with_target(data, tname)
            if y is None:
                continue
            target = y.view(len(y), -1).float()
            if target.size(1) != pred.size(1):
                target = target[:, : pred.size(1)]
            loss = self.loss_fn(pred, target)
            total = total + loss
            loss_dict[tname] = loss
        if isinstance(total, float):
            total = torch.zeros((), device=h.device)
        return out_dict, total, loss_dict
