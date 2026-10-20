"""构建对比实验用 GNN（SOTA 骨干），输出图级向量 [B, D]。"""
from __future__ import annotations

from typing import Any


def build_comparative_gnn(in_channels: int, flags: Any = None):
    """
    按 FLAGS.comparative_model 构造对照 GNN。
    约定 forward(data) → [B, FLAGS.D]（与 CDFGEncoder 输出对齐）。
    """
    if flags is None:
        from multimodal_moe.src.config import FLAGS as flags

    name = str(getattr(flags, "comparative_model", "ironman")).lower().strip()
    D = int(getattr(flags, "D", 128))
    hidden = int(getattr(flags, "hidden_num", 128))
    drop = float(getattr(flags, "dropout", 0.1))
    n_layers = int(getattr(flags, "gnn_layers", 6))

    if name == "ironman":
        from .Ironman import GCNNet

        return GCNNet(in_channels=in_channels)
    if name in ("ironman_no_edge", "ironman_no_edage"):
        from .ironman_no_edage import GCNNet_NO_Edage

        return GCNNet_NO_Edage(
            in_channels=in_channels,
            hidden_channels=D,
            num_layers=n_layers,
            drop_out=0,
        )
    if name == "hgp":
        from .HGP import HierNet

        return HierNet(
            in_channels, hidden, num_layers=3, conv_type="sage", drop_out=0.0
        )
    if name == "pna":
        from .pna import PNANet

        return PNANet(
            in_dim=in_channels,
            deg=None,
            num_layer=int(getattr(flags, "num_layers", 2) or 2),
            emb_dim=hidden,
        )
    if name in ("gnn-dse", "gnn_dse"):
        from .GNN_DSE import Net as GNN_DSE_Net

        return GNN_DSE_Net(in_channels, hidden)
    if name == "gin":
        from .GIN import GIN

        return GIN(
            in_channels=in_channels,
            hidden_channels=D,
            num_layers=6,
            drop_out=0.2,
        )
    if name == "graphsage":
        from .GraphSAGE import GraphSAGE

        return GraphSAGE(
            in_channels=in_channels,
            hidden_channels=D,
            num_layers=3,
            drop_out=0.1,
        )
    if name == "gat":
        from .GAT import GAT

        return GAT(
            in_channels=in_channels,
            hidden_channels=D,
            num_layers=6,
            drop_out=0.2,
        )
    if name in ("sotampm", "mpm", "cognn", "mpm_cognn"):
        from .sotampm import CoGNNNet

        edge_dim = int(getattr(flags, "cognn_edge_dim", 7) or 7)
        return CoGNNNet(
            in_channels=in_channels,
            edge_attr_dim=edge_dim,
            flags=flags,
            D=D,
        )
    # 默认 / gcn
    from .GCN import GCN

    return GCN(
        in_channels=in_channels,
        hidden_channels=hidden,
        num_layers=5,
        drop_out=0.2,
    )


__all__ = ["build_comparative_gnn"]
