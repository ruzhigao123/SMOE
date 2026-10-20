#
# from multimodal_moe.src.config import FLAGS
#
#
# import torch
# import torch.nn.functional as F
# from torch_geometric.loader import DataLoader
# from torch_geometric.data import Data
# from torch_geometric.nn import GATConv, GlobalAttention, JumpingKnowledge, TransformerConv, GCNConv
# from torch_geometric.nn import global_add_pool
# import torch.nn as nn
# from scipy.stats import rankdata, kendalltau
#
# from multimodal_moe.src.nn_att import MyGlobalAttention
# from torch.nn import Sequential, Linear, ReLU
#
#
#
#
# class Net(torch.nn.Module):
#     def __init__(self, in_channels, edge_dim = 7, init_pragma_dict = None, task = FLAGS.task, num_layers = FLAGS.num_layers, D = FLAGS.D, target = FLAGS.target):
#         super(Net, self).__init__()
#         self.D = D
#         conv_class = TransformerConv
#         self.conv_first = conv_class(in_channels, D)
#         self.conv_layers = nn.ModuleList()
#         for _ in range(num_layers - 1):
#             conv = conv_class(D, D)
#             self.conv_layers.append(conv)
#         # M6，每一conv_layers后加入JKN
#         self.jkn = JumpingKnowledge( 'max', channels=D, num_layers=2)
#         self.gate_nn = nn.Sequential(nn.Linear(self.D, self.D), ReLU(), Linear(self.D, self.D))
#         self.glob = MyGlobalAttention(self.gate_nn, None)
#
#     def forward(self, data):
#         x, edge_index, edge_attr, batch= \
#             data.x, data.edge_index, data.edge_attr, data.batch
#         outs = []
#         activation = F.elu
#         out = activation(self.conv_first(x, edge_index))
#
#         outs.append(out)
#
#         for i, conv in enumerate(self.conv_layers):
#             if FLAGS.encode_edge and  FLAGS.gnn_type == 'transformer':
#                 out = conv(out, edge_index, edge_attr=edge_attr)
#             else:
#                 out = conv(out, edge_index)
#             if i != len(self.conv_layers) - 1:
#                 out = activation(out)
#
#             outs.append(out)
#             out = self.jkn(outs)
#
#         out, _ = self.glob(out, batch)
#
#         return out

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, JumpingKnowledge, TransformerConv, GCNConv
from torch_geometric.nn import global_add_pool
from torch.nn import Sequential, Linear, ReLU

# IMPORTANT: align FLAGS import with the rest of the repo.
# The original upstream used `from config import FLAGS` inside its own tree,
# but in this repo we should use `src.config` to avoid importing a different module.
from multimodal_moe.src.config import FLAGS  # type: ignore
from multimodal_moe.src.nn_att import MyGlobalAttention  # type: ignore


class Net(torch.nn.Module):
    """
    Graph encoder adapted from `GNN-DSE-main/src/model.py::Net`.

    In this repo it is used as a backbone encoder only (see `src/model.py`),
    so it returns a graph embedding tensor [batch, D] and does NOT include QoR heads.
    """

    def __init__(self, in_channels: int, edge_dim: int = 7, num_layers: int = None, D: int = None):
        super().__init__()
        self.D = int(D if D is not None else FLAGS.D)
        num_layers = int(num_layers if num_layers is not None else FLAGS.num_layers)

        gnn_type = getattr(FLAGS, "gnn_type", "transformer")
        if gnn_type == "gat":
            conv_class = GATConv
        elif gnn_type == "gcn":
            conv_class = GCNConv
        elif gnn_type == "transformer":
            conv_class = TransformerConv
        else:
            raise NotImplementedError(f"Unsupported FLAGS.gnn_type={gnn_type}")

        self._is_transformer = conv_class is TransformerConv
        self._use_edge = bool(getattr(FLAGS, "encode_edge", False) and self._is_transformer)

        # conv_first + conv_layers (match upstream: only pass edge_dim when needed)
        if self._use_edge:
            self.conv_first = conv_class(in_channels, self.D, edge_dim=edge_dim)
        else:
            self.conv_first = conv_class(in_channels, self.D)

        self.conv_layers = nn.ModuleList()
        for _ in range(max(0, num_layers - 1)):
            if self._use_edge:
                conv = conv_class(self.D, self.D, edge_dim=edge_dim)
            else:
                conv = conv_class(self.D, self.D)
            self.conv_layers.append(conv)

        # JK: apply once after collecting all layer outputs (upstream behavior).
        self.jkn = JumpingKnowledge(getattr(FLAGS, "jkn_mode", "max"), channels=self.D, num_layers=2)
        self.jkn_enable = bool(getattr(FLAGS, "jkn_enable", True))

        # Pooling: upstream behavior (node_attention -> attention pool, else global_add_pool)
        self.node_attention = bool(getattr(FLAGS, "node_attention", True))
        if self.node_attention:
            self.gate_nn = Sequential(Linear(self.D, self.D), ReLU(), Linear(self.D, 1))
            self.glob = MyGlobalAttention(self.gate_nn, None)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        batch = data.batch

        act = getattr(FLAGS, "activation", "elu")
        if act == "relu":
            activation = F.relu
        elif act == "elu":
            activation = F.elu
        else:
            raise NotImplementedError(f"Unsupported FLAGS.activation={act}")

        outs = []
        if self._use_edge and edge_attr is not None:
            out = activation(self.conv_first(x, edge_index, edge_attr=edge_attr))
        else:
            out = activation(self.conv_first(x, edge_index))
        outs.append(out)

        for i, conv in enumerate(self.conv_layers):
            if self._use_edge and edge_attr is not None:
                out = conv(out, edge_index, edge_attr=edge_attr)
            else:
                out = conv(out, edge_index)
            if i != len(self.conv_layers) - 1:
                out = activation(out)
            outs.append(out)

        if self.jkn_enable:
            out = self.jkn(outs)

        if self.node_attention:
            out, _ = self.glob(out, batch)
        else:
            out = global_add_pool(out, batch)

        return out