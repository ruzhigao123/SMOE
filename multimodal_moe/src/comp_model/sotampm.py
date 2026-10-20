"""
SOTA-MPM 骨干：MPM-Innovation 中的 CoGNN（图级向量 [B, D]）。

与 MPM src/model.py::CoGNN 对齐：Gumbel action + env GNN + GlobalAttention 池化。
供 comparative_model=sotampm / mpm 使用。
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Identity, LayerNorm, Linear, Module, ReLU, Sequential
from torch_geometric.typing import Adj

from multimodal_moe.src.config import FLAGS
from multimodal_moe.src.nn_att import MyGlobalAttention
from multimodal_moe.src.utils import MLP

from .CoGNN import (
    ActionNet,
    ActionNetArgs,
    ActivationType,
    EnvArgs,
    GumbelArgs,
    ModelType,
    TempSoftPlus,
)


def _gin_mlp_func() -> Callable:
    def mlp_func(in_channels: int, out_channels: int, bias: bool):
        return Sequential(
            Linear(in_channels, out_channels, bias=bias),
            ReLU(),
            Linear(out_channels, out_channels, bias=bias),
        )

    return mlp_func


def _model_type(name: str) -> ModelType:
    if isinstance(name, ModelType):
        return name
    return ModelType.from_string(str(name))


def build_cognn_args(flags=None, out_dim: Optional[int] = None):
    """用 FLAGS（或传入对象）构造与 MPM 一致的 CoGNN 超参。"""
    f = flags if flags is not None else FLAGS
    gin_mlp = _gin_mlp_func()
    D = int(out_dim if out_dim is not None else getattr(f, "D", 128))
    env_dim = int(getattr(f, "env_dim", D) or D)
    dropout = float(getattr(f, "dropout", 0.2))

    gumbel_args = GumbelArgs(
        learn_temp=bool(getattr(f, "learn_temp", False)),
        temp_model_type=_model_type(getattr(f, "temp_model_type", "LIN")),
        tau0=float(getattr(f, "tau0", 0.5)),
        temp=float(getattr(f, "temp", 0.01)),
        gin_mlp_func=gin_mlp,
    )
    env_args = EnvArgs(
        model_type=_model_type(getattr(f, "env_model_type", "SUM_GNN")),
        num_layers=int(getattr(f, "env_num_layers", 3)),
        env_dim=env_dim,
        layer_norm=bool(getattr(f, "layer_norm", False)),
        skip=bool(getattr(f, "skip", False)),
        batch_norm=bool(getattr(f, "batch_norm", False)),
        dropout=dropout,
        in_dim=int(getattr(f, "num_features", env_dim) or env_dim),
        out_dim=D,
        dec_num_layers=int(getattr(f, "dec_num_layers", 1)),
        gin_mlp_func=gin_mlp,
        act_type=ActivationType.RELU,
    )
    action_args = ActionNetArgs(
        model_type=_model_type(getattr(f, "act_model_type", "MEAN_GNN")),
        num_layers=int(getattr(f, "act_num_layers", 2)),
        hidden_dim=int(getattr(f, "act_dim", 16)),
        dropout=dropout,
        act_type=ActivationType.RELU,
        gin_mlp_func=gin_mlp,
        env_dim=env_dim,
    )
    return gumbel_args, env_args, action_args


class CoGNNNet(Module):
    """
    MPM CoGNN 封装：forward(data) → [B, D]。

    in_channels / edge_attr_dim 可随消融 CDFG 变化；缺 edge_attr 时用零填充。
    """

    def __init__(
        self,
        in_channels: int,
        edge_attr_dim: int = 7,
        flags=None,
        D: Optional[int] = None,
    ):
        super().__init__()
        f = flags if flags is not None else FLAGS
        self.in_channels = int(in_channels)
        self.edge_attr_dim = int(edge_attr_dim)
        self.D = int(D if D is not None else getattr(f, "D", 128))
        act = str(getattr(f, "gnn_activation", getattr(f, "activation", "relu")))

        gumbel_args, env_args, action_args = build_cognn_args(f, out_dim=self.D)
        self.env_args = env_args
        self.learn_temp = gumbel_args.learn_temp
        self.first_MLP_env_attr = MLP(
            self.edge_attr_dim, env_args.env_dim, activation_type=act
        )
        self.first_MLP_act_attr = MLP(
            self.edge_attr_dim, action_args.hidden_dim, activation_type=act
        )
        self.first_MLP_node = MLP(
            self.in_channels, env_args.env_dim, activation_type=act
        )
        if gumbel_args.learn_temp:
            self.temp_model = TempSoftPlus(
                gumbel_args=gumbel_args, env_dim=env_args.env_dim
            )
        self.temp = gumbel_args.temp

        self.num_layers = env_args.num_layers
        self.env_net = env_args.load_net()
        layer_norm_cls = LayerNorm if env_args.layer_norm else Identity
        self.hidden_layer_norm = layer_norm_cls(env_args.env_dim)
        self.skip = env_args.skip
        self.dropout = Dropout(p=env_args.dropout)
        self.act = env_args.act_type.get()
        self.in_act_net = ActionNet(action_args=action_args)
        self.out_act_net = ActionNet(action_args=action_args)

        self.gate_nn = Sequential(
            Linear(self.D, self.D), ReLU(), Linear(self.D, self.D)
        )
        self.glob = MyGlobalAttention(self.gate_nn, None)

    def _fix_edge_attr(self, edge_attr: Optional[Tensor], num_edges: int, device) -> Tensor:
        if edge_attr is None:
            return torch.zeros(num_edges, self.edge_attr_dim, device=device, dtype=torch.float32)
        ea = edge_attr.float()
        if ea.dim() == 1:
            ea = ea.view(-1, 1)
        if ea.size(-1) == self.edge_attr_dim:
            return ea
        # pad / truncate to expected MPM edge dim (default 7)
        out = ea.new_zeros(ea.size(0), self.edge_attr_dim)
        n = min(self.edge_attr_dim, ea.size(-1))
        out[:, :n] = ea[:, :n]
        return out

    def forward(self, data) -> Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        edge_attr = self._fix_edge_attr(
            getattr(data, "edge_attr", None),
            int(edge_index.size(1)),
            x.device,
        )
        env_edge_attr = self.first_MLP_env_attr(edge_attr)
        act_edge_attr = self.first_MLP_act_attr(edge_attr)
        x = self.first_MLP_node(x)
        for gnn_idx in range(self.num_layers):
            x = self.hidden_layer_norm(x)
            in_logits = self.in_act_net(
                x=x,
                edge_index=edge_index,
                env_edge_attr=env_edge_attr,
                act_edge_attr=act_edge_attr,
            )
            out_logits = self.out_act_net(
                x=x,
                edge_index=edge_index,
                env_edge_attr=env_edge_attr,
                act_edge_attr=act_edge_attr,
            )
            temp = (
                self.temp_model(x=x, edge_index=edge_index, edge_attr=env_edge_attr)
                if self.learn_temp
                else self.temp
            )
            in_probs = F.gumbel_softmax(logits=in_logits, tau=temp, hard=True)
            out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=True)
            edge_weight = self.create_edge_weight(
                edge_index=edge_index,
                keep_in_prob=in_probs[:, 0],
                keep_out_prob=out_probs[:, 0],
            )
            out = self.env_net[0 + gnn_idx](
                x=x,
                edge_index=edge_index,
                edge_weight=edge_weight,
                edge_attr=env_edge_attr,
            )
            out = self.dropout(out)
            out = self.act(out)
            x = x + out if self.skip else out
        x = self.hidden_layer_norm(x)
        x = self.env_net[-1](x)
        out, _ = self.glob(x, batch)
        return out

    def create_edge_weight(
        self, edge_index: Adj, keep_in_prob: Tensor, keep_out_prob: Tensor
    ) -> Tensor:
        u, v = edge_index
        return keep_in_prob[v] * keep_out_prob[u]


# 兼容别名
SotaMPM = CoGNNNet
