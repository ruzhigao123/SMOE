"""精简工具（参考 MPM src/utils.py）。"""
from __future__ import annotations

from os import makedirs
from os.path import exists

import torch
import torch.nn as nn


def create_dir_if_not_exists(dir_path: str):
    if not exists(dir_path):
        makedirs(dir_path)


def _get_y_with_target(data, target: str):
    return getattr(data, target.replace("-", "_"), None)


def create_act(act: str):
    if act in ("relu", "ReLU"):
        return nn.ReLU()
    if act in ("identity", "None"):
        return nn.Identity()
    if act in ("elu",):
        return nn.ELU()
    raise ValueError(f"Unknown activation {act}")


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        activation_type="relu",
        num_hidden_lyr=2,
        hidden_channels=None,
        bn=False,
    ):
        super().__init__()
        if not hidden_channels:
            hidden_channels = [input_dim for _ in range(num_hidden_lyr)]
        elif len(hidden_channels) != num_hidden_lyr:
            raise ValueError("hidden_channels length must equal num_hidden_lyr")
        dims = [input_dim] + list(hidden_channels) + [output_dim]
        self.activation = create_act(activation_type)
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        for m in self.layers:
            torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
        self.bn = torch.nn.BatchNorm1d(output_dim) if bn else None

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        if self.bn is not None:
            h = self.bn(h)
        return h
