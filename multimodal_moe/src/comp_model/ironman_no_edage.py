# Ironman.py 重构版

import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.pool import global_mean_pool
import torch.nn as nn
from multimodal_moe.src.config import FLAGS


class GCNNet_NO_Edage(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, drop_out):
        """
        取消固定传参后的 Ironman：
        所有超参数必须从外部（如 Net 类或 FLAGS）显式传入。
        """
        super(GCNNet_NO_Edage, self).__init__()

        # 1. 动态处理 hidden_channels 列表
        if isinstance(hidden_channels, int):
            # 如果传入整数，自动扩展为对应层数的列表
            self.hidden_list = [hidden_channels] * num_layers
        else:
            self.hidden_list = hidden_channels

        self.drop_out = drop_out
        self.convs = torch.nn.ModuleList()

        # 2. 构建卷积层
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(in_channels, self.hidden_list[i]))
            else:
                self.convs.append(GCNConv(self.hidden_list[i - 1], self.hidden_list[i]))

        # 3. 映射层对齐到全局维度 D
        self.global_pool = global_mean_pool
        self.first_MLP = Linear(self.hidden_list[-1], FLAGS.D)

    def forward(self, data):
        # 忽略 edge_attr，仅使用拓扑结构
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for idx in range(len(self.convs)):
            x = self.convs[idx](x, edge_index)
            x = F.relu(x)
            # 使用传入的 drop_out 参数
            x = F.dropout(x, p=self.drop_out, training=self.training)

        x = self.global_pool(x, batch)
        x = self.first_MLP(x)

        return x