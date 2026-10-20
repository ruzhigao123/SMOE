import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.pool import global_mean_pool
import torch.nn as nn
from multimodal_moe.src.config import FLAGS


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, drop_out, heads=4):
        super(GAT, self).__init__()

        if isinstance(hidden_channels, int):
            h = hidden_channels
        else:
            h = hidden_channels[0]

        self.drop_out = drop_out
        self.convs = torch.nn.ModuleList()
        self.heads = heads

        for i in range(num_layers):
            if i == 0:
                # concat=True 会使输出维度变为 h * heads
                self.convs.append(GATConv(in_channels, h, heads=self.heads, concat=True))
            elif i == num_layers - 1:
                # 最后一层通常设为 concat=False 以保持维度
                self.convs.append(GATConv(h * self.heads, h, heads=1, concat=False))
            else:
                self.convs.append(GATConv(h * self.heads, h, heads=self.heads, concat=True))

        self.global_pool = global_mean_pool
        self.first_MLP = Linear(h, FLAGS.D)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for idx in range(len(self.convs)):
            x = self.convs[idx](x, edge_index)
            x = F.elu(x)  # GAT 常用 ELU 激活
            x = F.dropout(x, p=self.drop_out, training=self.training)

        x = self.global_pool(x, batch)
        x = self.first_MLP(x)
        return x