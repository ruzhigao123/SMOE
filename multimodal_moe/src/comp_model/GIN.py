import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.pool import global_add_pool
import torch.nn as nn
from multimodal_moe.src.config import FLAGS


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, drop_out):
        super(GIN, self).__init__()

        if isinstance(hidden_channels, int):
            h = hidden_channels
        else:
            h = hidden_channels[0]

        self.drop_out = drop_out
        self.convs = torch.nn.ModuleList()

        for i in range(num_layers):
            # GIN 的核心是一个内部 MLP
            din = in_channels if i == 0 else h
            nn_module = nn.Sequential(
                Linear(din, h),
                nn.ReLU(),
                Linear(h, h)
            )
            self.convs.append(GINConv(nn_module))

        # GIN 通常配合 Sum Pooling 以获得更强的判别力
        self.global_pool = global_add_pool
        self.first_MLP = Linear(h, FLAGS.D)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for idx in range(len(self.convs)):
            x = self.convs[idx](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_out, training=self.training)

        x = self.global_pool(x, batch)
        x = self.first_MLP(x)
        return x