import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.pool import global_mean_pool
import torch.nn as nn
from multimodal_moe.src.config import FLAGS


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, drop_out):
        super(GraphSAGE, self).__init__()

        if isinstance(hidden_channels, int):
            self.hidden_list = [hidden_channels] * num_layers
        else:
            self.hidden_list = hidden_channels

        self.drop_out = drop_out
        self.convs = torch.nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                # SAGEConv 默认不使用 edge_attr
                self.convs.append(SAGEConv(in_channels, self.hidden_list[i]))
            else:
                self.convs.append(SAGEConv(self.hidden_list[i - 1], self.hidden_list[i]))

        self.global_pool = global_mean_pool
        # 映射到全局维度 D
        self.first_MLP = Linear(self.hidden_list[-1], FLAGS.D)

    def forward(self, data):
        # 仅使用 x 和 edge_index
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for idx in range(len(self.convs)):
            x = self.convs[idx](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_out, training=self.training)

        x = self.global_pool(x, batch)
        x = self.first_MLP(x)
        return x