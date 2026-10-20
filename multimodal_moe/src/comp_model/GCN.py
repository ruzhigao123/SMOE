# # src/comp_model/GCN.py

import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.pool import global_add_pool, global_max_pool
from torch_geometric.nn.dense import Linear
from multimodal_moe.src.config import FLAGS


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, drop_out):     #
        super(GCN, self).__init__()

        self.drop_out = drop_out
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()

        # 构建多层 GCN
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(in_channels, hidden_channels))
            else:
                self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # 这里的 D 对应你 config 里的全局隐藏维度
        self.D = FLAGS.D
        # 聚合后（max + add）维度会翻倍，所以是 hidden_channels * 2
        self.first_MLP = Linear(hidden_channels * 2, self.D)

    def forward(self, data):
        # 注意：这里我们只解构 x, edge_index, batch，完全忽略 edge_attr
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. 消息传递与卷积
        for step in range(self.num_layers):
            x = self.convs[step](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_out, training=self.training)

        # 2. 全局池化 (参考 HGP 的池化混合策略)
        # 将全局最大池化和全局加和池化拼接，增强图表示能力
        x_max = global_max_pool(x, batch)
        x_add = global_add_pool(x, batch)
        x = torch.cat([x_max, x_add], dim=1)

        # 3. 映射到指定的 D 维度
        x = self.first_MLP(x)

        return x

# src/comp_model/GCN.py
#--------------------------------------------------------------------------------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn.conv import GCNConv
# from torch_geometric.nn.pool import global_add_pool, global_max_pool
# from torch_geometric.nn.dense import Linear
# from multimodal_moe.src.config import FLAGS
#
#
# class GCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_layers, drop_out):
#         super(GCN, self).__init__()
#
#         self.drop_out = drop_out
#         self.num_layers = num_layers
#
#         # --- 新增：特征投影层 ---
#         # 假设 in_channels 为 778 (768 语义 + 10 硬件)
#         self.sem_dim = 768
#         self.hw_dim = 16
#         self.proj_dim = 128  # 将两类特征映射到对等维度，防止信息淹没
#
#         self.sem_proj = nn.Linear(self.sem_dim, self.proj_dim)
#         self.hw_proj = nn.Linear(self.hw_dim, self.proj_dim)
#
#         # 拼接后的初始维度变为 proj_dim * 2 = 128
#         self.input_dim = self.proj_dim * 2
#
#         self.convs = torch.nn.ModuleList()
#
#         # 构建多层 GCN
#         for i in range(num_layers):
#             if i == 0:
#                 # 第一层接收投影融合后的 128 维特征
#                 self.convs.append(GCNConv(self.input_dim, hidden_channels))
#             else:
#                 self.convs.append(GCNConv(hidden_channels, hidden_channels))
#
#         self.D = FLAGS.D
#         # 聚合后（max + add）映射到全局维度 D
#         self.first_MLP = Linear(hidden_channels * 2, self.D)
#
#     def forward(self, data):
#         # 1. 解构数据
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#
#         # 2. 分离特征并投影
#         # x 的前 768 列是语义嵌入，后 10 列是手动定义的硬件特征
#         x_sem = x[:, :self.sem_dim]
#         x_hw = x[:, self.sem_dim:]
#
#         # 投影映射并激活
#         x_sem = F.relu(self.sem_proj(x_sem))
#         x_hw = F.relu(self.hw_proj(x_hw))
#
#         # 拼接融合：此时硬件信息与语义信息在 128 维空间中权重均等
#         x = torch.cat([x_sem, x_hw], dim=-1)
#
#         # 3. 消息传递与卷积
#         for step in range(self.num_layers):
#             x = self.convs[step](x, edge_index)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.drop_out, training=self.training)
#
#         # 4. 全局池化 (max + add)
#         x_max = global_max_pool(x, batch)
#         x_add = global_add_pool(x, batch)
#         x = torch.cat([x_max, x_add], dim=1)
#
#         # 5. 映射到指定的 D 维度
#         x = self.first_MLP(x)
#
#         return x

#--------------------------------------------------------------------------------------------
# src/comp_model/GCN.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn.conv import GCNConv
# from torch_geometric.nn.pool import global_add_pool, global_max_pool
# from torch_geometric.nn.dense import Linear
# from multimodal_moe.src.config import FLAGS
#
#
# class GCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_layers, drop_out):
#         super(GCN, self).__init__()
#
#         self.drop_out = drop_out
#         self.num_layers = num_layers
#
#         # --- 维度定义 ---
#         self.sem_dim = 768
#         self.hw_dim = 10
#         # 总输入维度保持 778
#
#         # --- 新增：可学习的硬件特征缩放因子 ---
#         # 初始化为全 10.0 的向量，形状为 (1, 10)
#         # 初始值设大（如 10.0）可以确保在训练初期硬件特征具有极高的存在感
#         self.hw_scale = nn.Parameter(torch.ones(1, self.hw_dim) * 10.0)
#
#         self.convs = torch.nn.ModuleList()
#
#         # 构建多层 GCN
#         for i in range(num_layers):
#             if i == 0:
#                 # 此时输入维度依然是原始的 in_channels (778)
#                 self.convs.append(GCNConv(in_channels, hidden_channels))
#             else:
#                 self.convs.append(GCNConv(hidden_channels, hidden_channels))
#
#         self.D = FLAGS.D
#         self.first_MLP = Linear(hidden_channels * 2, self.D)
#
#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#
#         # 1. 分离特征
#         x_sem = x[:, :self.sem_dim]
#         x_hw = x[:, self.sem_dim:]
#
#         # 2. 应用可学习缩放
#         # x_hw 原本数值较小（归一化后在 0-1 之间），乘以 hw_scale 后增强信号强度
#         x_hw = x_hw * self.hw_scale
#
#         # 3. 重新拼接，保持维度为 778
#         x = torch.cat([x_sem, x_hw], dim=-1)
#
#         # 4. 卷积与池化流程
#         for step in range(self.num_layers):
#             x = self.convs[step](x, edge_index)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.drop_out, training=self.training)
#
#         x_max = global_max_pool(x, batch)
#         x_add = global_add_pool(x, batch)
#         x = torch.cat([x_max, x_add], dim=1)
#
#         return self.first_MLP(x)


