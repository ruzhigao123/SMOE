"""
层级模态 MoE（典型 Hierarchical Modal MoE，端到端回归）。

对齐 AAAI'25 HierarchicalMoE：
  Low-level  : CDFG / Text / AST 可学习专家（对应论文 node/block/graph）
  High-level : 按任务 soft-route 三专家 + CV(importance) 正则
  GNN        : HARP 同款 TransformerConv（非 GCN）
  训练       : 两阶段（warmup 分专家单独训 + 后期轮流 joint/separate）
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_add_pool, global_max_pool

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)  # multimodal_moe
_HMOE_ROOT = os.path.dirname(_ROOT)
_HMOE_SRC = os.path.join(_HMOE_ROOT, "src")
for _p in (_HMOE_SRC, _HMOE_ROOT, _ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.config import FLAGS
from src.utils import MLP

from experts import ModalExperts
from gate import HierarchicalExpertGate

# 可选对照 GNN：HMoE 仓库无 comp_model 时仅用 TransformerConv
try:
    from src.comp_model.GNN_DSE import Net as GNN_DSE_Net
    from src.comp_model.HGP import HierNet
    from src.comp_model.Ironman import GCNNet
    from src.comp_model.pna import PNANet
    from src.comp_model.GCN import GCN
    from src.comp_model.ironman_no_edage import GCNNet_NO_Edage
    from src.comp_model.GraphSAGE import GraphSAGE
    from src.comp_model.GAT import GAT
    from src.comp_model.GIN import GIN
    _HAS_COMP_MODEL = True
except ImportError:
    GNN_DSE_Net = HierNet = GCNNet = PNANet = GCN = None
    GCNNet_NO_Edage = GraphSAGE = GAT = GIN = None
    _HAS_COMP_MODEL = False


class TripleModalFusionBlock(nn.Module):
    """CDFG→Text→AST 串行交叉注意力融合。"""

    def __init__(self, g_dim=128, s_dim=768, a_dim=128, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.proj_g = nn.Linear(g_dim, embed_dim)
        self.proj_s = nn.Linear(s_dim, embed_dim)
        self.proj_a = nn.Linear(a_dim, embed_dim)
        self.attn_gs = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn_gsa = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.gate = nn.Sequential(nn.Linear(embed_dim * 2, 1), nn.Sigmoid())

    def forward(self, h_g, h_s, h_a):
        g_emb = self.proj_g(h_g).unsqueeze(1)
        s_emb = self.proj_s(h_s).unsqueeze(1)
        a_emb = self.proj_a(h_a).unsqueeze(1)

        h_gs, _ = self.attn_gs(query=g_emb, key=s_emb, value=s_emb)
        h_gs = self.norm1(h_gs + g_emb)

        h_fuse, _ = self.attn_gsa(query=h_gs, key=a_emb, value=a_emb)
        h_fuse = self.norm2(h_fuse + h_gs).squeeze(1)

        g_proj = g_emb.squeeze(1)
        gate_val = self.gate(torch.cat([g_proj, h_fuse], dim=-1))
        return gate_val * g_proj + (1 - gate_val) * h_fuse


class TransformerGraphEncoder(nn.Module):
    """
    对齐 HierarchicalMoE / HARP：多层 TransformerConv + (max∥add) 池化。
    edge_attr 可选；无边特征时退化为无 edge_dim 的 TransformerConv。
    """

    def __init__(self, in_channels, out_dim=128, hidden=128, num_layers=6, dropout=0.1, edge_dim=None):
        super().__init__()
        self.dropout = float(dropout)
        self.convs = nn.ModuleList()
        self.edge_dim = edge_dim
        for i in range(int(num_layers)):
            in_c = in_channels if i == 0 else hidden
            if edge_dim is not None and int(edge_dim) > 0:
                self.convs.append(
                    TransformerConv(in_c, hidden, edge_dim=int(edge_dim), dropout=self.dropout)
                )
            else:
                self.convs.append(TransformerConv(in_c, hidden, dropout=self.dropout))
        self.out_proj = nn.Linear(hidden * 2, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)
        for conv in self.convs:
            if self.edge_dim is not None and edge_attr is not None:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat([global_max_pool(x, batch), global_add_pool(x, batch)], dim=-1)
        return self.out_proj(x)


class GNNEncoder(nn.Module):
    """
    CDFG / AST 编码器。
    默认（comparative_if=False）用论文同款 TransformerConv；
    comparative_if=True 时走 FLAGS.comparative_model 对照实验。
    """

    def __init__(self, in_channels, deg=None, out_dim=128):
        super().__init__()
        del deg
        self.out_dim = int(out_dim)
        if not getattr(FLAGS, "comparative_if", False):
            # 论文 / HARP 默认：TransformerConv × num_layers
            edge_dim = None
            if getattr(FLAGS, "encode_edge", False):
                edge_dim = int(getattr(FLAGS, "edge_dim", 0) or 0) or None
            self.gnn_model = TransformerGraphEncoder(
                in_channels=in_channels,
                out_dim=self.out_dim,
                hidden=int(getattr(FLAGS, "hidden_num", 128)),
                num_layers=int(getattr(FLAGS, "num_layers", 6)),
                dropout=float(getattr(FLAGS, "dropout", 0.1)),
                edge_dim=edge_dim,
            )
            return

        if not _HAS_COMP_MODEL:
            raise ImportError(
                "comparative_if=True 需要 src.comp_model（MPM），"
                "HierarchicalMoE 默认请保持 comparative_if=False 使用 TransformerConv"
            )
        model_name = str(getattr(FLAGS, "comparative_model", "gcn")).lower()
        if model_name == "pna":
            self.gnn_model = PNANet(
                in_dim=in_channels,
                deg=None,
                num_layer=getattr(FLAGS, "num_layers", 2),
                emb_dim=getattr(FLAGS, "hidden_num", 128),
            )
        elif model_name in ("gnn-dse", "gin"):
            if model_name == "gin":
                self.gnn_model = GIN(
                    in_channels=in_channels,
                    hidden_channels=FLAGS.D,
                    num_layers=6,
                    drop_out=0.2,
                )
            else:
                self.gnn_model = GNN_DSE_Net(in_channels, FLAGS.hidden_num)
        elif model_name == "hgp":
            self.gnn_model = HierNet(in_channels, FLAGS.hidden_num, num_layers=3, conv_type="sage", drop_out=0.0)
        elif model_name == "ironman":
            self.gnn_model = GCNNet(in_channels=in_channels)
        elif model_name == "ironman_no_edge":
            self.gnn_model = GCNNet_NO_Edage(
                in_channels=in_channels, hidden_channels=FLAGS.D, num_layers=6, drop_out=0
            )
        elif model_name == "graphsage":
            self.gnn_model = GraphSAGE(
                in_channels=in_channels, hidden_channels=FLAGS.D, num_layers=3, drop_out=0.1
            )
        elif model_name == "gat":
            self.gnn_model = GAT(
                in_channels=in_channels, hidden_channels=FLAGS.D, num_layers=6, drop_out=0.2
            )
        else:
            self.gnn_model = GCN(
                in_channels=in_channels,
                hidden_channels=FLAGS.hidden_num,
                num_layers=5,
                drop_out=0.2,
            )

    def forward(self, data):
        return self.gnn_model(data)


class HierarchicalModalMoE(nn.Module):
    """
    多模态 Hierarchical MoE（端到端回归）:

      Low-level  : CDFG / Text / AST 三个可学习模态专家
      High-level : 按任务 soft-route（CV 正则 + 均匀初始化）
      训练       : 两阶段（对齐论文 Eq.6）
    """

    def __init__(
        self,
        cdfg_in_channels=153,
        ast_in_channels=784,
        target_list=None,
        deg=None,
        expert_dim=256,
        gnn_out_dim=128,
        gate_type="modal",
        warmup_epochs=0,
        top_k=2,
        lambda_balance=5e-3,
        pragma_dim=0,
        soft_epochs=20,
        expert_depth=2,
        soft_mix_alpha=0.5,
        allow_single_modal=True,
        lambda_route=0.0,
        route_warmup_epochs=0,
        hierarchical_moe_epoch=None,
        hierarchical_alternate_train=True,
    ):
        super().__init__()
        del gate_type, warmup_epochs, top_k, allow_single_modal, lambda_route, route_warmup_epochs
        self.target_list = list(target_list or FLAGS.target)
        self.expert_dim = expert_dim
        self.gnn_out_dim = gnn_out_dim
        # 论文默认 α=β=5e-3；不再随 epoch 衰减到 0
        self.lambda_balance = float(
            getattr(FLAGS, "moe_lmbda", None)
            if getattr(FLAGS, "moe_lmbda", None) is not None
            else lambda_balance
        )
        self.soft_epochs = int(soft_epochs)
        self.soft_mix_alpha = float(soft_mix_alpha)
        self.pragma_dim = int(pragma_dim)
        # 两阶段：warmup 默认总 epoch 的一半（调用方可覆盖）
        default_epochs = int(getattr(FLAGS, "epoch_num", 300))
        self.hierarchical_moe_epoch = int(
            hierarchical_moe_epoch
            if hierarchical_moe_epoch is not None
            else getattr(FLAGS, "hierarchical_moe_epoch", default_epochs // 2)
        )
        self.hierarchical_alternate_train = bool(
            getattr(
                FLAGS,
                "hierarchical_alternate_train",
                hierarchical_alternate_train,
            )
        )

        self.cdfg_encoder = GNNEncoder(in_channels=cdfg_in_channels, deg=deg, out_dim=gnn_out_dim)
        self.ast_encoder = GNNEncoder(in_channels=ast_in_channels, deg=deg, out_dim=gnn_out_dim)
        self.text_gate_proj = nn.Linear(768, gnn_out_dim)

        self.expert_pool = ModalExperts(
            cdfg_dim=gnn_out_dim,
            text_dim=768,
            ast_dim=gnn_out_dim,
            expert_dim=expert_dim,
            depth=expert_depth,
        )

        gate_in_dim = gnn_out_dim * 3
        n_tasks = len(self.target_list)
        self.gating_net = HierarchicalExpertGate(
            gate_in_dim=gate_in_dim,
            num_tasks=n_tasks,
            num_experts=3,
            soft_epochs=soft_epochs,
        )

        self.fusion_block = TripleModalFusionBlock(
            g_dim=gnn_out_dim,
            s_dim=768,
            a_dim=gnn_out_dim,
            embed_dim=expert_dim,
        )

        self.MLPs = nn.ModuleDict()
        for t in self.target_list:
            self.MLPs[t] = MLP(
                input_dim=expert_dim,
                output_dim=FLAGS.out_dim,
                activation_type="relu",
                num_hidden_lyr=2,
                hidden_channels=[FLAGS.hidden_num, FLAGS.hidden_num],
            )

        self.loss_fn = nn.MSELoss()
        self.register_buffer(
            "task_loss_weight",
            torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0][:n_tasks], dtype=torch.float32),
        )

    def _encode_ast(self, data):
        ast_batch = getattr(data, "ast_graph", None)
        if ast_batch is None:
            raise RuntimeError("data.ast_graph 缺失：inject_modalities 须注入对齐的 AST 图 Batch")
        return self.ast_encoder(ast_batch)

    def _fuse(self, f_g, f_s, f_a, expert_weights, experts_out):
        """expert_weights [B,3], experts_out [B,3,D] → 融合表示。"""
        h_expert = torch.einsum("bi,bid->bd", expert_weights, experts_out)
        f_g_m = f_g * expert_weights[:, 0:1]
        f_s_m = f_s * expert_weights[:, 1:2]
        f_a_m = f_a * expert_weights[:, 2:3]
        h_attn = self.fusion_block(f_g_m, f_s_m, f_a_m)
        a = self.soft_mix_alpha
        return (1.0 - a) * h_attn + a * h_expert

    def _task_mse(self, pred, y, task_i):
        target = y.view(len(y), FLAGS.out_dim).float()
        raw_mse = self.loss_fn(pred, target)
        y_scale = target.detach().pow(2).mean().clamp_min(1e-4)
        w = self.task_loss_weight[task_i] if task_i < self.task_loss_weight.numel() else 1.0
        return w * (raw_mse / y_scale), raw_mse

    def _separate_train(self, epoch) -> bool:
        """论文 Eq.6：warmup 或交替奇数 epoch → 分专家单独监督。"""
        if epoch is None or not self.training:
            return False
        # fine-tune 场景可关两阶段；此处无 finetune flag 时始终启用
        if bool(getattr(FLAGS, "finetune", False)):
            return False
        if epoch < self.hierarchical_moe_epoch:
            return True
        if self.hierarchical_alternate_train and (epoch % 2 == 1):
            return True
        return False

    def forward(self, data, epoch=None):
        cdfg_batch = getattr(data, "cdfg_feat", None)
        if cdfg_batch is None:
            raise RuntimeError("data.cdfg_feat 缺失：请在 collate/train 中注入 CDFG Batch")
        f_g = self.cdfg_encoder(cdfg_batch)
        f_s = data.text_feat
        if f_s is None:
            raise RuntimeError("data.text_feat 缺失")
        if f_s.dim() > 2:
            f_s = f_s.view(f_s.size(0), -1)
        f_a = self._encode_ast(data)

        B = f_g.size(0)
        if f_s.size(0) != B or f_a.size(0) != B:
            raise RuntimeError(
                f"三模态 batch 长度不一致: cdfg={B} text={f_s.size(0)} ast={f_a.size(0)}"
            )

        experts_out = self.expert_pool(f_g, f_s, f_a)  # [B, 3, D]
        gate_feat = torch.cat([f_g, self.text_gate_proj(f_s), f_a], dim=-1)

        out_dict = {}
        loss_dict = {}
        gate_records = {}
        total_loss = torch.zeros((), device=f_g.device)
        aux_loss = torch.zeros((), device=f_g.device)
        separate = self._separate_train(epoch)

        for i, target_name in enumerate(self.target_list):
            y = getattr(data, target_name.replace("-", "_"), None)

            if separate:
                # 论文 stage-1 / alternate：三个专家各自独力预测，loss 取平均
                # （等价关闭高层门控，防止收敛慢的模态专家被饿死）
                sep_loss = torch.zeros((), device=f_g.device)
                preds = []
                for e in range(3):
                    w = torch.zeros(B, 3, device=f_g.device, dtype=experts_out.dtype)
                    w[:, e] = 1.0
                    h_e = self._fuse(f_g, f_s, f_a, w, experts_out)
                    pred_e = self.MLPs[target_name](h_e)
                    preds.append(pred_e)
                    if y is not None:
                        scaled, _ = self._task_mse(pred_e, y, i)
                        sep_loss = sep_loss + scaled
                # 对外仍返回 soft gate 的加权结果（便于监控）；训练梯度来自分专家 loss
                weights, expert_idx, logits = self.gating_net(gate_feat, i, epoch=epoch)
                h_fuse = self._fuse(f_g, f_s, f_a, weights, experts_out)
                pred = self.MLPs[target_name](h_fuse)
                out_dict[target_name] = pred
                gate_records[target_name] = {
                    "weights": weights.detach(),
                    "expert": expert_idx.detach(),
                    "logits": logits.detach(),
                    "mode": "separate",
                }
                if y is not None:
                    sep_loss = sep_loss / 3.0
                    _, raw_mse = self._task_mse(pred.detach(), y, i)  # 仅记录
                    loss_dict[target_name] = raw_mse
                    total_loss = total_loss + sep_loss
                    # 低层：对当前 soft gate 仍加 CV 正则（论文 α LR）
                    aux_loss = aux_loss + HierarchicalExpertGate.compute_balance_loss(weights)
            else:
                weights, expert_idx, logits = self.gating_net(gate_feat, i, epoch=epoch)
                gate_records[target_name] = {
                    "weights": weights.detach(),
                    "expert": expert_idx.detach(),
                    "logits": logits.detach(),
                    "mode": "joint",
                }
                h_fuse = self._fuse(f_g, f_s, f_a, weights, experts_out)
                pred = self.MLPs[target_name](h_fuse)
                out_dict[target_name] = pred
                if y is not None:
                    scaled, raw_mse = self._task_mse(pred, y, i)
                    loss_dict[target_name] = raw_mse
                    total_loss = total_loss + scaled
                    aux_loss = aux_loss + HierarchicalExpertGate.compute_balance_loss(weights)

        aux_loss = aux_loss / max(len(self.target_list), 1)
        total_loss = total_loss + self.lambda_balance * aux_loss
        return out_dict, total_loss, loss_dict, aux_loss, gate_records


ModalMoESystem = HierarchicalModalMoE
