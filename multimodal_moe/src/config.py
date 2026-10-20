"""
消融实验配置（参考 MPM src/config.py，路径锚定 multimodal_moe）。

7 种方案（对齐论文 MFF）:
  单模态: cdfg_only, LM, ast_only
  双/三模态: TripleModalFusion + 缺失模态置零 — cdfg_text, ast_cdfg, ast_text, all
"""
import argparse
from os.path import join

import torch

from .get_root_path import get_root_path, get_user, get_host
from .kernel_protocol import ALL_KERNELS_42, MACHSUITE_KERNEL, POLY_KERNEL

TARGETS = [
    "perf",
    "quality",
    "util-BRAM",
    "util-DSP",
    "util-LUT",
    "util-FF",
    "total-BRAM",
    "total-DSP",
    "total-LUT",
    "total-FF",
]

ABLATION_MODES = [
    "all",
    "ast_cdfg",
    "ast_text",
    "cdfg_text",
    "cdfg_only",
    "ast_only",
    "LM",
]

parser = argparse.ArgumentParser(description="multimodal_moe ablation (MPM-style fusion for multi-modal)")
parser.add_argument("--dataset", default="programl")
parser.add_argument("--benchmarks", default=["machsuite", "poly"])
parser.add_argument("--tag", default="ablation-40kernel")
parser.add_argument("--task", default="regression")
parser.add_argument(
    "--subtask",
    default="inference",
    choices=["train", "inference", "split_only", "run_all", "finetune"],
)
parser.add_argument("--ablation_mode", type=str, default="ast_cdfg", choices=ABLATION_MODES)
parser.add_argument(
    "--seed",
    type=int,
    default=66,
    help="随机种子；产物目录追加 seed_{seed}（如 64..68）",
)
parser.add_argument(
    "--split_by",
    default="config",
    choices=["kernel", "config"],
    help="默认 kernel=30/10 再 24/6（最初方案）；config=30 核内配置随机 8:2",
)
parser.add_argument(
    "--train_ratio",
    type=float,
    default=0.8,
    help="仅 split_by=config：train_30 配置池内 train 占比",
)
parser.add_argument(
    "--val_ratio_kernels",
    type=float,
    default=0.2,
    help="kernel 协议里 30 核内 val 比例（由 kernel_protocol 使用）",
)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epoch_num", type=int, default=500)
parser.add_argument("--lr", type=float, default=1e-3, help="旧 MPM 消融默认")
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--D", type=int, default=128, help="modal / GNN out dim（旧 MPM=128）")
parser.add_argument("--hidden_num", type=int, default=128, help="GNN hidden（旧 MPM=128）")
parser.add_argument("--out_dim", type=int, default=1)
parser.add_argument("--gnn_layers", type=int, default=10, help="旧 MPM GCNConv×5")
parser.add_argument("--dropout", type=float, default=0.1, help="旧 MPM dropout=0.2")
parser.add_argument(
    "--gnn_type",
    type=str,
    default="gcn",
    choices=["transformer", "gcn"],
    help="gcn=旧 MPM（默认）；transformer=HierarchicalMoE",
)
parser.add_argument(
    "--gnn_activation",
    type=str,
    default="relu",
    choices=["relu", "elu"],
    help="GNN 激活：旧 MPM=relu；HMoE=elu",
)
# 对比实验（SOTA 骨干）：comparative_if=False 走默认 gnn_type；True 时切换 comparative_model
parser.add_argument(
    "--comparative_if",
    default=False,
    type=lambda v: str(v).lower() in ("1", "true", "yes", "y"),
    help="False=正常 GNN（gnn_type）；True=对比实验，改用 comparative_model",
)
parser.add_argument(
    "--comparative_model",
    type=str,
    default="sotampm",
    choices=[
        "ironman",
        "ironman_no_edge",
        "hgp",
        "pna",
        "gnn-dse",
        "gcn",
        "gin",
        "graphsage",
        "gat",
        "sotampm",
        "mpm",
        "cognn",
    ],
    help='comparative_if=True 时的 SOTA 骨干；"ironman"=IronMan-Pro；"sotampm"/"mpm"/"cognn"=MPM CoGNN',
)
# CoGNN / SOTA-MPM（与 MPM-Innovation config 默认对齐；仅 comparative_model=sotampm 时使用）
parser.add_argument("--learn_temp", default=False, type=lambda v: str(v).lower() in ("1", "true", "yes", "y"))
parser.add_argument("--temp_model_type", default="LIN", type=str)
parser.add_argument("--tau0", default=0.5, type=float)
parser.add_argument("--temp", default=0.01, type=float)
parser.add_argument("--env_model_type", default="SUM_GNN", type=str)
parser.add_argument("--env_num_layers", default=3, type=int)
parser.add_argument("--env_dim", default=128, type=int)
parser.add_argument("--skip", default=False, type=lambda v: str(v).lower() in ("1", "true", "yes", "y"))
parser.add_argument("--batch_norm", default=False, type=lambda v: str(v).lower() in ("1", "true", "yes", "y"))
parser.add_argument("--layer_norm", default=False, type=lambda v: str(v).lower() in ("1", "true", "yes", "y"))
parser.add_argument("--dec_num_layers", default=1, type=int)
parser.add_argument("--act_model_type", default="MEAN_GNN", type=str)
parser.add_argument("--act_num_layers", default=2, type=int)
parser.add_argument("--act_dim", default=16, type=int)
parser.add_argument(
    "--cognn_edge_dim",
    default=7,
    type=int,
    help="CoGNN 边特征维（MPM 默认 7）；消融 CDFG 维不一致时自动 pad/截断",
)
parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument(
    "--target",
    default=["perf", "util-LUT", "util-FF", "util-DSP", "util-BRAM"],
)
parser.add_argument(
    "--dataset_mode",
    type=str,
    default="filtered",
    choices=["filtered", "not_0", "del_0", "keep_zero"],
    help="filtered/del_0=two_tower_datasets*；not_0/keep_zero=*_not_0（保留零指标）",
)
parser.add_argument(
    "--skip_all_zero_config",
    type=bool,
    default=None,
    help="加载后再滤全零标签；默认 filtered=True、not_0=False",
)
parser.add_argument("--print_every", type=int, default=1, help="print train/val every N epochs")
parser.add_argument(
    "--patience",
    type=int,
    default=50,
    help="val 不提升早停；0=不早停（最初方案）",
)
parser.add_argument(
    "--resume",
    action="store_true",
    default=True,
    help="若存在 best.pt 则从中续训（默认开）",
)
parser.add_argument(
    "--no_resume",
    action="store_false",
    dest="resume",
    help="忽略已有 best.pt，从头训练",
)
parser.add_argument(
    "--save_dir",
    default=None,
    help="默认随 dataset_mode：ablation_outputs 或 ablation_outputs_not_0",
)
parser.add_argument("--model_path", default=None, help="inference/finetune 用 checkpoint .pt")
# 小样本适应（对齐 HMoE / Soft MoE：test_10 上每核 k-shot）
parser.add_argument(
    "--transfer_k_shot",
    type=int,
    default=50,
    help="finetune：每个未见核取 k 条适应",
)
parser.add_argument(
    "--coreset",
    type=str,
    default="kmeans",
    choices=["kmeans", "random"],
    help="finetune：k-shot 选取方式",
)
parser.add_argument(
    "--ft_epoch_num",
    type=int,
    default=None,
    help="finetune 轮数；默认与 --epoch_num 相同",
)
parser.add_argument(
    "--ft_lr",
    type=float,
    default=None,
    help="finetune 学习率；默认与 --lr 相同",
)
parser.add_argument(
    "--pretrained_path",
    default=None,
    help="finetune 源域权重；默认 {save_dir}/{ablation_mode}/best.pt",
)
parser.add_argument("--user", default=get_user())
parser.add_argument("--hostname", default=get_host())

FLAGS = parser.parse_args()

# 按 dataset_mode 解析默认产物目录（filtered / not_0 互不覆盖）
try:
    from multimodal_moe.moe_system.utils_new import (
        normalize_dataset_mode,
        set_dataset_mode,
    )

    _ds = normalize_dataset_mode(getattr(FLAGS, "dataset_mode", "filtered"))
    set_dataset_mode(_ds)
    FLAGS.dataset_mode = _ds
    _seed = int(getattr(FLAGS, "seed", 64))
    if not getattr(FLAGS, "save_dir", None):
        FLAGS.save_dir = default_output_dirs(_ds, seed=_seed)["ablation_root"]
    else:
        FLAGS.save_dir = ensure_seed_outdir(FLAGS.save_dir, _seed)
except Exception:
    if not getattr(FLAGS, "save_dir", None):
        _m = str(getattr(FLAGS, "dataset_mode", "filtered")).lower()
        _suf = "_not_0" if _m in ("not_0", "not0", "keep_zero", "keep0", "all", "unfiltered") else ""
        FLAGS.save_dir = join(get_root_path(), f"ablation_outputs{_suf}")
    _seed = int(getattr(FLAGS, "seed", 64))
    FLAGS.save_dir = join(FLAGS.save_dir, f"seed_{_seed}")

# 对比实验产物与默认消融隔离，避免互相覆盖
if bool(getattr(FLAGS, "comparative_if", False)):
    _cm = str(getattr(FLAGS, "comparative_model", "ironman")).lower().replace("-", "_")
    FLAGS.save_dir = join(FLAGS.save_dir, f"comp_{_cm}")

# 方便外部引用
FLAGS.MACHSUITE_KERNEL = MACHSUITE_KERNEL
FLAGS.poly_KERNEL = POLY_KERNEL
FLAGS.ALL_KERNELS_42 = ALL_KERNELS_42
FLAGS.ABLATION_MODES = ABLATION_MODES
