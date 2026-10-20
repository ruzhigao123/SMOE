# 冻结专家 + MoE 门控入口。
#
# 默认：cv10_outputs 10 折专家 + train_30 上 8:2 + 输入条件门控
#
#   # 源域训门控（默认从已有 best.pth 继续）
#   python -m multimodal_moe.moe_system.moe_cv10_main --subtask train
#   # 论文小样本适应：每个 test_10 核 k=50（kmeans）微调门控
#   python -m multimodal_moe.moe_system.moe_cv10_main --subtask finetune \
#     --transfer_k_shot 50 --coreset kmeans
#   # selector 风格风险门控与监督；始终保留三专家 soft MoE 融合
#   python -m multimodal_moe.moe_system.moe_cv10_main --subtask train --gate_strategy selector
#   # 消融：固定 1/3 权重 / 每指标硬选 1 个专家（其余权重为 0）
#   python -m multimodal_moe.moe_system.moe_cv10_main --subtask train --mix_mode uniform
#   python -m multimodal_moe.moe_system.moe_cv10_main --subtask train --mix_mode hard_select
from __future__ import annotations

import argparse
import sys
from os.path import abspath, dirname, join

import torch

_THIS = dirname(abspath(__file__))
_MM = dirname(_THIS)
_HMOE = dirname(_MM)
for _p in (_HMOE, _MM, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def parse_args():
    p = argparse.ArgumentParser(description="Frozen ablation experts + MoE gate")
    p.add_argument(
        "--gate_strategy",
        default="legacy",
        choices=["legacy", "selector"],
        help="legacy=保持当前 MoE 门控/训练不变；selector=残差风险门控、"
        "NLL/风险校准/排序/分类辅助监督 + 源域跨家族 MLDG（仍加权融合三专家）",
    )
    p.add_argument(
        "--subtask",
        default="finetune",
        choices=["train", "inference", "finetune"],
        help="train=源域训门控；finetune=论文小样本适应（每未见核 k-shot）",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=64,
        help="随机种子；save_dir/cv10_root/ablation_root 追加 seed_{seed}（如 64..68）",
    )
    p.add_argument("--train_ratio", type=float, default=0.8, help="30 核内 train 比例（默认 8:2）")
    p.add_argument(
        "--expert_source",
        default="cv10",
        choices=["ablation", "cv10"],
        help="cv10=train_30 用折权重、test_10 用 ablation；ablation=全程 ablation_outputs",
    )
    p.add_argument(
        "--gate_mode",
        default="free",
        choices=["free", "hard"],
        help="free=全 softmax（推荐）；hard=special≥min_special（仅 soft 混合内的门控结构）",
    )
    p.add_argument(
        "--mix_mode",
        default="hard_select",
        choices=["soft", "uniform", "hard_select"],
        help="专家混合消融："
        "soft=可学习软权重融合（默认 Soft MoE）；"
        "uniform=三专家固定各 1/3；"
        "hard_select=每个指标只选 1 个专家（argmax，其余权重为 0）",
    )
    p.add_argument(
        "--min_special",
        type=float,
        default=0.0,
        help="仅 gate_mode=hard：perf→AST / util→Text 最小权重",
    )
    p.add_argument(
        "--balance_coef",
        type=float,
        default=0.01,
        help="门控负载均衡 CV 正则系数",
    )
    p.add_argument(
        "--perf_loss_weight",
        type=float,
        default=3.0,
        help="perf 相对 util 的 MSE 权重（perf 是主误差源）",
    )
    p.add_argument("--use_pred_aware", action="store_true", default=True)
    p.add_argument("--no_pred_aware", action="store_false", dest="use_pred_aware")
    p.add_argument(
        "--pred_slot_dim",
        type=int,
        default=16,
        help="pred_aware 时每个专家槽位维度（分任务：emb‖3×slot，不再 flatten 全 T）",
    )
    p.add_argument(
        "--use_residual",
        action="store_true",
        default=False,
        help="残差校正默认关（易过拟合，best@epoch1）",
    )
    p.add_argument("--no_residual", action="store_false", dest="use_residual")
    p.add_argument(
        "--patience",
        type=int,
        default=50,
        help="已关闭早停：默认 0=跑满 epoch_num（保留参数兼容旧命令）",
    )
    p.add_argument(
        "--ckpt_select",
        default="beat_first",
        choices=["sum5", "beat_first", "gap"],
        help="best.pth 选模（指标为 MSE）：sum5=最低 val_mse_sum5；"
        "beat_first=优先 MoE 打败最强单专家的 epoch（默认）；"
        "gap=最大化 (专家sum5 - MoE sum5)",
    )
    p.add_argument(
        "--beat_margin",
        type=float,
        default=0.0,
        help="判定 MoE 有效：moe_mse_sum5 + margin < 最强单专家（默认 0）",
    )
    p.add_argument(
        "--prior_ast_on_perf",
        type=float,
        default=0.5,
        help="free 模式下 perf 对 AST 的 soft prior bias",
    )
    p.add_argument(
        "--prior_text_on_util",
        type=float,
        default=0.5,
        help="free 模式下 util 对 Text 的 soft prior bias",
    )
    p.add_argument("--use_prior", action="store_true", default=True)
    p.add_argument("--no_prior", action="store_false", dest="use_prior")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epoch_num", type=int, default=500)
    p.add_argument(
        "--lr",
        type=float,
        # default=1e-3,
        default=5e-4,
        help="对齐 HierarchicalMoE 论文/代码默认 lr",
    )
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--target",
        nargs="+",
        default=["perf", "util-LUT", "util-FF", "util-DSP", "util-BRAM"],
    )
    p.add_argument("--print_every", type=int, default=1)
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="默认开启：有 best.pth 则从其权重继续训（跑满后再开训会再训一轮 epoch_num）",
    )
    p.add_argument(
        "--no_resume",
        action="store_false",
        dest="resume",
        help="忽略已有 best.pth，门控从零初始化（结构变更时才需要）",
    )
    p.add_argument(
        "--dataset_mode",
        type=str,
        default="filtered",
        choices=["filtered", "not_0", "del_0", "keep_zero"],
        help="filtered=two_tower_datasets*；not_0=*_not_0（保留零指标）",
    )
    p.add_argument(
        "--save_dir",
        default=None,
        help="默认随 dataset_mode：moe_cv10_outputs 或 moe_cv10_outputs_not_0",
    )
    p.add_argument(
        "--model_path",
        default=None,
        help="inference 权重：默认 best.pth；也可 best_perf.pth / best_util_LUT.pth 等",
    )
    p.add_argument(
        "--select_target",
        default=None,
        help="inference 时按指标自动选 best_{target}.pth（如 perf / util-LUT）；"
        "与 --model_path 同时给时以 model_path 为准",
    )
    p.add_argument(
        "--cv10_root",
        default=None,
        help="默认随 dataset_mode：cv10_outputs 或 cv10_outputs_not_0",
    )
    p.add_argument(
        "--ablation_root",
        default=None,
        help="默认随 dataset_mode：ablation_outputs 或 ablation_outputs_not_0",
    )
    p.add_argument(
        "--folds_json",
        default=None,
        help="默认 {cv10_root}/folds.json",
    )
    p.add_argument(
        "--kernel_split_json",
        default=None,
        help="默认 {ablation_root}/kernel_split.json",
    )
    p.add_argument(
        "--gate_use_pragma",
        action="store_true",
        default=False,
        help="门控额外拼接 DesignPoint pragma 特征（points_list.pkl 或 .c）",
    )
    p.add_argument("--no_gate_use_pragma", action="store_false", dest="gate_use_pragma")
    p.add_argument(
        "--pragma_proj_dim",
        type=int,
        default=64,
        help="pragma 固定编码投影到门控的维度",
    )
    p.add_argument(
        "--skip_all_zero_config",
        type=bool,
        default=None,
        help="加载后再滤全零；默认 filtered=True、not_0=False",
    )
    p.add_argument(
        "--points_root",
        default=None,
        help="points_list.pkl 根目录（默认随 dataset_mode）",
    )
    p.add_argument(
        "--c_src_root",
        default=None,
        help=".c 源码根目录（points 缺失时回退解析 pragma）",
    )
    # ---- 小样本适应（对齐 HierarchicalMoE: finetune + transfer_k_shot + coreset）----
    p.add_argument(
        "--finetune",
        action="store_true",
        default=False,
        help="在 test_10 未见核上 k-shot 微调门控（专家仍冻结）；"
        "subtask=finetune 时自动打开",
    )
    p.add_argument(
        "--transfer_k_shot",
        type=int,
        default=50,
        help="每个未见 kernel 用于适应的标注条数（论文默认 50）",
    )
    p.add_argument(
        "--coreset",
        default="kmeans",
        choices=["kmeans", "random"],
        help="k-shot 选取：kmeans=门控编码聚类取质心近邻（论文默认）；random=按 seed 随机",
    )
    p.add_argument(
        "--ft_lr",
        type=float,
        default=None,
        help="微调学习率；默认与 --lr 相同",
    )
    p.add_argument(
        "--ft_epoch_num",
        type=int,
        default=None,
        help="微调轮数；默认与 --epoch_num 相同",
    )
    p.add_argument(
        "--ft_save_dir",
        default=None,
        help="微调产物目录；默认 {save_dir}/finetune_k{K}_{coreset}",
    )
    p.add_argument(
        "--pretrained_path",
        default=None,
        help="微调加载的源域预训练门控；默认 --model_path 或 {save_dir}/best.pth",
    )
    p.add_argument(
        "--ft_resume_from",
        default="train_model",
        choices=["train_model", "last", "auto", "none"],
        help="再次微调时从哪份上次 ft 权重继续："
        "train_model=默认 train_model.pth；last=last.pth；"
        "auto=有 train_model 用它否则 last；none=同 --no_resume 从源域重开",
    )
    selector = p.add_argument_group("selector 风格 MoE（仅 gate_strategy=selector 生效）")
    selector.add_argument("--selector_temperature", type=float, default=0.75,
                          help="权重 softmax(-预测误差风险/temperature) 的温度，必须 >0")
    selector.add_argument("--selector_aux_weight", type=float, default=1.0,
                          help="误差风险辅助监督系数；保留原融合 MSE 损失")
    selector.add_argument("--selector_supervised_epochs", type=int, default=20,
                          help="源域前 N 轮监督训练，剩余 epoch_num 轮数做 MLDG；微调始终用监督训练")
    selector.add_argument("--selector_mldg_steps", type=int, default=32,
                          help="每个 MLDG epoch 的跨家族 episode 数")
    selector.add_argument("--selector_inner_lr", type=float, default=2e-5,
                          help="MLDG support 临时更新步长")
    selector.add_argument("--selector_meta_lr", type=float, default=5e-5,
                          help="MLDG 阶段外层优化器学习率")
    selector.add_argument("--selector_points_per_kernel", type=int, default=24,
                          help="每个 episode 中每核抽取的训练配置数")
    selector.add_argument("--selector_support_families", type=int, default=4,
                          help="每个 episode 的 support 家族数")
    selector.add_argument("--selector_meta_families", type=int, default=2,
                          help="每个 episode 的 meta 家族数（与 support 不相交）")
    args = p.parse_args()
    args._explicit_selector_options = {
        flag.split("=", 1)[0].removeprefix("--selector_")
        for flag in sys.argv[1:] if flag.startswith("--selector_")
    }
    if args.gate_strategy == "selector":
        from multimodal_moe.moe_system.selector_moe import selector_config_from_args, validate_selector_config
        try:
            validate_selector_config(selector_config_from_args(args))
            if not args.use_pred_aware:
                raise ValueError("selector 风格 MoE 需要专家预测，不能使用 --no_pred_aware")
        except ValueError as error:
            p.error(str(error))
    return args


def _resolve_mode_paths(args):
    """按 dataset_mode + seed 填充 save_dir / cv10_root / ablation_root 等。"""
    from multimodal_moe.moe_system.utils_new import (
        default_output_dirs,
        ensure_seed_outdir,
        normalize_dataset_mode,
        set_dataset_mode,
    )

    ds = normalize_dataset_mode(getattr(args, "dataset_mode", "filtered"))
    set_dataset_mode(ds)
    args.dataset_mode = ds
    seed = int(getattr(args, "seed", 64))
    outs = default_output_dirs(ds, seed=seed)
    if not getattr(args, "save_dir", None):
        args.save_dir = outs["moe_cv10_root"]
        if getattr(args, "gate_strategy", "legacy") == "selector":
            # Independent checkpoints: switching methods never resumes an old gate.
            args.save_dir = join(args.save_dir, "selector")
        mix_mode = str(getattr(args, "mix_mode", "soft")).lower()
        if mix_mode and mix_mode != "soft":
            # 消融实验分目录，避免覆盖 soft 的 best.pth
            args.save_dir = join(args.save_dir, f"mix_{mix_mode}")
    else:
        args.save_dir = ensure_seed_outdir(args.save_dir, seed)
    if not getattr(args, "cv10_root", None):
        args.cv10_root = outs["cv10_root"]
    else:
        args.cv10_root = ensure_seed_outdir(args.cv10_root, seed)
    if not getattr(args, "ablation_root", None):
        args.ablation_root = outs["ablation_root"]
    else:
        args.ablation_root = ensure_seed_outdir(args.ablation_root, seed)
    if not getattr(args, "folds_json", None):
        args.folds_json = join(args.cv10_root, "folds.json")
    if not getattr(args, "kernel_split_json", None):
        args.kernel_split_json = join(args.ablation_root, "kernel_split.json")
    # 微调目录也落在同 seed 下（若未显式指定）
    if getattr(args, "ft_save_dir", None):
        args.ft_save_dir = ensure_seed_outdir(args.ft_save_dir, seed)
    print(
        f"[moe_cv10] dataset_mode={ds} seed={seed} "
        f"gate_strategy={args.gate_strategy} mix_mode={getattr(args, 'mix_mode', 'soft')}\n"
        f"  save_dir={args.save_dir}\n"
        f"  cv10_root={args.cv10_root}\n"
        f"  ablation_root={args.ablation_root}",
        flush=True,
    )
    return args


def main():
    args = parse_args()
    # 避免误触发其他模块 FLAGS 解析
    sys.argv = [sys.argv[0]]
    args = _resolve_mode_paths(args)
    from multimodal_moe.moe_system.moe_cv10_train import inference_moe_cv10, train_moe_cv10

    if args.subtask == "finetune":
        args.finetune = True
    if args.subtask in ("train", "finetune"):
        train_moe_cv10(args)
        return
    if args.subtask == "inference":
        inference_moe_cv10(args)
        return
    raise ValueError(args.subtask)


if __name__ == "__main__":
    main()
