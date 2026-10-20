"""
用 multimodal_moe 的 train_30/test_10 + seed，启动原版 HierarchicalMoE（src/main 训练路径）。

示例（在 HierarchicalMoE-master 下）：
  # 1) 确保 / 生成与 Soft MoE 相同的划分
  python -m multimodal_moe.hmoe_repro.run_hmoe --stage split_only --seed 67

  # 2) 源域预训练（核名单 = train_30）
  python -m multimodal_moe.hmoe_repro.run_hmoe --stage pretrain --seed 67 --force_regen

  # 3) 未见核微调（核名单 = test_10；需提供预训练 val_model）
  python -m multimodal_moe.hmoe_repro.run_hmoe --stage finetune --seed 67 \\
    --model_path /path/to/xxx_val_model_state_dict.pth --force_regen
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from os.path import abspath, dirname, join
from typing import List, Optional

_THIS = dirname(abspath(__file__))
_MM = dirname(_THIS)
_HMOE = dirname(_MM)
_SRC = join(_HMOE, "src")

for p in (_HMOE, _MM, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)


def _parse_ours():
    p = argparse.ArgumentParser(
        description="Reproduce HierarchicalMoE with multimodal_moe protocol split + seed"
    )
    p.add_argument(
        "--stage",
        choices=["split_only", "pretrain", "finetune"],
        required=True,
        help="split_only=只写 kernel_split；pretrain=train_30；finetune=test_10",
    )
    p.add_argument("--seed", type=int, default=64, help="与 Soft MoE 对齐，如 64..68")
    p.add_argument(
        "--dataset_mode",
        default="filtered",
        choices=["filtered", "not_0", "del_0", "keep_zero"],
        help="仅影响协议 CDFG 计数路径与默认 split 复用目录",
    )
    p.add_argument(
        "--split_json",
        default=None,
        help="默认：ablation_outputs*/seed_N/kernel_split.json（若存在）"
        " 否则 hmoe_repro_outputs/seed_N/kernel_split.json",
    )
    p.add_argument("--rebuild_split", action="store_true")
    p.add_argument(
        "--model_path",
        default=None,
        help="finetune 必填：预训练 val_model_state_dict.pth",
    )
    p.add_argument(
        "--force_regen",
        action="store_true",
        help="强制按当前核名单重建 HMoE programl 数据集（换 seed/名单时建议开）",
    )
    p.add_argument(
        "--no_cleanup_save",
        action="store_true",
        help="默认会在训后清理 save/ 缓存；加此开关则保留 data_*.pt",
    )
    p.add_argument("--transfer_k_shot", type=int, default=50)
    p.add_argument(
        "--coreset",
        default="kmeans",
        choices=["kmeans", "random"],
    )
    p.add_argument(
        "--moe_layers",
        nargs="+",
        default=["hierarchy-weighted-hidden"],
        help="默认论文最佳 hierarchical MoE",
    )
    p.add_argument("--subtask", default="train", help="传给原版 HMoE（默认 train）")
    # 透传给原版 config 的常用开关（写在我们的 parser 里，再塞回 sys.argv）
    p.add_argument("--epoch_num", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default=None)
    return p.parse_known_args()


def _partition_kernels(names: List[str]):
    from multimodal_moe.hmoe_repro.protocol_io import split_machsuite_poly

    return split_machsuite_poly(list(names))


def _rm_tree(path: str) -> None:
    import shutil

    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
        print(f"[hmoe_repro] removed dir {path}", flush=True)


def _cleanup_save_after_stage(
    *,
    stage: str,
    save_dir: str,
    meta_dir: str,
    seed: int,
    get_save_path,
    FLAGS,
    target_names,
) -> None:
    """
    预训练后：删 data_*.pt，并把 encoders 备份到 meta（微调优先读备份）。
    微调后：删掉本 seed 的 finetune_dataset + pretrain_dataset。
    权重在 src/logs/，不碰。
    """
    import glob as _glob
    import shutil

    if not save_dir or not os.path.isdir(save_dir):
        return

    if stage == "pretrain":
        enc_klepto = join(save_dir, "encoders.klepto")
        if os.path.isfile(enc_klepto):
            shutil.copy2(enc_klepto, join(meta_dir, "pretrain_encoders.klepto"))
            print(
                f"[hmoe_repro] backed up encoder → {join(meta_dir, 'pretrain_encoders.klepto')}",
                flush=True,
            )
        for name in ("pragma_dim.klepto", "pragma_dim"):
            src = join(save_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, join(meta_dir, "pretrain_" + os.path.basename(src)))
        n = 0
        for fp in _glob.glob(join(save_dir, "data_*.pt")):
            try:
                os.remove(fp)
                n += 1
            except OSError:
                pass
        for sub in ("raw", "processed"):
            _rm_tree(join(save_dir, sub))
        print(
            f"[hmoe_repro] pretrain cleanup: removed {n} data_*.pt under {save_dir}",
            flush=True,
        )
        return

    if stage == "finetune":
        _rm_tree(save_dir)
        from multimodal_moe.hmoe_repro.protocol_io import seed_tag

        pt_tag = f"mm-proto-{seed_tag(seed)}-pt"
        pt_save = join(
            get_save_path(),
            FLAGS.dataset,
            f"{FLAGS.v_db}_MLP-{FLAGS.pragma_as_MLP}-round{FLAGS.round_num}-"
            f"30kernel-icmp-feb-db-{FLAGS.graph_type}-{FLAGS.task}_"
            f"edge-position-{FLAGS.encode_edge_position}_norm_with-invalid_{FLAGS.invalid}-"
            f"normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_"
            f"tag_{pt_tag}_{''.join(target_names)}",
            "pretrain_dataset",
        )
        _rm_tree(pt_save)
        print(
            f"[hmoe_repro] finetune cleanup: removed ft+pt save caches for seed={seed}",
            flush=True,
        )


def main():
    ours, unknown = _parse_ours()
    from multimodal_moe.hmoe_repro.protocol_io import (
        load_or_build_protocol,
        project_roots,
        seed_tag,
    )
    from multimodal_moe.moe_system.utils_new import normalize_dataset_mode

    ds = normalize_dataset_mode(ours.dataset_mode)
    proto, split_path = load_or_build_protocol(
        seed=ours.seed,
        dataset_mode=ds,
        split_json=ours.split_json,
        rebuild=ours.rebuild_split,
    )
    meta_dir = join(project_roots()["out_root"], seed_tag(ours.seed))
    os.makedirs(meta_dir, exist_ok=True)
    with open(join(meta_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "stage": ours.stage,
                "seed": ours.seed,
                "dataset_mode": ds,
                "split_json": split_path,
                "train_30": proto["train_30"],
                "test_10": proto["test_10"],
                "dropped_2": proto.get("dropped_2"),
            },
            f,
            indent=2,
        )

    print(
        f"[hmoe_repro] seed={ours.seed} stage={ours.stage}\n"
        f"  split_json={split_path}\n"
        f"  train_30={len(proto['train_30'])} test_10={len(proto['test_10'])}\n"
        f"  dropped_2={proto.get('dropped_2')}",
        flush=True,
    )
    if ours.stage == "split_only":
        print("[hmoe_repro] split_only done.", flush=True)
        return

    if ours.stage == "pretrain":
        kernels = list(proto["train_30"])
        finetune = False
        model_path = None
    else:
        kernels = list(proto["test_10"])
        finetune = True
        model_path = ours.model_path
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(
                "finetune 需要有效 --model_path（预训练 val_model_state_dict.pth）\n"
                f"  got: {model_path}"
            )

    ms, poly = _partition_kernels(kernels)
    print(
        f"[hmoe_repro] kernels for this stage: total={len(kernels)} "
        f"machsuite={len(ms)} poly={len(poly)}\n"
        f"  machsuite={ms}\n  poly={poly}",
        flush=True,
    )

    # 原版 config 会 parse sys.argv：只保留未知参数 + 我们注入的 HMoE 参数
    hmoe_argv = [sys.argv[0], "--subtask", ours.subtask, "--random_seed", str(ours.seed)]
    if ours.force_regen:
        # 原版 force_regen 是 type=bool 的脆弱解析；用环境变量在 import 后强制覆盖
        pass
    if ours.epoch_num is not None:
        hmoe_argv += ["--epoch_num", str(ours.epoch_num)]
    if ours.batch_size is not None:
        hmoe_argv += ["--batch_size", str(ours.batch_size)]
    if ours.lr is not None:
        hmoe_argv += ["--lr", str(ours.lr)]
    if ours.moe_layers:
        hmoe_argv += ["--moe_layers", *ours.moe_layers]
    if ours.transfer_k_shot is not None:
        hmoe_argv += ["--transfer_k_shot", str(ours.transfer_k_shot)]
    if ours.coreset:
        hmoe_argv += ["--coreset", ours.coreset]
    if model_path:
        hmoe_argv += ["--model_path", model_path]
    # 追加用户透传的未知参数
    hmoe_argv += list(unknown)
    sys.argv = hmoe_argv

    # 原版 HMoE 约定 cwd=src/（data.py 里 ../dse_database 才指向仓库根）
    os.chdir(_SRC)
    print(f"[hmoe_repro] cwd → {_SRC}", flush=True)

    # 必须先 import config，再改核名单，再 import data（data 在 import 时拷贝名单）
    import config as hmoe_cfg  # noqa: E402

    hmoe_cfg.MACHSUITE_KERNEL = list(ms)
    hmoe_cfg.poly_KERNEL = list(poly)
    hmoe_cfg.simple_KERNEL = []
    hmoe_cfg.transfer_learning = bool(finetune)

    FLAGS = hmoe_cfg.FLAGS
    FLAGS.random_seed = int(ours.seed)
    FLAGS.finetune = bool(finetune)
    FLAGS.transfer_k_shot = int(ours.transfer_k_shot)
    FLAGS.coreset = ours.coreset
    FLAGS.moe_layers = list(ours.moe_layers)
    FLAGS.tag = f"mm-proto-{seed_tag(ours.seed)}-{'ft' if finetune else 'pt'}"
    # 必须保持空串：非空会走 contest/raw_graph 分支并 assert
    FLAGS.data_loading_path = ""
    if ours.force_regen:
        FLAGS.force_regen = True
    if model_path:
        FLAGS.model_path = model_path
    if ours.device:
        # 部分环境用 CUDA_VISIBLE_DEVICES；原版未必有 --device
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", ours.device.replace("cuda:", ""))

    # 对齐原版：预训练 1000 / 微调 500（config 在 import 时 transfer_learning=False，默认会变成 1000）
    if ours.epoch_num is not None:
        FLAGS.epoch_num = int(ours.epoch_num)
    elif finetune:
        FLAGS.epoch_num = 500
        FLAGS.hierarchical_moe_epoch = 0
    else:
        FLAGS.epoch_num = 1000
        FLAGS.hierarchical_moe_epoch = FLAGS.epoch_num // 2

    # encoder：预训练现场 fit（避开仓库 0.24 klepto）；微调必须复用预训练 encoder，
    # 否则 node/edge 维会变（如 127→92），无法 load 预训练权重。
    if not finetune:
        FLAGS.encoder_path = None
    else:
        # 下面在定好 get_save_path 后再解析具体路径；这里先占位
        FLAGS.encoder_path = None

    print(
        f"[hmoe_repro] HMoE FLAGS: finetune={FLAGS.finetune} seed={FLAGS.random_seed} "
        f"tag={FLAGS.tag} force_regen={FLAGS.force_regen} "
        f"moe_layers={FLAGS.moe_layers} model_path={FLAGS.model_path}",
        flush=True,
    )

    # 旧版 data.py：自定义 ALL_KERNEL 会 raise NotImplementedError。
    # import 时先塞入其硬编码的 36 核名单让 SAVE_DIR 分支命中，import 后再换回协议名单。
    _DECOY_MS = [
        "aes", "gemm-blocked", "gemm-ncubed", "spmv-crs", "spmv-ellpack",
        "stencil", "nw", "md", "stencil-3d",
    ]
    _DECOY_POLY = [
        "2mm", "3mm", "adi", "atax", "bicg", "bicg-large", "covariance",
        "doitgen", "doitgen-red", "fdtd-2d", "gemm-p-large", "gemver", "gesummv",
        "heat-3d", "jacobi-1d", "mvt", "seidel-2d", "symm", "symm-opt", "syrk",
        "trmm", "mvt-medium", "correlation", "atax-medium", "bicg-medium",
        "gesummv-medium", "symm-opt-medium",
    ]
    hmoe_cfg.MACHSUITE_KERNEL = list(_DECOY_MS)
    hmoe_cfg.poly_KERNEL = list(_DECOY_POLY)

    from utils import create_dir_if_not_exists, get_root_path, get_save_path, load  # noqa: E402
    import data as hmoe_data  # noqa: E402

    # 换回真实协议核名单，并校准 SAVE_DIR（隔离 tag + 核数）
    hmoe_cfg.MACHSUITE_KERNEL = list(ms)
    hmoe_cfg.poly_KERNEL = list(poly)
    hmoe_data.MACHSUITE_KERNEL = list(ms)
    hmoe_data.poly_KERNEL = list(poly)
    hmoe_data.ALL_KERNEL = list(ms) + list(poly)
    _target = ["perf", "util-DSP", "util-BRAM", "util-LUT", "util-FF"]
    _subdir = "finetune_dataset" if finetune else "pretrain_dataset"
    hmoe_data.SAVE_DIR = join(
        get_save_path(),
        FLAGS.dataset,
        f"{FLAGS.v_db}_MLP-{FLAGS.pragma_as_MLP}-round{FLAGS.round_num}-"
        f"{len(hmoe_data.ALL_KERNEL)}kernel-icmp-feb-db-{FLAGS.graph_type}-{FLAGS.task}_"
        f"edge-position-{FLAGS.encode_edge_position}_norm_with-invalid_{FLAGS.invalid}-"
        f"normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_"
        f"tag_{FLAGS.tag}_{''.join(_target)}",
        _subdir,
    )
    hmoe_data.ENCODER_PATH = join(hmoe_data.SAVE_DIR, "encoders")
    create_dir_if_not_exists(hmoe_data.SAVE_DIR)
    print(f"[hmoe_repro] SAVE_DIR={hmoe_data.SAVE_DIR}", flush=True)

    if finetune:
        # 优先 meta 备份（预训练 cleanup 后仍可用），否则用 pretrain SAVE_DIR
        _meta_enc = join(meta_dir, "pretrain_encoders")
        _meta_enc_file = (
            _meta_enc + ".klepto"
            if os.path.isfile(_meta_enc + ".klepto")
            else (_meta_enc if os.path.isfile(_meta_enc) else None)
        )
        _pt_tag = f"mm-proto-{seed_tag(ours.seed)}-pt"
        _n_pt = len(proto["train_30"])
        _pt_save = join(
            get_save_path(),
            FLAGS.dataset,
            f"{FLAGS.v_db}_MLP-{FLAGS.pragma_as_MLP}-round{FLAGS.round_num}-"
            f"{_n_pt}kernel-icmp-feb-db-{FLAGS.graph_type}-{FLAGS.task}_"
            f"edge-position-{FLAGS.encode_edge_position}_norm_with-invalid_{FLAGS.invalid}-"
            f"normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_"
            f"tag_{_pt_tag}_{''.join(_target)}",
            "pretrain_dataset",
        )
        _pt_enc = join(_pt_save, "encoders")
        _pt_enc_file = (
            _pt_enc + ".klepto"
            if os.path.isfile(_pt_enc + ".klepto")
            else (_pt_enc if os.path.isfile(_pt_enc) else None)
        )
        if _meta_enc_file is not None:
            FLAGS.encoder_path = _meta_enc
            _enc_file = _meta_enc_file
        elif _pt_enc_file is not None:
            FLAGS.encoder_path = _pt_enc
            _enc_file = _pt_enc_file
        else:
            raise FileNotFoundError(
                "微调需要预训练 encoder，未找到：\n"
                f"  {join(meta_dir, 'pretrain_encoders.klepto')}\n"
                f"  {_pt_enc}.klepto\n"
                "请先完成 --stage pretrain。"
            )
        print(
            f"[hmoe_repro] finetune 复用预训练 encoder: {FLAGS.encoder_path} "
            f"(file={_enc_file})",
            flush=True,
        )

    # data.py 默认用相对路径 ../dse_database；改成绝对路径，避免 cwd 不一致
    _root = get_root_path()
    _benchs = list(FLAGS.benchmarks) if FLAGS.benchmarks else ["machsuite", "poly"]
    hmoe_data.db_path = [
        join(_root, "dse_database", b, "databases", "**", "*") for b in _benchs
    ]
    print(
        f"[hmoe_repro] db_path={hmoe_data.db_path} v_db={FLAGS.v_db} "
        f"round_num={FLAGS.round_num} cwd={os.getcwd()}",
        flush=True,
    )
    # 快速自检：aes 是否能命中 1 个 db
    from glob import iglob as _iglob

    _probe = []
    for _db_p in hmoe_data.db_path:
        _probe.extend(
            [
                f
                for f in _iglob(_db_p, recursive=True)
                if f.endswith(".db")
                and "aes_" in f
                and "large-size" not in f
                and "archive" not in f
                and FLAGS.v_db in f
                and f"one-db-extended-round{FLAGS.round_num}" in f
            ]
        )
    print(f"[hmoe_repro] probe aes_ db hits={len(_probe)} {_probe[:3]}", flush=True)
    if len(_probe) != 1:
        raise FileNotFoundError(
            "找不到 aes 的 HMoE 数据库（期望恰好 1 个）。请确认服务器上存在类似：\n"
            f"  {_root}/dse_database/machsuite/databases/{FLAGS.v_db}/"
            f"**/one-db-extended-round{FLAGS.round_num}/aes_*.db\n"
            f"  当前命中={_probe}"
        )

    from train import train_main  # noqa: E402
    from saver import saver  # noqa: E402

    if not FLAGS.force_regen and os.path.isdir(hmoe_data.SAVE_DIR):
        try:
            dataset = hmoe_data.MyOwnDataset()
            pragma_dim = load(join(hmoe_data.SAVE_DIR, "pragma_dim"))
            print(f"[hmoe_repro] loaded dataset from {hmoe_data.SAVE_DIR}", flush=True)
        except Exception as e:
            print(
                f"[hmoe_repro] load cache failed ({e}); regenerate...",
                flush=True,
            )
            dataset, pragma_dim = hmoe_data.get_data_list()
    else:
        dataset, pragma_dim = hmoe_data.get_data_list()
        print(f"[hmoe_repro] regenerated dataset → {hmoe_data.SAVE_DIR}", flush=True)

    # 记录本 seed 产物指针，方便 Soft MoE 对照
    with open(join(meta_dir, f"{ours.stage}_last_run.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "stage": ours.stage,
                "seed": ours.seed,
                "tag": FLAGS.tag,
                "save_dir": hmoe_data.SAVE_DIR,
                "finetune": bool(FLAGS.finetune),
                "model_path": FLAGS.model_path,
                "kernels": list(hmoe_data.ALL_KERNEL),
                "logs_hint": "HierarchicalMoE-master/src/logs/ … look for tag in path",
            },
            f,
            indent=2,
        )

    test_ratio = FLAGS.test_ratio
    train_main(dataset, pragma_dim, test_ratio=test_ratio, resample=-1)

    # 回写 log 目录，便于 pipeline 定位权重
    try:
        with open(join(meta_dir, f"{ours.stage}_last_run.json"), encoding="utf-8") as f:
            _meta = json.load(f)
        _meta["model_logdir"] = getattr(saver, "model_logdir", None)
        _meta["logdir"] = getattr(saver, "logdir", None)
        with open(join(meta_dir, f"{ours.stage}_last_run.json"), "w", encoding="utf-8") as f:
            json.dump(_meta, f, indent=2)
    except Exception as e:
        print(f"[hmoe_repro] warn: update last_run.json failed: {e}", flush=True)

    do_cleanup = not bool(ours.no_cleanup_save)
    if do_cleanup:
        _cleanup_save_after_stage(
            stage=ours.stage,
            save_dir=hmoe_data.SAVE_DIR,
            meta_dir=meta_dir,
            seed=int(ours.seed),
            get_save_path=get_save_path,
            FLAGS=FLAGS,
            target_names=_target,
        )

    saver.close()
    print(
        f"[hmoe_repro] DONE stage={ours.stage} seed={ours.seed}.\n"
        f"  meta={join(meta_dir, ours.stage + '_last_run.json')}\n"
        f"  权重目录：{getattr(saver, 'model_logdir', None)}\n"
        f"  cleanup_save={do_cleanup}",
        flush=True,
    )


if __name__ == "__main__":
    main()
