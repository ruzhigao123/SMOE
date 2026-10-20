"""
入口：消融训练（恢复最初 only-cdfg / only-ast 方案为默认）。

最初协议：
  - 划分：40→30/10 核，30 内再 24/6 核
  - 单模态：cdfg_only / ast_only / LM → 编码到 D → MLP（无融合）

  python -m multimodal_moe.src.main_ablation --subtask split_only
  python -m multimodal_moe.src.main_ablation --subtask train --ablation_mode cdfg_only --no_resume
  # IronMan + 小样本适应（test_10, k=50）
  python -m multimodal_moe.src.main_ablation --subtask train \\
    --ablation_mode cdfg_only --comparative_if true --comparative_model ironman --no_resume
  python -m multimodal_moe.src.main_ablation --subtask finetune \\
    --ablation_mode cdfg_only --comparative_if true --comparative_model ironman \\
    --transfer_k_shot 50 --coreset kmeans
"""
from __future__ import annotations

import sys
from os.path import abspath, dirname, join

_SRC = dirname(abspath(__file__))
_MM = dirname(_SRC)
_HMOE = dirname(_MM)
if _HMOE not in sys.path:
    sys.path.insert(0, _HMOE)

from multimodal_moe.src.config import FLAGS  # noqa: E402
from multimodal_moe.src.kernel_protocol import build_protocol  # noqa: E402
from multimodal_moe.src.train_ablation import (  # noqa: E402
    finetune_one_mode,
    infer_one_mode,
    run_all_modes,
    train_one_mode,
)
from multimodal_moe.src.utils import create_dir_if_not_exists  # noqa: E402


def main():
    create_dir_if_not_exists(FLAGS.save_dir)
    if FLAGS.subtask == "split_only":
        proto = build_protocol(
            seed=FLAGS.seed,
            save_json=join(FLAGS.save_dir, "kernel_split.json"),
        )
        print("dropped_2:", proto["dropped_2"])
        print("train_30:", proto["train_30"])
        print("test_10:", proto["test_10"])
        print("train_24:", proto["train_24"])
        print("val_6:", proto["val_6"])
        return
    if FLAGS.subtask == "run_all":
        run_all_modes()
        return
    if FLAGS.subtask == "train":
        train_one_mode(FLAGS.ablation_mode)
        return
    if FLAGS.subtask == "finetune":
        finetune_one_mode(FLAGS.ablation_mode)
        return
    if FLAGS.subtask == "inference":
        infer_one_mode(FLAGS.ablation_mode)
        return
    raise ValueError(FLAGS.subtask)


if __name__ == "__main__":
    main()
