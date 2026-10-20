#!/usr/bin/env bash
# 一键跑 HMoE：seed 列表上 split → pretrain(1000) → finetune(500) → 汇总结果
# 用法（在 HierarchicalMoE-master 下）:
#   bash multimodal_moe/hmoe_repro/run_seeds.sh
#   bash multimodal_moe/hmoe_repro/run_seeds.sh 65 66 67 68
#   SEEDS="65 66" bash multimodal_moe/hmoe_repro/run_seeds.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ $# -gt 0 ]]; then
  SEEDS=("$@")
elif [[ -n "${SEEDS:-}" ]]; then
  # shellcheck disable=SC2206
  SEEDS=($SEEDS)
else
  SEEDS=(65 66 67 68)
fi

OUT_ROOT="$ROOT/multimodal_moe/hmoe_repro_outputs"
SUMMARY="$OUT_ROOT/pipeline_summary.tsv"
mkdir -p "$OUT_ROOT"
echo -e "seed\tstage\tstatus\tbest_val\tfinal_test\tbest_test\tmodel_path\tlog_dir" > "$SUMMARY"

LOG_ROOT="$ROOT/src/logs"
PY="${PYTHON:-python}"

pick_newest_val_ckpt() {
  # $1 = unix mtime lower bound (seconds)
  local since="$1"
  local best="" best_epoch=-1 best_mtime=0
  while IFS= read -r -d '' f; do
    local mt epoch
    mt=$(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f")
    (( mt < since )) && continue
    # 文件名如 1000_val_model_state_dict.pth
    epoch=$(basename "$f" | sed -n 's/^\([0-9]*\)_val_model_state_dict\.pth$/\1/p')
    [[ -z "$epoch" ]] && continue
    if (( epoch > best_epoch )) || { (( epoch == best_epoch )) && (( mt > best_mtime )); }; then
      best="$f"
      best_epoch=$epoch
      best_mtime=$mt
    fi
  done < <(find "$LOG_ROOT" -name '*_val_model_state_dict.pth' -print0 2>/dev/null)
  echo "$best"
}

pick_newest_train_ckpt() {
  local since="$1"
  local best="" best_epoch=-1 best_mtime=0
  while IFS= read -r -d '' f; do
    local mt epoch
    mt=$(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f")
    (( mt < since )) && continue
    epoch=$(basename "$f" | sed -n 's/^\([0-9]*\)_train_model_state_dict\.pth$/\1/p')
    [[ -z "$epoch" ]] && continue
    if (( epoch > best_epoch )) || { (( epoch == best_epoch )) && (( mt > best_mtime )); }; then
      best="$f"
      best_epoch=$epoch
      best_mtime=$mt
    fi
  done < <(find "$LOG_ROOT" -name '*_train_model_state_dict.pth' -print0 2>/dev/null)
  echo "$best"
}

parse_best_line() {
  # 从日志文件尾部抓 Best val / Best train
  local logf="$1"
  grep -E '^Best (val|train):' "$logf" | tail -n 1 || true
}

echo "[pipeline] roots=$ROOT seeds=${SEEDS[*]}"

for seed in "${SEEDS[@]}"; do
  seed_dir="$OUT_ROOT/seed_${seed}"
  mkdir -p "$seed_dir"
  echo "========== seed=${seed} =========="

  # 1) split
  $PY -m multimodal_moe.hmoe_repro.run_hmoe --stage split_only --seed "$seed" \
    | tee "$seed_dir/split_only.log"

  # 2) pretrain
  t0=$(date +%s)
  set +e
  $PY -m multimodal_moe.hmoe_repro.run_hmoe \
    --stage pretrain --seed "$seed" --force_regen --epoch_num 1000 \
    2>&1 | tee "$seed_dir/pretrain.log"
  pt_rc=${PIPESTATUS[0]}
  set -e

  val_ckpt="$(pick_newest_val_ckpt "$t0")"
  pt_best="$(parse_best_line "$seed_dir/pretrain.log")"
  # Best val: 0.1081, final test: 0.0779, best test: 0.0778
  bv=$(echo "$pt_best" | sed -n 's/.*Best val: \([0-9.]*\).*/\1/p')
  ft=$(echo "$pt_best" | sed -n 's/.*final test: \([0-9.]*\).*/\1/p')
  bt=$(echo "$pt_best" | sed -n 's/.*best test: \([0-9.]*\).*/\1/p')
  log_dir=""
  [[ -n "$val_ckpt" ]] && log_dir="$(dirname "$val_ckpt")"

  if [[ -z "$val_ckpt" || ! -f "$val_ckpt" ]]; then
    echo -e "${seed}\tpretrain\tFAIL\t${bv}\t${ft}\t${bt}\t\t" >> "$SUMMARY"
    echo "[pipeline] seed=${seed} pretrain FAILED (no val ckpt). skip finetune."
    continue
  fi
  echo -e "${seed}\tpretrain\tOK\t${bv}\t${ft}\t${bt}\t${val_ckpt}\t${log_dir}" >> "$SUMMARY"
  echo "[pipeline] seed=${seed} val_ckpt=$val_ckpt"

  # 3) finetune（原版默认 500）
  t1=$(date +%s)
  set +e
  $PY -m multimodal_moe.hmoe_repro.run_hmoe \
    --stage finetune --seed "$seed" --force_regen --epoch_num 500 \
    --model_path "$val_ckpt" \
    2>&1 | tee "$seed_dir/finetune.log"
  ft_rc=${PIPESTATUS[0]}
  set -e

  # 普通 finetune 与 pretrain 共用 train_main，存的是 *_val_model_state_dict.pth
  ft_ckpt="$(pick_newest_val_ckpt "$t1")"
  if [[ -z "$ft_ckpt" ]]; then
    ft_ckpt="$(pick_newest_train_ckpt "$t1")"
  fi
  ft_best="$(parse_best_line "$seed_dir/finetune.log")"
  # 常见: Best val: ..., final test: ..., best test: ...
  bva=$(echo "$ft_best" | sed -n 's/.*Best val: \([0-9.]*\).*/\1/p')
  fv=$(echo "$ft_best" | sed -n 's/.*final test: \([0-9.]*\).*/\1/p')
  bt2=$(echo "$ft_best" | sed -n 's/.*best test: \([0-9.]*\).*/\1/p')
  if [[ -z "$bva" ]]; then
    bva=$(echo "$ft_best" | sed -n 's/.*best val: \([0-9.]*\).*/\1/p')
    fv=$(echo "$ft_best" | sed -n 's/.*final val: \([0-9.]*\).*/\1/p')
    bt2=$(echo "$ft_best" | sed -n 's/.*Best train: \([0-9.]*\).*/\1/p')
  fi
  ft_log_dir=""
  [[ -n "$ft_ckpt" ]] && ft_log_dir="$(dirname "$ft_ckpt")"
  status=OK
  [[ -z "$ft_ckpt" || ! -f "$ft_ckpt" ]] && status=FAIL
  echo -e "${seed}\tfinetune\t${status}\t${bva}\t${fv}\t${bt2}\t${ft_ckpt}\t${ft_log_dir}" >> "$SUMMARY"
  echo "[pipeline] seed=${seed} finetune status=${status} ckpt=${ft_ckpt}"
  # 方便 Soft MoE 对照的指针
  printf '%s\n' "$ft_ckpt" > "$seed_dir/finetune_best_ckpt.txt"
  printf '%s\n' "$val_ckpt" > "$seed_dir/pretrain_val_ckpt.txt"
done

echo
echo "[pipeline] DONE. summary → $SUMMARY"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
