#!/usr/bin/env bash
# A1: training-data curve on ZS-CSD. Qwen2.5-7B LoRA SFT on 10/25/50% of train (100% already done),
# then SPIN with a knowledge base built from the SAME subset, plus the gold-parent upper bound.
# Run from make_rag_data/, inside tmux:   GPU=3 bash run_a1.sh
# Put qwen_sft_zscsd.py, map_qwen_preds.py and make_subset_instances.py in make_rag_data/ first.
set -euo pipefail
GPU=${GPU:-3}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
DATA=StanceAgent-R1-main/data/default
I=zscsd_exp/data/zscsd_semantic_instances_llm.jsonl
L=0,0.25,0.5,0.75,1,1.5,2,3,4,6
SUM=zscsd_exp/outputs/a1_data_curve_summary.txt

for F in ${FRACS:-0.1 0.25 0.5}; do
  TAG=f$(python -c "print(int(round($F*100)))")
  D=zscsd_exp/outputs/qwen7b_sft_$TAG
  echo "################ train_frac=$F  ($D)"
  if [ ! -f $D/best_adapter/adapter_config.json ]; then
    T0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$GPU python -u qwen_sft_zscsd.py train --data_dir $DATA --model $MODEL \
      --out $D --train_frac $F 2>&1 | tee $D.train.log
    echo "$TAG train_seconds $(( $(date +%s) - T0 ))" | tee -a zscsd_exp/outputs/a1_train_times.txt
  fi
  [ -f $D/preds_test.jsonl ] || CUDA_VISIBLE_DEVICES=$GPU python -u qwen_sft_zscsd.py predict \
      --data_dir $DATA --model $MODEL --out $D
  python map_qwen_preds.py --instances $I --data_dir $DATA --pred_dir $D --out_dir $D --name qwen7b_sft
  python make_subset_instances.py --instances $I --data_dir $DATA --ids $D/train_ids_used.json \
      --out $D/instances_subset.jsonl
done

# ---- SPIN + gold-parent bound for every fraction (100% uses the full KB)
{
for TAG in f10 f25 f50 full; do
  if [ $TAG = full ]; then D=zscsd_exp/outputs/qwen7b_sft; IS=$I; else D=zscsd_exp/outputs/qwen7b_sft_$TAG; IS=$D/instances_subset.jsonl; fi
  [ -f $D/predictions_qwen7b_sft_test.jsonl ] || continue
  P=$D/predictions_qwen7b_sft
  echo "======== $TAG"
  python -u ${SPIN_REPO:-spin}/spin_core/analyze_combined_prior.py --instances $IS --pred ${P}_test.jsonl \
    --valid_pred ${P}_valid.jsonl --lambdas $L --output_dir $D/combined_prior | sed -n '/^train/p;/基础预测/,/组合/p'
  python ${SPIN_REPO:-spin}/experiments/main/collect_paper_numbers.py --pred Z/SFT-$TAG=$D/combined_prior/pred_combined.jsonl \
    | grep -E "合并 MacroF1|合并 F_avg"
  python -u ${SPIN_REPO:-spin}/experiments/ablation/analyze_combined_ablation.py --instances $IS --pred ${P}_test.jsonl \
    --valid_pred ${P}_valid.jsonl --output_dir $D/ablation | grep -E "基础预测|完整组合（主方法）|人工父立场"
done
} 2>&1 | tee $SUM
echo "summary -> $SUM"