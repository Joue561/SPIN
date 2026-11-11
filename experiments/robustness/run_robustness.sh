#!/usr/bin/env bash
# 一次性运行全部审稿回应实验（问题一 A/B/C + 问题二），三个实验并行，纯 CPU，不调用 API，不改 SPIN 主代码。
# 用法（在 tmux 中）：
#   tmux new -s robust
#   cd ~/wlgcn/wlgcn/make_rag_data && bash ${SPIN_REPO:-spin}/experiments/robustness/run_robustness.sh
# 查看进度：tail -f robustness/symmetry.log  （或 channel.log / perturb.log）
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "${WORKDIR:-$HOME/wlgcn/wlgcn/make_rag_data}"   # 数据所在目录
TOOLS=$REPO/spin_core
R=$REPO/experiments/robustness
M=${M:-$REPO/configs/runs.json}
OUT=${OUT:-robustness}
mkdir -p $OUT/symmetry $OUT/channel $OUT/perturb $OUT/figs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

echo "[$(date +%T)] 启动三个实验（并行）"
python -u $R/transition_symmetry.py --tools_dir $TOOLS --manifest $M --output_dir $OUT/symmetry \
  --variants sym,full > $OUT/symmetry.log 2>&1 &
P1=$!
python -u $R/channel_shift_diagnosis.py --tools_dir $TOOLS --manifest $M --output_dir $OUT/channel \
  --n_perm 1000 > $OUT/channel.log 2>&1 &
P2=$!
python -u $R/channel_perturbation.py --tools_dir $TOOLS --manifest $M --output_dir $OUT/perturb \
  --n_boot 200 --em_label_shift > $OUT/perturb.log 2>&1 &
P3=$!

FAIL=0
for p in "$P1:symmetry" "$P2:channel" "$P3:perturb"; do
  pid=${p%%:*}; name=${p##*:}
  if wait "$pid"; then echo "[$(date +%T)] $name 完成"; else echo "[$(date +%T)] $name 出错，见 $OUT/$name.log"; FAIL=1; fi
done

python $R/plot_robustness.py --root $OUT --out $OUT/figs > $OUT/plot.log 2>&1 || echo "绘图出错，见 $OUT/plot.log"

# 打包要发给我的结果（只含日志和 CSV/JSON，不含数据）
tar czf $OUT/robustness_results.tgz $OUT/*.log $OUT/symmetry/*.csv $OUT/channel/*.csv $OUT/channel/*.json \
  $OUT/perturb/*.csv $OUT/figs/*.pdf 2>/dev/null
echo "[$(date +%T)] 全部结束（FAIL=$FAIL）。请把 $OUT/robustness_results.tgz 发给我。"