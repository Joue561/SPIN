#!/usr/bin/env bash
# ==============================================================================
# run_mtcsd_pipeline.sh（整理修复后的最终版）
#
# MT-CSD（英文 Reddit 多轮会话立场检测数据集）完整实验流水线：
# 数据标准化 → 目标感知动态窗口 → LLM语义传导标注 → RAG知识库构建
# → LLM零样本消融实验（8方法）→ PLM微调消融实验（8方法）→ 结果汇总
#
# 这一版整合了调试过程中发现并修复的全部 bug，从头跑不会再踩到已知的坑：
#   1. build_target_aware_windows_en.py：英文短文本判断改用词数而非字符数，
#      contains_any 改成词边界匹配，避免 "this"/"true" 这类高频词在长文本
#      里被当子串误判为模糊评论
#   2. prepare_plm_stance_data.py（原 prepare_plm_stance_data_en.py 改名）：
#      标签全部英文化 + current_text 固定放在最前面，无论后面截断多少都
#      不会丢失当前评论本身（之前 plm_semantic_rag_text 截断率一度到96%）
#   3. simple_rag_stance_predictor_en.py：
#      - prompt 里去掉了容易诱导模型对无立场评论强行给方向性判断的措辞
#      - 检索打分改成"可靠性只能在已有内容相关性的基础上调节"，不能
#        凭空让完全不相关的高可靠性文档抢占 top_k
#   4. build_llm_pattern_memory_en.py：修复了 composite_pattern 类型记忆
#      的标题取值 bug（原来会被内部同时存在的 semantic_propagation 分量
#      字段抢先取到，导致上百条本该各不相同的窄类目记忆全部显示成同样
#      的 "stance_reversal"/"stance_amplification"）
#
# 运行前准备：
#   1. git clone https://github.com/nfq729/MT-CSD.git ./MT-CSD_repo
#   2. export OPENAI_API_KEY=...   （用于 LLM 语义传导标注 + LLM 立场预测实验）
#   3. pip install openai transformers torch scikit-learn --break-system-packages
#
# 用法：
#   bash run_mtcsd_pipeline.sh
#
# 重要提醒（调试过程中反复踩过的坑）：
#   如果你改了 prompt / 检索逻辑 / 知识库之后想单独重跑某个方法，
#   --resume true 只看 predictions_{method}_{split}.jsonl 里有没有对应的
#   node_id，不知道你换了新逻辑。旧文件不删，新逻辑不会生效、也不会报错，
#   只会安静地跳过全部样本。改了代码后重跑前，先手动删掉对应的
#   predictions_*.jsonl 和 predictions_*_failed.jsonl。
# ==============================================================================
set -e

MTCSD_REPO=${MTCSD_REPO:-./MT-CSD_repo}
ROOT=${ROOT:-./mtcsd_experiment}
PLM_BASE_MODEL=${PLM_BASE_MODEL:-roberta-base}
LLM_MODEL=${LLM_MODEL:-gpt-4o-mini}
SPLIT=${SPLIT:-test}

mkdir -p "$ROOT"

echo "================================================================"
echo "Step 0/6  数据标准化：MT-CSD -> stance instances"
echo "================================================================"
python3 convert_mtcsd_to_stance_instances.py \
  --input_dir "$MTCSD_REPO" \
  --output_dir "$ROOT/processed_mtcsd"

INSTANCE_COUNT=$(python3 -c "print(sum(1 for _ in open('$ROOT/processed_mtcsd/mtcsd_stance_instances.jsonl', encoding='utf-8')))" 2>/dev/null || echo 0)
if [ "$INSTANCE_COUNT" -eq 0 ]; then
  echo "Step 0 生成了 0 条样本，请检查 MTCSD_REPO=$MTCSD_REPO 是否指向正确的数据目录，流水线终止。"
  exit 1
fi
echo "Step 0 生成 $INSTANCE_COUNT 条样本，继续。"

echo "================================================================"
echo "Step 1/6  目标感知动态会话窗口构建（英文版）"
echo "================================================================"
python3 build_target_aware_windows_en.py \
  --instances "$ROOT/processed_mtcsd/mtcsd_stance_instances.jsonl" \
  --children_map "$ROOT/processed_mtcsd/mtcsd_children_map.json" \
  --output_dir "$ROOT/processed_mtcsd_windows" \
  --up_k_default 2 --up_k_expanded 3 --down_k 5

echo "================================================================"
echo "Step 2/6  LLM 语义传导标注（英文 prompt，train + valid/test 分别跑）"
echo "================================================================"
python3 build_semantic_propagation_annotations_llm_en.py \
  --windowed_instances "$ROOT/processed_mtcsd_windows/mtcsd_windowed_instances.jsonl" \
  --output_dir "$ROOT/processed_mtcsd_semantic_llm_train" \
  --provider openai --model "$LLM_MODEL" \
  --only_split train --context_mode both --resume true

python3 build_semantic_propagation_annotations_llm_en.py \
  --windowed_instances "$ROOT/processed_mtcsd_windows/mtcsd_windowed_instances.jsonl" \
  --output_dir "$ROOT/processed_mtcsd_semantic_llm_validtest" \
  --provider openai --model "$LLM_MODEL" \
  --context_mode both --resume true

python3 - <<PYEOF
import json, os
paths = [
    "$ROOT/processed_mtcsd_semantic_llm_train/mtcsd_semantic_instances_llm.jsonl",
    "$ROOT/processed_mtcsd_semantic_llm_validtest/mtcsd_semantic_instances_llm.jsonl",
]
out = "$ROOT/processed_mtcsd_semantic_llm_all/mtcsd_semantic_instances_llm.jsonl"
os.makedirs(os.path.dirname(out), exist_ok=True)
n = 0
with open(out, "w", encoding="utf-8") as fo:
    for p in paths:
        if not os.path.exists(p):
            print(f"  跳过不存在的文件：{p}")
            continue
        with open(p, encoding="utf-8") as fi:
            for line in fi:
                fo.write(line)
                n += 1
print(f"merged -> {out}（{n} 条）")
PYEOF

echo "================================================================"
echo "Step 3/6  结构化语言模式记忆库 + RAG 知识库构建（已修复 pname bug）"
echo "================================================================"
python3 build_llm_pattern_memory_en.py \
  --input "$ROOT/processed_mtcsd_semantic_llm_all/mtcsd_semantic_instances_llm.jsonl" \
  --output_dir "$ROOT/processed_mtcsd_rag_memory" \
  --min_confidence 0.70 --min_pattern_count 10 --train_only true

echo "验证 composite_pattern 标题是否都是完整组合键..."
python3 -c "
import json
bad = []
with open('$ROOT/processed_mtcsd_rag_memory/rag_knowledge_documents_llm.jsonl', encoding='utf-8') as f:
    for line in f:
        doc = json.loads(line)
        if doc.get('memory_type') == 'composite_pattern' and '+' not in str(doc.get('pattern_name')):
            bad.append(doc.get('doc_id'))
if bad:
    print(f'仍有 {len(bad)} 条 composite_pattern 标题异常，pname bug 没有修复干净')
    raise SystemExit(1)
else:
    print('知识库 pname 检查通过')
"

echo "================================================================"
echo "Step 4/6  实验一：LLM 零样本提示消融实验（8 种方法，split=$SPLIT）"
echo "================================================================"
for method in current_only root_current parent_current full_path \
              dynamic_window dynamic_semantic dynamic_rag full_rag; do
  echo ""
  echo "---- [$method] 预测 ----"
  python3 simple_rag_stance_predictor_en.py \
    --instances "$ROOT/processed_mtcsd_semantic_llm_all/mtcsd_semantic_instances_llm.jsonl" \
    --knowledge "$ROOT/processed_mtcsd_rag_memory/rag_knowledge_documents_llm.jsonl" \
    --output_dir "$ROOT/rag_prediction_outputs" \
    --split "$SPLIT" --method "$method" --top_k 5 \
    --provider openai --model "$LLM_MODEL" \
    --resume true

  echo "---- [$method] 评估 ----"
  python3 evaluate_stance_predictions.py \
    --predictions "$ROOT/rag_prediction_outputs/predictions_${method}_${SPLIT}.jsonl" \
    --output_dir "$ROOT/rag_evaluation_outputs/${method}_${SPLIT}" \
    --group_by true
done

echo ""
echo "汇总 LLM 消融结果..."
python3 collect_llm_summary_results.py \
  --eval_dir "$ROOT/rag_evaluation_outputs" \
  --split "$SPLIT" \
  --output_dir "$ROOT/rag_evaluation_outputs"

echo "================================================================"
echo "Step 5/6  实验二：PLM 微调消融实验（8 种方法，base_model=$PLM_BASE_MODEL）"
echo "================================================================"
set +e
python3 run_full_plm_pipeline.py \
  --instances "$ROOT/processed_mtcsd_semantic_llm_all/mtcsd_semantic_instances_llm.jsonl" \
  --knowledge "$ROOT/processed_mtcsd_rag_memory/rag_knowledge_documents_llm.jsonl" \
  --data_dir "$ROOT/plm_experiment_data" \
  --model_output_dir "$ROOT/plm_model_outputs" \
  --error_output_dir "$ROOT/plm_error_analysis" \
  --log_dir "$ROOT/pipeline_logs" \
  --base_model "$PLM_BASE_MODEL" \
  --top_k 5 --max_length 512 --epochs 5 --batch_size 16 \
  --learning_rate 2e-5 --seed 42 --fp16 true
PLM_EXIT=$?
set -e
if [ $PLM_EXIT -ne 0 ]; then
  echo "run_full_plm_pipeline.py 退出码非 0（可能部分方法失败），继续尝试独立汇总已完成的部分。"
fi

echo "汇总 PLM 消融结果（不依赖上一步是否全部成功）..."
python3 collect_plm_summary_results.py \
  --model_output_dir "$ROOT/plm_model_outputs" \
  --base_model "$PLM_BASE_MODEL" \
  --eval_split "$SPLIT" \
  --output_dir "$ROOT/plm_model_outputs"

echo "================================================================"
echo "Step 6/6  下一步：跟 C-MTCSD 结果对照（可选）"
echo "================================================================"
echo "如果你有 C-MTCSD 的历史结果文件，运行："
echo ""
echo "python3 compare_cmtscd_mtcsd_results.py \\"
echo "  --mtcsd_llm_eval_dir $ROOT/rag_evaluation_outputs \\"
echo "  --mtcsd_plm_summary $ROOT/plm_model_outputs/summary_results.json \\"
echo "  --cmtscd_llm_phase2_summary <C-MTCSD LLM汇总文件路径> \\"
echo "  --cmtscd_plm_summary <C-MTCSD PLM汇总文件路径> \\"
echo "  --output $ROOT/cmtscd_vs_mtcsd_comparison.md"
echo ""
echo "全部完成。"
echo "  LLM 消融汇总： $ROOT/rag_evaluation_outputs/llm_summary_results.csv"
echo "  PLM 消融汇总： $ROOT/plm_model_outputs/summary_results.csv"
