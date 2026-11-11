# C-MTCSD 跨目标泛化：结构化模式记忆增强立场检测（Leave-One-Topic-Out）

## 目录

```
ct_common.py                              共享工具（IO/日志/规范化/去话题化/JSD/指标）
01_build_cross_topic_splits.py            LOTO 划分（source_train / target_eval / to_annotate）
02_build_cross_topic_memory.py            仅用 source_train 构建 4 类 memory + source / pattern-only RAG 文档
03_compute_memory_transferability.py      Cov / Stab / Rel / TransRel
04_retrieve_cross_topic_memory.py         source / pattern_only / transferable / random 四种检索 + rag_features
05_build_cross_topic_prompts.py           6 种 method 的 prompt（字段白名单 + 泄漏扫描）
06_run_cross_topic_llm_prediction.py      openai / deepseek / qwen / local / mock；断点续跑、重试、JSON 修复、dry-run
07_evaluate_cross_topic_predictions.py    主指标、复杂子集、transfer_gain、配对 bootstrap / McNemar、论文总表
08_analyze_transferability.py             TransRel 分桶、HTR 相关、稳定模式、帮助/误导案例、证据有效性、自动结论
09_run_all_cross_topic_experiments.py     一键运行
tools/make_synthetic_data.py              合成数据（仅冒烟测试）
detopic_lexicon.example.json              去话题化补充词表示例
```

## 安装

```bash
pip install -r requirements.txt
```

## 一键运行

```bash
export OPENAI_API_KEY=sk-xxx
# 如通过代理/中转访问 OpenAI：export OPENAI_BASE_URL=https://your-proxy/v1  （或 --base_url）
python 09_run_all_cross_topic_experiments.py \
  --instances processed_cmtscd_semantic_llm/cmtscd_semantic_instances_llm.jsonl \
  --output_dir cross_topic_outputs \
  --methods current_only dynamic_semantic rag_empty source_rag pattern_only_rag transferable_rag random_memory_rag \
  --top_k 5 --min_confidence 0.70 --min_pattern_count 5 --high_transfer_threshold 0.45 \
  --provider openai --model gpt-4o-mini --run_prediction true
```

建议先小规模验证：

```bash
# 先不调 API，检查 prompt
python 09_run_all_cross_topic_experiments.py --instances ... --output_dir cross_topic_outputs --dry_run true
# 单个 held-out、每个 30 条、DEBUG 日志
python 09_run_all_cross_topic_experiments.py --instances ... --output_dir ct_debug --debug true --only_heldout_topic "iPhone 15"
# 中断后续跑（已完成的步骤/topic 跳过，预测断点续跑）
python 09_run_all_cross_topic_experiments.py --instances ... --output_dir cross_topic_outputs --skip_existing true
```

其他常用开关：`--from_step 4 --to_step 8`（只重跑后半段，配合 `--overwrite true` 或 `--skip_existing true`）、
`--transfer_mode filter`、`--context_mode posthoc`（含子回复反馈，需作为 posthoc setting 单独报告）、
`--provider local --base_url http://localhost:8000/v1 --model xxx`（vLLM/Ollama 等 OpenAI 兼容服务）。
qwen 使用 `DASHSCOPE_API_KEY`，openai 使用 `OPENAI_API_KEY`。

## 防泄漏机制

| 环节 | 机制 |
|---|---|
| 01 | `source_train` 只含非 held-out 且 `split=train`；断言 held-out 不在 source、uid 无重叠 |
| 02 | 读取 `source_train` 后再次检查：出现 held-out 样本或非 train 样本直接终止；memory 的 `source_topics` 不得含 held-out |
| 03 | 只用 source topics 的分布计算 TransRel |
| 04 | 只读 target_eval 的文本与 LLM 语义字段；输出不含标签；若 memory 来源含 held-out 直接终止 |
| 05 | 字段白名单构建 prompt；`stance_label`、`stance_label_zh`、`llm_inferred_stance_to_main_target`、`llm_reason_short` 永不进入；构建后扫描禁用字段名 |
| 07/08 | 唯一读取 gold 标签的阶段 |

`processed_cmtscd_rag_memory/rag_knowledge_documents_llm.jsonl`（全量 memory）在本实验中**不使用**，每一轮都由 source topics 重建。

## 关键实现选择（论文方法部分需写明）

1. **评估集口径**：默认 held-out 的 `split=test`；若其语义覆盖率 < 0.5 或可用样本 < 20，回退到 `--fallback_eval_split`（默认 all）。由于 held-out 的任何样本都不参与 memory，用其 train/dev 评估不构成泄漏，但需在论文中注明。每轮实际口径记录在 `split_report.json → eval_selection`。
2. **缺语义字段**：held-out 中缺 `llm_reply_relation / llm_semantic_propagation / llm_target_relation` 的样本写入 `to_annotate.jsonl`，默认不进入评估（`--keep_missing_semantic true` 可保留并填 unknown）。
3. **低置信样本**：默认降权为 0.5（`--low_conf_mode exclude` 可排除）；缺失置信度权重 0.8。
4. **弱结构样本**（目标关系 irrelevant/unclear/unknown 或回复关系 unknown）：默认只进入 target_relation memory 且 `suggested_usage=weak`，不进入 strong memory。
5. **Stab**：公式对全部源话题平均，但无样本话题的 P_q 无定义。默认只在覆盖话题（N≥m）上平均，未覆盖的惩罚由 Cov 承担（`--stab_over all` 则未覆盖话题按 JSD=1）。JSD 以 2 为底，取值 [0,1]；`--jsd_form distance` 使用 √JSD，区分度更高，可作敏感性分析。
6. **Rel**：`0.4·level + 0.3·(p_dom−1/3)/(2/3) + 0.2·avg_conf + 0.1·min(1, log(1+n)/log(51))`，level: high=1/medium=0.7/low=0.4（`--rel_weights` 可调）。
7. **检索打分**：严格按 rule_hybrid（+3/+2/+2/+2/+1/+1/+0.5×重叠率）。在此之上加了两个约束（均可关闭）：结构匹配分 ≥ 2（`--min_structural_score`），复合 memory 必须命中候选模式（`--composite_require_pattern`）。原因：合成数据测试中，高可靠的复合 memory 仅凭 `target_relation=direct` 就能拿到 4 分，被大量误召回，导致 TransRel 与证据正确率负相关。
8. **TransRel 在 prompt 中只对 transferable_rag 显示**，这样 source_rag 与 transferable_rag 的差别只在检索排序，比较更干净。transferable_rag 摘要默认用 source 风格（`--transferable_text pattern_only` 可切换）。
9. **pattern-only**：用内置五个话题的词表 + 每折样本的 `main_target / target_anchor / topic` 替换为 `[TARGET]`，属性词 → `[ATTRIBUTE]`，后果词 → `[CONSEQUENCE]`；去话题化后重复的例句会去重。残留话题词比例写在 `memory_build_report.json → pattern_only_residual_rate`。真实数据上建议检查后用 `--detopic_lexicon` 补词。
10. **公平比较**：07 默认只在同一 held-out 下所有方法都有有效预测的样本交集上计算（`--eval_on_intersection false` 可关闭）。

## v2 修改（试跑诊断后）

试跑发现：四种 RAG 方法逐条预测几乎完全一致（random 也一样），且都把 gold=none 的样本改判为有立场，模型理由显示是被原第 8 条规则“映射到当前主目标”推着硬找立场。为此：

1. **重写 RAG 规则 8–9**（规则 1–7 与 dynamic_semantic 保持一致，dynamic_semantic / current_only 的 prompt 未改动）：记忆只用于理解语用功能，不能直接推断对主目标的立场；只讨论相关话题而未对主目标表态 → none。
2. **记忆摘要增加“立场提示”行**：给出该模式在源话题中 none 的比例。需重跑 04 生成新摘要。
3. **新增对照组 `rag_empty`**：RAG 规则与格式完全相同，但记忆为空。
   - 指令效应 = rag_empty − dynamic_semantic
   - 记忆内容效应 = X_rag − rag_empty
   07 的配对比较和 08 的报告（“指令效应 vs 记忆内容效应”一节）已包含这些对比。
4. **断点续跑按 prompt_hash 判断**：prompt 改过的旧预测会自动移入 `predictions/predictions_stale/` 备份并重新预测，不会新旧混用。

## v3 修改：改设计（路线 B）

v2 试跑显示：结构化记忆跨话题仍有信号（证据正确率约 0.60，随机 0.33），但总是弱于模型自身判断（dynamic_semantic 约 0.69），且离线门控也无法超过基线——记忆提供的结构信息已被 prompt 中的语义字段覆盖。因此新增三个方法：

| 方法 | prompt 组成 | 检验的问题 |
|---|---|---|
| `context_only` | 会话上下文 + 当前评论（无语义字段、无记忆） | 去语义字段的基线 |
| `context_memory` | 会话上下文 + 当前评论 + source_rag 检索到的结构化记忆（无语义字段） | 记忆能否替代显式语义标注（对比 context_only） |
| `example_rag` | dynamic_semantic 的全部内容 + 源话题结构相似的真实示例（父评论→当前评论→标签） | 示例比统计分布更有用吗（对比 dynamic_semantic / rag_empty / source_rag） |

- `example_rag` 在 04 中从 source_train 检索（结构打分规则同 rule_hybrid），每个源话题最多 2 条（`--example_max_per_topic`），只用源话题标签，不会泄漏。
- `context_memory` 复用 `retrieval_source_rag.jsonl`，不需要单独检索；检索键仍是目标样本的 LLM 语义字段，只是这些字段不直接展示给预测模型。
- 已有方法的 prompt 完全不变（prompt_hash 相同），已跑的预测可以继续复用。

## v4：同话题对照 `intopic_rag`

用于在**同一批 held-out 评估样本**上比较“同话题记忆”与“跨话题记忆”：

- 记忆 = 源话题 train + held-out 话题自身的 train（由 `02 --intopic true --instances ...` 构建，输出到 `memories_intopic/`，memory_id 带 `IN_` 前缀）。
- 绝不含评估样本：02 用与 01 相同的 sample_uid 规则读取 held-out train，并排除所有 target_eval 样本；04 再检查一次记忆示例与评估集无重叠。
- 检索方式、prompt 格式与 source_rag 完全一致，只有记忆来源不同。
- 对比：`intopic_rag − dynamic_semantic` = 同话题记忆增益；`source_rag − dynamic_semantic` = 跨话题记忆增益；`intopic_rag − source_rag` = 迁移损失。07 的配对比较和 08 报告中的“同话题 vs 跨话题记忆”一节会给出这些结果。

其他改动：
- 06 新增 `--sample_order hash`：配合 `--sample_limit`，按 (seed, sample_uid) 哈希抽样，各方法抽到同一批样本，比取文件前 N 条（集中在少数讨论串）更有代表性。
- 08 新增 `--methods` 与 `--eval_on_intersection`（默认 true），与 07 一样只在所选方法共同预测的样本上分析。

## 主要输出

```
splits/{topic}/source_train.jsonl, target_eval.jsonl, to_annotate.jsonl, split_report.json
memories/{topic}/{language_pattern,semantic_propagation,target_relation,composite_pattern}_memory.jsonl
memories/{topic}/memory_units.jsonl, rag_knowledge_documents_{source,pattern_only}.jsonl, memory_build_report.json
memories/{topic}/memory_transferability.jsonl, high/low_transfer_memory.jsonl, memory_transferability_report.json
retrieval/{topic}/retrieval_{method}.jsonl, retrieval_report.json
prompts/{topic}/prompts_{method}.jsonl
predictions/predictions_{topic}_{method}.jsonl, prediction_failures.jsonl, prediction_run_report.json, pending/
metrics/summary_by_topic_method.csv, summary_average.csv, complex_subset_metrics.json,
        confusion_matrices.json, transfer_gain_report.json, final_summary_table.md
analysis/transferability_bucket_metrics.json, top_transferable_patterns.csv, low_transfer_failure_cases.jsonl,
         rag_help_cases.jsonl, rag_harm_cases.jsonl, memory_type_contribution.json,
         rag_evidence_validity.json, htr_correlation.json, analysis_report.json, analysis_report.md
```

- `final_summary_table.md`：论文总表（逐 topic + average 行）及配对显著性。
- `analysis_report.md`：九个问题的自动结论（Q1–Q9）。
- `rag_evidence_validity.json`：检索证据主导立场与 gold 的一致率、模型跟随率、跟随与不跟随时的准确率——用于论证“RAG 不是装饰”（结构化检索的证据正确率应明显高于 random_memory_rag）。

## 覆盖保护

所有脚本默认不覆盖已有输出（报错退出）；`--overwrite true` 覆盖，`--skip_existing true` 跳过已完整完成的部分。全局汇总报告（如 `all_split_summary.json`）已存在时自动另存为带时间戳的文件。06 的预测文件按 `--resume true` 追加续跑。

## 冒烟测试

```bash
python tools/make_synthetic_data.py --output synthetic/inst.jsonl --n_per_topic 250
python 09_run_all_cross_topic_experiments.py --instances synthetic/inst.jsonl --output_dir synthetic_out --provider mock
```

`provider=mock` 只是启发式规则（RAG 时跟随记忆多数票），仅用于验证流程，结果不能用于论文。