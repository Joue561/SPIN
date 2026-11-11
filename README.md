# SPIN: Semantic Propagation INference for Multi-Turn Conversational Stance Detection

SPIN 不需要训练任何模型，用来修正任意基础模型在会话树上的立场预测：
1. 用 LLM 给每条回复标注它与父节点的语义关系；
2. 用训练集统计出语义传导知识库；
3. 把基础模型的预测当作带噪声的观测，沿会话树自上而下做一次推断。

## 目录结构

```
spin/
├── data_prep/              1. 数据准备：原始数据 → 统一的节点格式、会话树、上下文窗口
│   ├── cmtcsd/             C-MTCSD（中文，目标内）
│   ├── zscsd/              ZS-CSD（中文，零样本）
│   └── mtcsd/              MT-CSD（英文，附录）
├── annotation/             2. 语义传导标注：LLM 给每条回复标注回复关系 / 传导类型 / 目标关系（知识库的检索键）
├── base_models/            3. 基础模型：产出 valid / test 逐条预测，作为 SPIN 的输入
│   ├── llm/                提示式 LLM（kNN 少样本 / 零样本 CoT / 完整路径）
│   ├── glan/               GLAN（C-MTCSD 上的监督 SOTA）
│   ├── sitpcl/             SITPCL（ZS-CSD 上的监督 SOTA）
│   └── qwen_sft/           Qwen2.5-7B LoRA SFT（含低资源 10/25/50%）
├── spin_core/              4. SPIN 本体：知识库构建 + 观测信道 + 会话树推断（6 个文件互相依赖，须放在同一目录）
├── experiments/            5. 论文中的实验与分析
│   ├── main/               主表、显著性检验、数据集统计
│   ├── ablation/           消融、检索键显著性、动机分析、学习式融合对照
│   ├── analysis/           误差分析、修正空间、迁移诊断、分组结果
│   ├── negative/           附录"负面结果"中报告的尝试
│   └── robustness/         审稿回应：观测信道鲁棒性、转移矩阵对称性
├── figures/                论文图
├── configs/                实验清单（runs*.json）
├── tools/                  辅助工具（只读导出 SPIN 实现，便于核对）
├── tests/                  合成数据（冒烟测试用，不可用于论文结果）
└── legacy/                 早期方案（论文最终未使用，保留备查）
    ├── rag_memory/         结构化模式记忆 + RAG 提示（SPIN 之前的方案）
    ├── plm/                PLM 微调消融（MacBERT 等）
    ├── cross_topic_loto/   留一话题的跨话题 RAG 实验（01–09）
    ├── human_audit/        人工审核与 AI 预标注
    └── misc/
```

所有脚本都**在数据所在的工作目录下运行**（例如 `~/wlgcn/wlgcn/make_rag_data`），代码通过路径调用，数据不进仓库。

## SPIN 本体（`spin_core/`）

这 6 个文件是历次离线实验逐步叠加形成的，各自的 `main()` 多为探索性实验；SPIN 真正用到的只有下表中的函数。它们之间按"同目录文件路径"互相导入，所以必须放在同一目录。

| 文件 | SPIN 用到的部分 | 论文对应 |
|---|---|---|
| `analyze_depth_dynamics.py` | `load_instances`（节点、父节点、`pa_gold`、语义字段）、`norm`、`scores`、`relation`、`depth_bucket` | 问题定义 §IV-A |
| `analyze_propagation_kb.py` | `PatternKB`（相对知识 R / N、绝对知识 A、逐表回退、0.5 平滑）、检索键 `k_full` | 知识库 §IV-C，式 (2) |
| `analyze_kb_confidence.py` | `confusion`（观测信道 E）、`load_pred_conf` | 观测模型 §IV-D，式 (1) |
| `analyze_kb_calibration.py` | 被上面的文件导入（其 `main()` 是附录"跨话题知识库"的 EM 基准率校准） | 附录负面结果 |
| `analyze_conflict_detection.py` | `KB_KEYS`、`scores_of`（其 `main()` 是附录"LLM 复核"的选点实验） | — |
| `analyze_combined_prior.py` | **主入口**：`forward`（式 (3) 的单次前向推断）、λ 网格在 valid 上选择 | §IV-E、§IV-F，表 II / III |

运行 SPIN（论文用的 λ 网格是 0–6，脚本默认只到 2，**必须显式指定**）：
```bash
python -u spin/spin_core/analyze_combined_prior.py \
  --instances processed_cmtscd_semantic_llm_causal_v4/cmtscd_semantic_instances_llm.jsonl \
  --valid_pred baseline_outputs/predictions_knn_fewshot_valid.jsonl \
  --pred baseline_outputs/predictions_knn_fewshot_test.jsonl \
  --lambdas 0,0.25,0.5,0.75,1,1.5,2,3,4,6 --output_dir combined_prior/c_gpt4omini_knn
```

## 流程与论文对应

| 步骤 | 脚本 | 论文 |
|---|---|---|
| C-MTCSD 转换 | `data_prep/cmtcsd/convert_cmtscd_to_stance_instances.py` → `build_target_aware_windows.py` → `add_root_posts.py`（补原帖为深度 0 节点） | §V-A、附录 A |
| 划分核对 | `data_prep/cmtcsd/check_official_split.py`、`find_missing_samples.py` | 附录 A |
| ZS-CSD 转换 | `data_prep/zscsd/convert_zscsd.py`（路径式实例 → 会话树）、`zscsd_stats.py` | §V-A |
| MT-CSD 转换 | `data_prep/mtcsd/convert_mtcsd_to_stance_instances.py`、`build_target_aware_windows_en.py` | 附录 MT-CSD |
| 语义标注 | `annotation/build_semantic_propagation_annotations_llm.py --context_mode causal`（英文用 `_en`） | §IV-B |
| 标注泄漏修复 | `annotation/fix_child_leakage.py`（重标注曾看到子回复的样本）、`check_annotation_context_mode.py` | 附录 A |
| 标注成本 | `annotation/cost_report.py` | 附录（Cost） |
| LLM 基础预测 | `base_models/llm/simple_rag_stance_predictor.py --method knn_fewshot / cot / full_path` | §V-B，表 II、IV |
| GLAN | `base_models/glan/run_glan_cmtcsd.py` | 表 IV |
| SITPCL | `base_models/sitpcl/run_sitpcl.py` → `map_sitpcl_preds.py` | 表 IV |
| Qwen SFT | `base_models/qwen_sft/qwen_sft_zscsd.py` → `map_qwen_preds.py`；低资源：`make_subset_instances.py`、`run_a1.sh` | 表 IV、附录 |
| **SPIN** | `spin_core/analyze_combined_prior.py` | 表 II–IV |
| 主表数字与显著性 | `experiments/main/collect_paper_numbers.py`、`paired_significance.py`、`summarize_methods.py` | 表 II、III |
| 数据集统计 | `experiments/main/dataset_stats.py` | 表 I |
| 为什么不放进 prompt | `experiments/ablation/motivation_stats.py`；prompt 变体用 `simple_rag_stance_predictor.py --method dynamic_window / dynamic_semantic / dynamic_rag` | §VI-A |
| 消融（检索键、gold 父立场上界、随机键） | `experiments/ablation/analyze_combined_ablation.py` | §VI-B、§VI-C |
| 检索键显著性 | `experiments/ablation/analyze_retrieval_significance.py` | §VI-B |
| 学习式融合对照 | `experiments/ablation/learned_correction.py` | 附录（Learned combinations） |
| 误差分析 | `experiments/analysis/error_analysis.py`、`error_correlation.py` | §VI-E |
| 迁移诊断 | `experiments/analysis/diagnose_transfer.py`、`analyze_combined_prior_cv.py` | 附录（Supervised Models） |
| 分组 / 深度 | `experiments/analysis/zscsd_breakdown.py`、`depth_breakdown.py` | 附录 |
| kNN 重复检查 | `experiments/analysis/check_knn_overlap.py` | 附录 |
| 负面结果 | `experiments/negative/`：跨话题少样本校准、自适应 λ、复核选点、LLM 复核、说话人先验、相对 vs 绝对记忆 | 附录（Negative Results） |
| 审稿回应 | `experiments/robustness/`（`run_robustness.sh` 一键运行） | §VI-D、附录 F |
| 论文图 | `figures/make_figures.py`、`more_figures.py`、`kb_heatmap.py`、`experiments/robustness/plot_robustness.py` | 图 3–7 |

## 数据协议

- 知识库只用 `split == "train"` 的标注节点构建。
- E 与 λ 只在 valid 上估计和选择。
- 测试集标签只用于最终评测，以及明确标注为"诊断 / oracle"的分析。
