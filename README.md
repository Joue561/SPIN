# SPIN: Semantic Propagation INference for Multi-Turn Conversational Stance Detection

SPIN does not train any models. It is designed to correct stance predictions made by arbitrary base models over conversation trees:

1. Use an LLM to annotate the semantic relation between each reply and its parent.
2. Estimate a semantic propagation knowledge base from the training data.
3. Treat base-model predictions as noisy observations and perform a single top-down inference pass over the conversation tree.

## Directory Structure

```text
spin/
├── data_prep/              # 1. Data preparation: raw data -> unified node format, conversation trees, and context windows
│   ├── cmtcsd/             # C-MTCSD (Chinese, in-target)
│   ├── zscsd/              # ZS-CSD (Chinese, zero-shot)
│   └── mtcsd/              # MT-CSD (English, appendix)
├── annotation/             # 2. Semantic propagation annotation: reply relation / propagation type / target relation (knowledge-base retrieval keys)
├── base_models/            # 3. Base models: generate per-instance validation/test predictions as SPIN inputs
│   ├── llm/                # Prompt-based LLMs (kNN few-shot / zero-shot CoT / full path)
│   ├── glan/               # GLAN (supervised SOTA on C-MTCSD)
│   ├── sitpcl/             # SITPCL (supervised SOTA on ZS-CSD)
│   └── qwen_sft/           # Qwen2.5-7B LoRA SFT (including 10/25/50% low-resource settings)
├── spin_core/              # 4. SPIN itself: knowledge-base construction + observation channel + conversation-tree inference (six interdependent files; keep them together)
├── experiments/            # 5. Experiments and analyses reported in the paper
│   ├── main/               # Main tables, significance tests, and dataset statistics
│   ├── ablation/            # Ablations, retrieval-key significance, motivation analysis, learned-fusion comparisons
│   ├── analysis/            # Error analysis, correction potential, transfer diagnostics, subgroup results
│   ├── negative/            # Negative results reported in the appendix
│   └── robustness/         # Observation-channel robustness and transition-matrix symmetry analyses
├── figures/                # Paper figures
├── configs/                # Experiment manifests (runs*.json)
├── tools/                  # Auxiliary tools (read-only exports of the SPIN implementation, for verification)
├── tests/                  # Synthetic data for smoke tests (not for paper results)
└── legacy/                 # Earlier approaches not used in the final paper; retained for reference
    ├── rag_memory/         # Structured pattern memory + RAG prompting (the approach preceding SPIN)
    ├── plm/                # PLM fine-tuning ablations (e.g., MacBERT)
    ├── cross_topic_loto/   # Leave-one-topic-out cross-topic RAG experiments (01–09)
    ├── human_audit/        # Human review and AI pre-annotation
    └── misc/
```

All scripts are run from the working directory containing the data (for example, `~/wlgcn/wlgcn/make_rag_data`). Code accesses data through paths; **data are not stored in the repository**.

## SPIN Core (`spin_core/`)

These six files accumulated incrementally through successive offline experiments. Most of their `main()` functions are exploratory; only the components listed below are used by SPIN itself. They import one another using paths within the same directory, so all six files must remain together.

| File | Components used by SPIN | Corresponding paper section |
|---|---|---|
| `analyze_depth_dynamics.py` | `load_instances` (nodes, parent nodes, `pa_gold`, semantic fields), `norm`, `scores`, `relation`, `depth_bucket` | Problem formulation, §IV-A |
| `analyze_propagation_kb.py` | `PatternKB` (relative knowledge R / N, absolute knowledge A, per-table backoff, 0.5 smoothing), retrieval key `k_full` | Knowledge base, §IV-C, Eq. (2) |
| `analyze_kb_confidence.py` | `confusion` (observation channel E), `load_pred_conf` | Observation model, §IV-D, Eq. (1) |
| `analyze_kb_calibration.py` | Imported by the file above (its `main()` implements EM baseline-rate calibration for the appendix's cross-topic knowledge-base experiment) | Appendix: Negative Results |
| `analyze_conflict_detection.py` | `KB_KEYS`, `scores_of` (its `main()` implements the point-selection experiment for the appendix's LLM review) | — |
| `analyze_combined_prior.py` | **Main entry point**: `forward` (the single forward inference pass in Eq. (3)); selects λ on validation data using a grid search | §IV-E, §IV-F; Tables II / III |

To run SPIN, explicitly specify the full λ grid used in the paper (0–6). The script default only goes up to 2:

```bash
python -u spin/spin_core/analyze_combined_prior.py \\
  --instances processed_cmtscd_semantic_llm_causal_v4/cmtscd_semantic_instances_llm.jsonl \\
  --valid_pred baseline_outputs/predictions_knn_fewshot_valid.jsonl \\
  --pred baseline_outputs/predictions_knn_fewshot_test.jsonl \\
  --lambdas 0,0.25,0.5,0.75,1,1.5,2,3,4,6 --output_dir combined_prior/c_gpt4omini_knn
```

## Workflow and Mapping to the Paper

| Step | Script | Corresponding paper section |
|---|---|---|
| C-MTCSD conversion | `data_prep/cmtcsd/convert_cmtscd_to_stance_instances.py` → `build_target_aware_windows.py` → `add_root_posts.py` (adds the original post as a depth-0 node) | §V-A, Appendix A |
| Split verification | `data_prep/cmtcsd/check_official_split.py`, `find_missing_samples.py` | Appendix A |
| ZS-CSD conversion | `data_prep/zscsd/convert_zscsd.py` (converts path-based instances into conversation trees), `zscsd_stats.py` | §V-A |
| MT-CSD conversion | `data_prep/mtcsd/convert_mtcsd_to_stance_instances.py`, `build_target_aware_windows_en.py` | MT-CSD appendix |
| Semantic annotation | `annotation/build_semantic_propagation_annotations_llm.py --context_mode causal` (use the `_en` variant for English) | §IV-B |
| Annotation leakage fix | `annotation/fix_child_leakage.py` (re-annotates samples whose original annotations had seen child replies), `check_annotation_context_mode.py` | Appendix A |
| Annotation cost | `annotation/cost_report.py` | Appendix: Cost |
| LLM base predictions | `base_models/llm/simple_rag_stance_predictor.py --method knn_fewshot / cot / full_path` | §V-B, Tables II and IV |
| GLAN | `base_models/glan/run_glan_cmtcsd.py` | Table IV |
| SITPCL | `base_models/sitpcl/run_sitpcl.py` → `map_sitpcl_preds.py` | Table IV |
| Qwen SFT | `base_models/qwen_sft/qwen_sft_zscsd.py` → `map_qwen_preds.py`; low-resource settings: `make_subset_instances.py`, `run_a1.sh` | Table IV, Appendix |
| **SPIN** | `spin_core/analyze_combined_prior.py` | Tables II–IV |
| Main-table numbers and significance tests | `experiments/main/collect_paper_numbers.py`, `paired_significance.py`, `summarize_methods.py` | Tables II and III |
| Dataset statistics | `experiments/main/dataset_stats.py` | Table I |
| Why knowledge is not injected into prompts | `experiments/ablation/motivation_stats.py`; prompt variants use `simple_rag_stance_predictor.py --method dynamic_window / dynamic_semantic / dynamic_rag` | §VI-A |
| Ablations (retrieval keys, gold-parent upper bound, random keys) | `experiments/ablation/analyze_combined_ablation.py` | §VI-B, §VI-C |
| Retrieval-key significance | `experiments/ablation/analyze_retrieval_significance.py` | §VI-B |
| Learned-fusion comparison | `experiments/ablation/learned_correction.py` | Appendix: Learned combinations |
| Error analysis | `experiments/analysis/error_analysis.py`, `error_correlation.py` | §VI-E |
| Transfer diagnostics | `experiments/analysis/diagnose_transfer.py`, `analyze_combined_prior_cv.py` | Appendix: Supervised Models |
| Subgroup / depth analyses | `experiments/analysis/zscsd_breakdown.py`, `depth_breakdown.py` | Appendix |
| kNN overlap check | `experiments/analysis/check_knn_overlap.py` | Appendix |
| Negative results | `experiments/negative/`: cross-topic few-shot calibration, adaptive λ, review-point selection, LLM review, speaker priors, relative vs. absolute memory | Appendix: Negative Results |
| Reviewer-response analyses | `experiments/robustness/` (`run_robustness.sh` runs the analyses end to end) | §VI-D, Appendix F |
| Paper figures | `figures/make_figures.py`, `more_figures.py`, `kb_heatmap.py`, `experiments/robustness/plot_robustness.py` | Figures 3–7 |

## Data Protocol

- The knowledge base is constructed using only annotated nodes with `split == "train"`.
- The observation channel E and λ are estimated or selected using validation data only.
- Test-set labels are used only for final evaluation and for analyses explicitly identified as diagnostic or oracle-based.
