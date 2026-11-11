#!/usr/bin/env python3
"""
check_rag_target_mismatch.py（修正版）

诊断假设：RAG 检索完全不看 main_target/topic，导致跨 target 的记忆被
误用到不相关的样本上，是 dynamic_rag/full_rag 表现变差的真正原因。

做法：找出"没用 RAG 时预测对了，加了 RAG 之后反而错了"的样本（回归案例），
看这些样本检索到的记忆（retrieved_memory_summaries 字段，simple_rag_stance_predictor_en.py
已经把 pattern_name / dominant_label / reliability 直接存进预测文件了，不用
再额外查知识库）是不是主要是窄类目（composite_pattern），并且 dominant_label
经常和 gold 相反。

用法：
python check_rag_target_mismatch.py \
  --baseline_predictions mtcsd_experiment/rag_prediction_outputs/predictions_dynamic_window_test.jsonl \
  --rag_predictions mtcsd_experiment/rag_prediction_outputs/predictions_full_rag_test.jsonl \
  --show_examples 10
"""

import json
import argparse
from pathlib import Path
from collections import Counter


def load_jsonl_by_id(path: Path) -> dict:
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[rec.get("node_id")] = rec
    return records


def main():
    parser = argparse.ArgumentParser(description="诊断RAG记忆是否存在跨target误用问题")
    parser.add_argument("--baseline_predictions", required=True,
                         help="不用RAG的方法预测文件，比如 dynamic_window")
    parser.add_argument("--rag_predictions", required=True,
                         help="用RAG的方法预测文件，比如 full_rag 或 dynamic_rag")
    parser.add_argument("--show_examples", type=int, default=10)
    args = parser.parse_args()

    print("加载数据...")
    baseline = load_jsonl_by_id(Path(args.baseline_predictions))
    rag = load_jsonl_by_id(Path(args.rag_predictions))

    print(f"baseline: {len(baseline)} 条  rag: {len(rag)} 条")

    common_ids = set(baseline.keys()) & set(rag.keys())
    print(f"共同样本：{len(common_ids)} 条\n")

    regressions = []
    improvements = []

    for nid in common_ids:
        b, r = baseline[nid], rag[nid]
        gold = b.get("stance_label")
        b_pred = b.get("predicted_stance")
        r_pred = r.get("predicted_stance")
        if not gold:
            continue
        if b_pred == gold and r_pred != gold:
            regressions.append((nid, gold, b_pred, r_pred, r))
        elif b_pred != gold and r_pred == gold:
            improvements.append((nid, gold, b_pred, r_pred, r))

    print(f"回归案例（baseline对、RAG方法错）：{len(regressions)} 条")
    print(f"修正案例（baseline错、RAG方法对）：{len(improvements)} 条")
    print(f"（如果回归 > 修正，说明这次加 RAG 整体是负收益，跟宏观指标一致）\n")

    def pattern_type_distribution(cases):
        type_counter = Counter()
        mismatch_counter = Counter()
        for nid, gold, b_pred, r_pred, r in cases:
            summaries = r.get("retrieved_memory_summaries", [])
            for s in summaries:
                pname = s.get("pattern_name", "")
                is_composite = "+" in str(pname)
                mtype = "composite_pattern" if is_composite else "broad_pattern"
                type_counter[mtype] += 1
                dom = s.get("dominant_label")
                if dom and dom != gold:
                    mismatch_counter[mtype] += 1
        total = sum(type_counter.values())
        return type_counter, mismatch_counter, total

    reg_types, reg_mismatch, reg_total = pattern_type_distribution(regressions)
    imp_types, imp_mismatch, imp_total = pattern_type_distribution(improvements)

    print(f"{'='*90}")
    print("检索到的记忆类型分布对比：回归案例 vs 修正案例")
    print(f"{'='*90}")
    print(f"{'类型':<20}{'回归案例次数':>14}{'回归中占比':>12}{'修正案例次数':>14}{'修正中占比':>12}")
    for mtype in ["composite_pattern", "broad_pattern"]:
        rc = reg_types.get(mtype, 0)
        ic = imp_types.get(mtype, 0)
        rp = round(rc / reg_total * 100, 1) if reg_total else 0
        ip = round(ic / imp_total * 100, 1) if imp_total else 0
        print(f"{mtype:<20}{rc:>14}{rp:>11}%{ic:>14}{ip:>11}%")

    print(f"\n检索到的记忆 dominant_label 与 gold 不一致的比例：")
    for mtype in ["composite_pattern", "broad_pattern"]:
        rc = reg_types.get(mtype, 0)
        rm = reg_mismatch.get(mtype, 0)
        ic = imp_types.get(mtype, 0)
        im = imp_mismatch.get(mtype, 0)
        rp = round(rm / rc * 100, 1) if rc else 0
        ip = round(im / ic * 100, 1) if ic else 0
        print(f"  {mtype:<20} 回归案例中不一致率={rp}%   修正案例中不一致率={ip}%")

    print(f"\n（如果 composite_pattern 在回归案例里占比明显高于修正案例，"
          f"或者回归案例里 dominant_label 不一致率明显更高，"
          f"就实锤支持'窄类目/跨target记忆误导模型'这个假设）\n")

    print(f"{'='*90}")
    print(f"具体回归案例抽样（前 {min(args.show_examples, len(regressions))} 条）")
    print(f"{'='*90}")
    for nid, gold, b_pred, r_pred, r in regressions[:args.show_examples]:
        main_target = r.get("main_target", "未知")
        current_text = str(r.get("current_text", ""))[:80]
        summaries = r.get("retrieved_memory_summaries", [])

        print(f"\n--- node_id: {nid}  main_target: {main_target} ---")
        print(f"gold={gold}  baseline预测={b_pred}(对)  RAG预测={r_pred}(错)")
        print(f"当前评论: {current_text}")
        print(f"检索到的记忆：")
        for s in summaries:
            pname = s.get("pattern_name", "")
            is_composite = "+" in str(pname)
            print(f"  [{'composite' if is_composite else 'broad':<9}] "
                  f"pattern={pname}  dominant_label={s.get('dominant_label')}  "
                  f"reliability={s.get('reliability')}")

    print(f"\n{'='*90}")
    print("怎么判断")
    print(f"{'='*90}")
    print("看上面'回归案例 vs 修正案例'的对比表：")
    print("- 如果 composite_pattern 占比在回归案例里明显更高 → 窄类目记忆确实更容易带偏")
    print("- 如果 dominant_label 不一致率在回归案例里明显更高 → 检索到的记忆本身就跟当前")
    print("  样本的真实立场对不上，大概率是因为记忆库没按 main_target 区分导致的跨target误用")
    print("- 如果两组差异不明显 → 说明问题不在记忆内容本身，可能要回头看 prompt 怎么使用")
    print("  这些记忆信息的措辞，或者干脆做一版'完全不用RAG记忆'的消融确认RAG这个组件本身")
    print("  在英文Reddit语料上是不是就是不适用，而不是实现细节的问题")


if __name__ == "__main__":
    main()
