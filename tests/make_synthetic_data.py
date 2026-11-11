# -*- coding: utf-8 -*-
"""
生成与 C-MTCSD 语义实例同结构的合成数据，仅用于冒烟测试整条流水线（不可用于论文结果）。
python tools/make_synthetic_data.py --output synthetic/cmtscd_semantic_instances_llm.jsonl --n_per_topic 200
"""
import argparse
import json
import random
from pathlib import Path

TOPICS = {
    "Apollo Go": ("萝卜快跑", ["萝卜快跑", "无人驾驶", "百度"]),
    "Naked Resignation": ("裸辞", ["裸辞", "辞职"]),
    "Non-marriage Doctrine": ("不婚主义", ["不婚", "结婚"]),
    "Pre-made Meals": ("预制菜", ["预制菜", "料理包"]),
    "iPhone 15": ("iPhone 15", ["iPhone 15", "苹果"]),
}

# pattern: (reply_relation, propagation, target_relation, mapping, label_probs[favor,against,none], text templates)
PATTERNS = {
    "反讽": ("mock_parent", "sarcastic_reversal", "direct", "reverse", [0.1, 0.8, 0.1],
             ["呵呵，{t}真是太棒了呢", "对对对，{t}最好了，谁信谁傻"]),
    "短文本附和": ("agree_with_parent", "stance_inheritance", "via_parent", "same", None,
               ["确实", "666", "说得对", "+1"]),
    "反问": ("disagree_with_parent", "stance_reversal", "direct", "reverse", [0.15, 0.7, 0.15],
           ["{t}有什么好的？", "你确定{t}靠谱吗？"]),
    "直接支持": ("independent", "explicit_stance", "direct", "independent", [0.85, 0.05, 0.1],
             ["我支持{t}，挺好的", "{t}确实方便，价格也合适"]),
    "直接反对": ("independent", "explicit_stance", "direct", "independent", [0.05, 0.85, 0.1],
             ["{t}就是垃圾", "坚决反对{t}，安全没保障"]),
    "话题转移": ("topic_shift_reply", "topic_drift", "topic_shift", "none", [0.08, 0.07, 0.85],
             ["今天天气不错", "楼上头像好看"]),
    "攻击回复者": ("attack_replier", "no_propagation", "reply_author_only", "none", [0.1, 0.1, 0.8],
              ["你懂个屁", "楼上是水军吧"]),
    "让步转折": ("partial_agree", "concession", "direct", "partial", "TOPIC_DEPENDENT",
             ["虽然{t}有问题，但总体还行", "{t}是不错，可是太贵了"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_per_topic", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for ti, (topic, (target, terms)) in enumerate(TOPICS.items()):
        for i in range(a.n_per_topic):
            pat = rng.choice(list(PATTERNS))
            rr, prop, trel, mp, probs, temps = PATTERNS[pat]
            parent_label = rng.choice(["favor", "against"])
            if probs is None:
                probs = [0.8, 0.1, 0.1] if parent_label == "favor" else [0.1, 0.8, 0.1]
            elif probs == "TOPIC_DEPENDENT":
                probs = [0.75, 0.15, 0.1] if ti % 2 == 0 else [0.15, 0.75, 0.1]
            label = rng.choices(["favor", "against", "none"], probs)[0]
            t = rng.choice(terms)
            cur = rng.choice(temps).format(t=t)
            parent = f"我觉得{t}{'挺好' if parent_label == 'favor' else '不行'}"
            depth = rng.choice([1, 2, 2, 3, 4])
            split = rng.choices(["train", "dev", "test"], [0.7, 0.1, 0.2])[0]
            rec = {
                "dataset": "C-MTCSD-synthetic", "topic": topic, "main_target": target, "split": split,
                "node_id": f"n{i}", "root_id": f"r{i // 5}", "parent_id": f"p{i}", "depth": depth,
                "root_text": f"大家怎么看{t}？", "parent_text": parent, "current_text": cur,
                "target_anchor": t,
                "causal_context_text": f"[根] 大家怎么看{t}？\n[父] {parent}\n[当前] {cur}",
                "posthoc_context_text": f"[根] 大家怎么看{t}？\n[父] {parent}\n[当前] {cur}\n[子] 同意",
                "stance_label": label, "is_ambiguous_current": len(cur) <= 4,
                "llm_reply_relation": rr, "llm_semantic_propagation": prop,
                "llm_stance_mapping_to_parent": mp, "llm_target_relation": trel,
                "llm_candidate_patterns": pat if rng.random() < 0.5 else [pat],
                "llm_confidence": round(rng.uniform(0.5, 0.98), 2),
                "llm_reason_short": "合成数据",
                "llm_inferred_stance_to_main_target": label,
            }
            if rng.random() < 0.04:  # 模拟缺失语义字段
                for f in ["llm_reply_relation", "llm_semantic_propagation", "llm_target_relation"]:
                    rec.pop(f, None)
            lines.append(json.dumps(rec, ensure_ascii=False))
    lines.insert(5, "{this is a broken line")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} lines to {out}")


if __name__ == "__main__":
    main()
