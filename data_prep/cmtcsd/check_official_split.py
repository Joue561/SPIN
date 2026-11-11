# -*- coding: utf-8 -*-
"""
check_official_split.py
核对：实验用的实例文件与 C-MTCSD 官方划分（每个话题目录下的 train/valid/test.json）是否一致，
以及与原论文的评测口径（每个话题单独算 F_avg 再平均）能否直接对比。

检查内容：
  1. 官方 JSON 中各话题、各划分的样本数（区分 原帖 / 评论）
  2. 实例文件中各话题、各划分的样本数
  3. 官方样本没有进入实例文件的原因（原帖被排除 / CSV 中找不到 / 无标签 / 其他）
  4. 实例文件中官方 JSON 里没有的样本（split=unknown 等）
  5. 划分是否一致（同一 node_id 在两边是否同一划分）
  6. 标签是否一致（若官方 JSON 自带标签字段，与实例文件的 stance_label 比较）
  7. 官方 JSON 的字段名（第一条记录），便于确认格式

用法（make_rag_data 目录下；--raw_dir 为当初传给 convert_cmtscd_to_stance_instances.py 的 --input_dir）：
python spin/data_prep/cmtcsd/check_official_split.py \
  --raw_dir ./C-MTCSD-main \
  --instances processed_cmtscd_semantic_llm_causal_v2/cmtscd_semantic_instances_llm.jsonl
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

TOPICS = ["Apollo Go", "Naked Resignation", "Non-marriage Doctrine", "Pre-made Meals", "iPhone 15"]
SPLITS = ["train", "valid", "test"]
LABEL_KEYS = ("label", "stance", "stance_label", "labels")
ALIAS = {"favor": "favor", "favour": "favor", "support": "favor", "支持": "favor", "1": "favor",
         "against": "against", "oppose": "against", "反对": "against", "0": "against",
         "none": "none", "neutral": "none", "中立": "none", "2": "none"}


def read_json_any(p):
    if not p.exists():
        return []
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    try:
        d = json.loads(txt)
        return d if isinstance(d, list) else [d]
    except json.JSONDecodeError:
        out = []
        for line in txt.splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out


def rec_id(rec):
    idx = rec.get("index") or rec.get("path") or rec.get("ids") or []
    if isinstance(idx, list) and idx:
        return str(idx[-1]).strip()
    if isinstance(idx, str):
        return idx.strip()
    return str(rec.get("id", "")).strip()


def rec_label(rec):
    for k in LABEL_KEYS:
        v = rec.get(k)
        if isinstance(v, list) and v:
            v = v[-1]
        if v is not None and str(v).strip().lower() in ALIAS:
            return ALIAS[str(v).strip().lower()], k
    return None, None


def find_dirs(raw):
    topic_dirs, csvs = {}, {}
    for root in (raw / "data" / "data", raw / "data", raw):
        if root.exists():
            for d in root.iterdir():
                if d.is_dir() and d.name in TOPICS:
                    topic_dirs.setdefault(d.name, d)
            for c in root.glob("*.csv"):
                csvs.setdefault(c.stem, c)
    return topic_dirs, csvs


def load_csv(path):
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            with open(path, encoding=enc) as f:
                return {str(r.get("id", "")).strip(): r for r in csv.DictReader(f)}
        except Exception:
            continue
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--instances", required=True)
    a = ap.parse_args()
    raw = Path(a.raw_dir)
    topic_dirs, csvs = find_dirs(raw)
    print(f"找到话题目录：{sorted(topic_dirs)}")
    print(f"找到 CSV：{sorted(csvs)}")
    if not topic_dirs:
        print("⚠️ 没找到话题目录，请确认 --raw_dir 与当初转换时的 --input_dir 相同")
        return

    # ---- 官方划分
    off = {}                      # (topic, nid) -> (split, label)
    first = None
    dup = 0
    for t in TOPICS:
        d = topic_dirs.get(t)
        if not d:
            continue
        for sp in SPLITS:
            for rec in read_json_any(d / f"{sp}.json"):
                if first is None:
                    first = (t, sp, rec)
                nid = rec_id(rec)
                if not nid:
                    continue
                lab, _ = rec_label(rec)
                if (t, nid) in off and off[(t, nid)][0] != sp:
                    dup += 1
                off[(t, nid)] = (sp, lab)
    if first:
        t, sp, rec = first
        print(f"\n官方 JSON 第一条记录（{t}/{sp}.json）的字段：{sorted(rec.keys())}")
        print("  示例：" + json.dumps({k: (v if len(str(v)) < 80 else str(v)[:80] + '…') for k, v in rec.items()}, ensure_ascii=False))
    has_label = sum(1 for v in off.values() if v[1]) / max(len(off), 1)
    print(f"官方样本 {len(off)} 条；跨划分重复 {dup} 条；官方 JSON 自带可识别标签的比例 {has_label:.1%}")

    # ---- 实例文件
    ins = {}
    for line in open(a.instances, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            ins[(str(r.get("topic")), str(r.get("node_id")))] = r
    print(f"实例文件 {len(ins)} 条")

    # ---- 1/2 计数
    print("\n【各话题、各划分样本数】官方（原帖 + 评论） vs 实例文件")
    print(f"  {'话题':24s} " + " ".join(f"{sp + '官方':>10s} {sp + '原帖':>8s} {sp + '实例':>9s}" for sp in SPLITS))
    tot = Counter()
    for t in TOPICS:
        row = []
        for sp in SPLITS:
            ks = [k for k, v in off.items() if k[0] == t and v[0] == sp]
            posts = sum(1 for k in ks if "-" not in k[1])
            mine = sum(1 for k, r in ins.items() if k[0] == t and r.get("split") == sp)
            tot[(sp, "off")] += len(ks); tot[(sp, "post")] += posts; tot[(sp, "ins")] += mine
            row.append(f"{len(ks):10d} {posts:8d} {mine:9d}")
        print(f"  {t:24s} " + " ".join(row))
    print(f"  {'合计':24s} " + " ".join(f"{tot[(sp, 'off')]:10d} {tot[(sp, 'post')]:8d} {tot[(sp, 'ins')]:9d}" for sp in SPLITS))
    n_off = sum(tot[(sp, 'off')] for sp in SPLITS)
    if n_off:
        print("  官方划分比例：" + " / ".join(f"{sp} {tot[(sp, 'off')] / n_off:.1%}" for sp in SPLITS))

    # ---- 3 官方样本缺失原因
    csv_maps = {t: load_csv(csvs[t]) for t in TOPICS if t in csvs}
    miss = defaultdict(Counter)
    for k, (sp, lab) in off.items():
        if k in ins:
            continue
        t, nid = k
        if "-" not in nid:
            reason = "原帖（转换时只保留评论）"
        elif t in csv_maps and nid not in csv_maps[t]:
            reason = "CSV 中找不到该 id"
        elif t in csv_maps and str(csv_maps[t][nid].get("stance", "")).strip().lower() not in ALIAS:
            reason = "CSV 中无有效标签"
        else:
            reason = "其他"
        miss[sp][reason] += 1
    print("\n【官方样本未进入实例文件的原因】")
    for sp in SPLITS:
        print(f"  {sp}: " + ("  ".join(f"{r} {c}" for r, c in miss[sp].most_common()) or "无"))

    # ---- 4/5 实例文件中多出的样本、划分是否一致
    extra = Counter(); mism = Counter()
    for k, r in ins.items():
        if k not in off:
            extra[r.get("split")] += 1
        elif off[k][0] != r.get("split"):
            mism[(off[k][0], r.get("split"))] += 1
    print("\n【实例文件中官方 JSON 没有的样本】按实例文件的 split：" + (str(dict(extra)) if extra else "无"))
    print("【两边都有但划分不一致】" + (str({f"{a}→{b}": c for (a, b), c in mism.items()}) if mism else "无（划分完全一致）"))

    # ---- 6 标签一致性
    if has_label > 0.5:
        agree = Counter()
        for k, (sp, lab) in off.items():
            if k in ins and lab:
                agree[(sp, lab == str(ins[k].get("stance_label")).lower())] += 1
        print("\n【标签一致性】官方 JSON 标签 vs 实例文件 stance_label")
        for sp in SPLITS:
            y, n = agree[(sp, True)], agree[(sp, False)]
            print(f"  {sp}: 一致 {y}，不一致 {n}")
    else:
        print("\n官方 JSON 不含可识别的标签字段，标签来自 CSV（与原论文一致与否需看原论文代码）")

    # ---- 测试集原帖：原论文评测是否包含原帖
    test_posts = tot[("test", "post")]
    print(f"\n提示：官方 test 中原帖 {test_posts} 条，实例文件不含原帖。若原论文评测包含原帖，"
          f"对比时需在表注中说明，或把原帖补进评测。")


if __name__ == "__main__":
    main()