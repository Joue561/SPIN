#!/usr/bin/env python3
"""
check_token_truncation.py

验证问题①：plm_semantic_rag_text（语义+RAG文本叠加）的 input_text 是否在
max_length=512 处被大量截断，导致"叠加"效果反而变差。

同时会顺带曝光一个真实存在的问题：prepare_plm_stance_data.py 的
build_input_text() 里拼接标签是硬编码中文（"目标对象：""会话上下文："等），
在 RoBERTa 的 byte-level BPE 下，中文字符通常会被切成不止一个 token，
比按字符数估算的 MAX_CONTEXT_CHARS/MAX_MEMORY_SUMMARY_CHARS 预算膨胀得多，
这也是导致截断比预期严重的一个原因。本脚本会把"中文标签占用的 token 数占比"
单独统计出来，方便你判断这部分影响有多大。

用法（对比多个 method 一起看更清楚）：
python check_token_truncation.py \
  --data_dirs mtcsd_experiment/plm_experiment_data/plm_current \
              mtcsd_experiment/plm_experiment_data/plm_semantic_text \
              mtcsd_experiment/plm_experiment_data/plm_rag_text \
              mtcsd_experiment/plm_experiment_data/plm_semantic_rag_text \
  --base_model roberta-base \
  --max_length 512 \
  --split train
"""

import json
import argparse
import re
from pathlib import Path
from collections import Counter

try:
    from transformers import AutoTokenizer
except ImportError:
    raise SystemExit("请先 pip install transformers --break-system-packages")


def load_jsonl(path: Path) -> list:
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def percentile(sorted_vals: list, p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def analyze_one_method(data_dir: Path, tokenizer, max_length: int, split: str) -> dict:
    records = load_jsonl(data_dir / f"{split}.jsonl")
    if not records:
        return {"method": data_dir.name, "error": f"{split}.jsonl 不存在或为空"}

    lengths = []
    truncated_count = 0
    current_text_lost_count = 0  # 截断后 current_text 完全找不到踪迹的样本数
    cjk_token_counts = []

    for rec in records:
        input_text = rec.get("input_text", "")
        current_text = rec.get("current_text", "")

        # 完整 token 数（不截断），用于判断真实长度
        full_ids = tokenizer.encode(input_text, add_special_tokens=True)
        length = len(full_ids)
        lengths.append(length)

        is_truncated = length > max_length
        if is_truncated:
            truncated_count += 1
            # 截断后实际喂给模型的 token
            truncated_ids = full_ids[:max_length]
            decoded_kept = tokenizer.decode(truncated_ids, skip_special_tokens=True)

            # current_text 是否还完整/部分保留在截断后的文本里
            # 用 current_text 的前 15 个字符做一个宽松匹配（考虑 tokenizer 可能引入空格差异）
            probe = current_text.strip()[:15]
            if probe and probe not in decoded_kept:
                current_text_lost_count += 1

        # 统计中文字符本身在这条 input_text 里消耗了多少 token
        # （粗略方法：把所有中文字符抽出来单独编码，估算其 token 占比）
        cjk_chars = "".join(CJK_PATTERN.findall(input_text))
        if cjk_chars:
            cjk_ids = tokenizer.encode(cjk_chars, add_special_tokens=False)
            cjk_token_counts.append(len(cjk_ids))

    lengths_sorted = sorted(lengths)
    n = len(lengths)

    result = {
        "method": data_dir.name,
        "split": split,
        "sample_count": n,
        "token_length_mean": round(sum(lengths) / n, 1),
        "token_length_median": round(percentile(lengths_sorted, 0.5), 1),
        "token_length_p90": round(percentile(lengths_sorted, 0.90), 1),
        "token_length_p95": round(percentile(lengths_sorted, 0.95), 1),
        "token_length_max": max(lengths),
        "truncated_count": truncated_count,
        "truncated_ratio": round(truncated_count / n, 4),
        "current_text_lost_after_truncation_count": current_text_lost_count,
        "current_text_lost_ratio_among_truncated": (
            round(current_text_lost_count / truncated_count, 4) if truncated_count else 0.0
        ),
        "avg_cjk_label_tokens_per_sample": (
            round(sum(cjk_token_counts) / len(cjk_token_counts), 1) if cjk_token_counts else 0.0
        ),
        "samples_with_cjk_content": len(cjk_token_counts),
    }
    return result


def print_report(results: list, max_length: int):
    print(f"\n{'方法':<28}{'样本数':>8}{'均值':>8}{'中位数':>8}{'P90':>8}{'P95':>8}{'最大':>8}{'截断率':>9}{'截断后丢失current_text':>14}")
    print("-" * 120)
    for r in results:
        if "error" in r:
            print(f"{r['method']:<28}  ⚠️ {r['error']}")
            continue
        print(
            f"{r['method']:<28}"
            f"{r['sample_count']:>8}"
            f"{r['token_length_mean']:>8}"
            f"{r['token_length_median']:>8}"
            f"{r['token_length_p90']:>8}"
            f"{r['token_length_p95']:>8}"
            f"{r['token_length_max']:>8}"
            f"{r['truncated_ratio']*100:>8.1f}%"
            f"{r['current_text_lost_ratio_among_truncated']*100:>13.1f}%"
        )

    print(f"\n（max_length={max_length}；'截断后丢失current_text' 是指截断后的文本里已经找不到"
          f"current_text 开头 15 个字符，说明模型这条样本训练时看不到当前评论本身，label 和输入严重不匹配）")

    print("\n中文标签 token 开销（如果 > 0，说明 build_input_text 里的中文模板正在消耗 RoBERTa token 预算）：")
    for r in results:
        if "error" in r:
            continue
        if r["avg_cjk_label_tokens_per_sample"] > 0:
            print(f"  {r['method']}: 平均每条样本中文标签占用 ~{r['avg_cjk_label_tokens_per_sample']} 个 token"
                  f"（{r['samples_with_cjk_content']}/{r['sample_count']} 条样本含中文字符）")


def main():
    parser = argparse.ArgumentParser(description="检查 PLM 输入 token 长度分布与截断情况")
    parser.add_argument("--data_dirs", nargs="+", required=True,
                         help="一个或多个 prepare_plm_stance_data.py 输出目录（每个目录对应一个 method）")
    parser.add_argument("--base_model", default="roberta-base",
                         help="tokenizer 来源，可以是 huggingface 模型名，也可以是本地目录路径")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--output_json", default=None, help="可选：把结果另存为 json")
    args = parser.parse_args()

    print(f"加载 tokenizer：{args.base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    results = []
    for d in args.data_dirs:
        data_dir = Path(d)
        print(f"分析 {data_dir} ...")
        r = analyze_one_method(data_dir, tokenizer, args.max_length, args.split)
        results.append(r)

    print_report(results, args.max_length)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到 {args.output_json}")


if __name__ == "__main__":
    main()
