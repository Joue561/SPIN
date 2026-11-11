# -*- coding: utf-8 -*-
"""
inspect_spin_code.py  —— 第一步：只读检查现有 SPIN 实现（不运行任何实验、不修改任何文件）

把以下内容的源码和所在文件原样导出到一个文本文件，便于核对：
  - analyze_combined_prior.py 中的 forward / topic_avg / 网格选择相关函数
  - 观测信道 E 的估计函数 kc.confusion、预测读取 kc.load_pred_conf
  - 知识库 kbm.PatternKB（相对知识矩阵 M、绝对知识 P_abs、平滑与 back-off）
  - 检索键定义 cd.KB_KEYS、数据读取 dd.load_instances
并列出现有的预测缓存文件（*_valid.jsonl / *_test.jsonl）与其中的字段。

用法（放在 cross_topic_exp/tools/ 下，与 analyze_combined_prior.py 同目录）：
  python -u cross_topic_exp/tools/inspect_spin_code.py --root . --out spin_code_inspection.txt
"""
import argparse
import importlib.util
import inspect
import json
from pathlib import Path


def load_acp(tools_dir):
    p = Path(tools_dir) / "analyze_combined_prior.py"
    s = importlib.util.spec_from_file_location("acp", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def src(obj):
    try:
        return f"# file: {inspect.getsourcefile(obj)}\n" + inspect.getsource(obj)
    except Exception as e:  # noqa
        return f"# <无法获取源码: {e}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools_dir", default=str((Path(__file__).resolve().parent / "../spin_core").resolve()))
    ap.add_argument("--root", default=".", help="搜索预测缓存的根目录")
    ap.add_argument("--out", default="spin_code_inspection.txt")
    ap.add_argument("--code_only", action="store_true", help="只导出源码，不列缓存文件")
    a = ap.parse_args()
    acp = load_acp(a.tools_dir)
    out = []
    w = out.append
    w("=" * 80 + "\n模块位置\n" + "=" * 80)
    for name in ("cd", "kc", "kbm", "dd"):
        mod = getattr(acp, name, None)
        w(f"acp.{name} -> {getattr(mod, '__file__', mod)}")
    w(f"LABELS = {getattr(acp, 'LABELS', None)}   LID = {getattr(acp, 'LID', None)}")
    try:
        w(f"cd.KB_KEYS = {json.dumps(acp.cd.KB_KEYS, ensure_ascii=False, default=str)}")
    except Exception as e:  # noqa
        w(f"cd.KB_KEYS: <{e}>")

    targets = [
        ("acp.forward", getattr(acp, "forward", None)),
        ("acp.topic_avg", getattr(acp, "topic_avg", None)),
        ("acp.main", getattr(acp, "main", None)),
        ("kc.confusion", getattr(acp.kc, "confusion", None)),
        ("kc.load_pred_conf", getattr(acp.kc, "load_pred_conf", None)),
        ("kc.make_binner", getattr(acp.kc, "make_binner", None)),
        ("kbm.PatternKB", getattr(acp.kbm, "PatternKB", None)),
        ("dd.load_instances", getattr(acp.dd, "load_instances", None)),
        ("cd.scores_of", getattr(acp.cd, "scores_of", None)),
    ]
    for name, obj in targets:
        w("\n" + "=" * 80 + f"\n{name}\n" + "=" * 80)
        w(src(obj) if obj is not None else "<不存在>")

    # PatternKB 用 spec 动态加载后 inspect 取不到类源码 → 直接导出整个模块文件；KB_KEYS 中的键函数单独导出
    for name in ("kbm",):
        f = getattr(getattr(acp, name, None), "__file__", None)
        w("\n" + "=" * 80 + f"\n完整文件 acp.{name}: {f}\n" + "=" * 80)
        w(Path(f).read_text(encoding="utf-8") if f and Path(f).exists() else "<找不到文件>")
    try:
        for kname, fn in acp.cd.KB_KEYS.items():
            w("\n" + "=" * 80 + f"\ncd.KB_KEYS[{kname!r}]\n" + "=" * 80)
            w(src(fn))
    except Exception as e:  # noqa
        w(f"KB_KEYS 源码: <{e}>")
    for fname in ("norm", "nv", "scores"):
        fn = getattr(acp.dd, fname, None)
        if fn is not None:
            w("\n" + "=" * 80 + f"\ndd.{fname}\n" + "=" * 80)
            w(src(fn))
    if getattr(__import__("sys"), "argv", None) and "--code_only" in __import__("sys").argv:
        Path(a.out).write_text("\n".join(out), encoding="utf-8")
        print(f"已写出 {a.out}（仅源码）。请把这个文件发给我核对。")
        return

    w("\n" + "=" * 80 + "\n预测缓存文件（valid/test）\n" + "=" * 80)
    root = Path(a.root)
    files = sorted(set(root.rglob("predictions*_valid*.jsonl")) | set(root.rglob("predictions*_test*.jsonl"))
                   | set(root.rglob("*pred*valid*.jsonl")) | set(root.rglob("*pred*test*.jsonl")))
    for f in files[:400]:
        try:
            with open(f, encoding="utf-8") as fh:
                first = json.loads(fh.readline())
            n = sum(1 for _ in open(f, encoding="utf-8"))
            w(f"{f}  ({n} 行)  字段: {sorted(first.keys())}")
        except Exception as e:  # noqa
            w(f"{f}  <读取失败 {e}>")
    w("\n实例文件：")
    for f in sorted(root.rglob("*semantic_instances*.jsonl"))[:200]:
        with open(f, encoding="utf-8") as fh:
            first = json.loads(fh.readline())
        w(f"{f}  字段: {sorted(first.keys())}")
    Path(a.out).write_text("\n".join(out), encoding="utf-8")
    print(f"已写出 {a.out}（{len(files)} 个预测文件）。请把这个文件发给我核对。")


if __name__ == "__main__":
    main()