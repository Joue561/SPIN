# -*- coding: utf-8 -*-
"""
analyze_kb_confidence.py
离线检验（不调用 API）：会话树推断中，把 LLM 自报的置信度（预测文件里的 confidence 字段）用进观测模型，
即"按置信度分档的混淆矩阵"——没把握的预测更容易被知识库修正，有把握的预测更少被改动。

观测模型 P(预测 | 真实, 置信度档)：
  plain  ：不分档（与之前所有脚本相同）
  conf   ：按 --conf_bins 分档（默认 0.9：<0.9 一档，≥0.9 一档）

知识库（均为话题内，只用该话题 train 构建）：KB:传导类型+深度 / KB:完整组合 / KB:仅深度（对照）/ LR:表层特征（参照）

协议（观测模型与 λ 从哪里估计）：
  fold  ：按会话树 2 折，在 test 另一半上估计（与 analyze_propagation_kb.py 相同）
  loto  ：只在其他话题的 test 上估计（与 analyze_kb_fewshot.py 的严格协议相同）
  valid ：在【同一话题的 valid 集】上估计观测模型，并在 valid 上运行同样的会话树推断来选 λ（最干净；需 --valid_pred）
  valid_shared：观测模型同 valid；λ 各话题共用一个，在全部 valid 上按话题平均 MacroF1 选（方差更小）

用法：
python -u spin/spin_core/analyze_kb_confidence.py \
  --instances processed_cmtscd_semantic_llm_causal_v2/cmtscd_semantic_instances_llm.jsonl \
  --pred rag_prediction_outputs/predictions_dynamic_semantic_test.jsonl \
  [--valid_pred rag_prediction_outputs/predictions_dynamic_semantic_valid_probs_logprobs.jsonl] \
  --output_dir kb_confidence_analysis
"""
import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_s = importlib.util.spec_from_file_location("cal", Path(__file__).resolve().parent / "analyze_kb_calibration.py")
cal = importlib.util.module_from_spec(_s)
_s.loader.exec_module(cal)
kbm, dd = cal.kbm, cal.dd
LABELS, LID = dd.LABELS, dd.LID


def load_pred_conf(path, pred_field, conf_field):
    lab, conf = {}, {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        p = dd.norm(r.get(pred_field))
        if p is None:
            continue
        k = (str(r.get("topic")), str(r.get("node_id")))
        lab[k] = p
        try:
            conf[k] = float(r.get(conf_field))
        except (TypeError, ValueError):
            conf[k] = float("nan")
    return lab, conf


def make_binner(thresholds):
    def b(c):
        if thresholds is None or c != c:      # 不分档，或置信度缺失
            return 0
        return int(sum(c >= t for t in thresholds))
    return b


def confusion(nodes_list, base, conf, binner, nbins):
    """返回 {档: 3×3 矩阵 P(预测 | 真实)}；某档样本过少时与整体矩阵混合平滑。"""
    allm = np.ones((3, 3))
    per = {b: np.zeros((3, 3)) for b in range(nbins)}
    for n in nodes_list:
        k = (n["topic"], n["node_id"])
        y, p = LID[n["y"]], LID[base[k]]
        allm[y, p] += 1
        per[binner(conf.get(k, float("nan")))][y, p] += 1
    allm = allm / allm.sum(1, keepdims=True)
    out = {}
    for b, m in per.items():
        sm = m + 5 * allm                       # 以整体矩阵为先验平滑（相当于 5 个伪样本/行）
        out[b] = sm / sm.sum(1, keepdims=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--pred", required=True, help="test 上的 dynamic_semantic 预测（需含 confidence 字段）")
    ap.add_argument("--valid_pred", default=None, help="valid 上的 dynamic_semantic 预测（可选，用于 valid 协议）")
    ap.add_argument("--pred_field", default="predicted_stance")
    ap.add_argument("--conf_field", default="confidence")
    ap.add_argument("--conf_bins", default="0.9", help="置信度分档阈值，逗号分隔")
    ap.add_argument("--parent_field", default="parent_stance_label")
    ap.add_argument("--lambdas", default="0,0.25,0.5,0.75,1,1.5,2")
    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--emissions", default="plain", help="plain,conf（逗号分隔；conf 已证明无效，默认只跑 plain）")
    ap.add_argument("--modes", default="pred,noparent",
                    help="pred=相对知识+预测父立场；noparent=绝对知识（只用知识单元的立场分布，不看父立场）。"
                         "KB:仅深度 的 noparent 即'按话题、按深度校准'对照")
    ap.add_argument("--output_dir", default="kb_confidence_analysis")
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    lams = [float(x) for x in a.lambdas.split(",")]
    thr = [float(x) for x in a.conf_bins.split(",")]

    nodes, _ = dd.load_instances(a.instances, a.parent_field)
    topics = sorted({n["topic"] for n in nodes.values()})
    train = [n for n in nodes.values() if n["split"] == "train" and n["y"]]
    base, conf = load_pred_conf(a.pred, a.pred_field, a.conf_field)
    ev = [n for k, n in nodes.items() if k in base and n["split"] == "test" and n["y"]]
    vbase, vconf, vev = {}, {}, []
    if a.valid_pred:
        vbase, vconf = load_pred_conf(a.valid_pred, a.pred_field, a.conf_field)
        vev = [n for k, n in nodes.items() if k in vbase and n["split"] == "valid" and n["y"]]
    print(f"train {len(train)} | test {len(ev)} | valid {len(vev)}（{'未提供' if not a.valid_pred else '已提供'}）")

    # ---- 置信度诊断
    binner = make_binner(thr)
    nb = len(thr) + 1
    print(f"\n【置信度分档】阈值 {thr}")
    for name, E, B, C in (("test", ev, base, conf), ("valid", vev, vbase, vconf)):
        if not E:
            continue
        g = defaultdict(lambda: [0, 0])
        for n in E:
            k = (n["topic"], n["node_id"])
            b = binner(C.get(k, float("nan")))
            g[b][0] += 1; g[b][1] += B[k] == n["y"]
        print(f"  {name}: " + " | ".join(f"档{b}: n={g[b][0]} 准确率={g[b][1] / max(g[b][0], 1):.3f}" for b in sorted(g)))

    # ---- 模型（话题内）
    models = {
        "KB:传导类型+深度": {t: kbm.PatternKB([n for n in train if n["topic"] == t], kbm.k_sp_depth, a.min_count) for t in topics},
        "KB:完整组合": {t: kbm.PatternKB([n for n in train if n["topic"] == t], kbm.k_full, a.min_count) for t in topics},
        "KB:仅深度": {t: kbm.PatternKB([n for n in train if n["topic"] == t], kbm.k_depth, a.min_count) for t in topics},
        "LR:表层特征": {t: dd.Transition([n for n in train if n["topic"] == t]) for t in topics},
    }
    for ms in models.values():
        for t in topics:
            ms[t].precompute([n for n in ev + vev if n["topic"] == t])

    # ---- 观测模型
    def emissions(kind):
        bn = binner if kind == "conf" else make_binner(None)
        nbk = nb if kind == "conf" else 1
        fold = {k: confusion([n for n in ev if dd.fold_of(n) != k], base, conf, bn, nbk) for k in (0, 1)}
        loto = {t: confusion([n for n in ev if n["topic"] != t], base, conf, bn, nbk) for t in topics}
        val = {t: confusion([n for n in vev if n["topic"] == t], vbase, vconf, bn, nbk) for t in topics} if vev else None
        e_fold = lambda n: fold[dd.fold_of(n)][bn(conf.get((n["topic"], n["node_id"]), float("nan")))]
        e_loto = lambda n: loto[n["topic"]][bn(conf.get((n["topic"], n["node_id"]), float("nan")))]
        e_val_test = (lambda n: val[n["topic"]][bn(conf.get((n["topic"], n["node_id"]), float("nan")))]) if val else None
        e_val_valid = (lambda n: val[n["topic"]][bn(vconf.get((n["topic"], n["node_id"]), float("nan")))]) if val else None
        return e_fold, e_loto, e_val_test, e_val_valid

    rows = [dd.summarize(ev, base, base, {"model": "dynamic_semantic(原始)", "mode": "-", "emission": "-", "protocol": "-", "lambda": "-"})]
    mode_name = {"pred": "相对+预测父", "noparent": "绝对(不看父)"}
    for ekind in a.emissions.split(","):
        e_fold, e_loto, e_vt, e_vv = emissions(ekind)
        for mname, ms in models.items():
          for mode in a.modes.split(","):
            protos = [("fold", e_fold), ("loto", e_loto)] + ([("valid", e_vt), ("valid_shared", e_vt)] if e_vt else [])
            vgrid = None
            for proto, emis in protos:
                grid = {l: cal.smooth_e(ev, base, ms, l, emis, mode) for l in lams}
                if proto in ("valid", "valid_shared"):
                    if vgrid is None:
                        vgrid = {l: cal.smooth_e(vev, vbase, ms, l, e_vv, mode) for l in lams}

                    def vscore(l, tt):
                        vt = [n for n in vev if n["topic"] == tt]
                        return dd.scores([n["y"] for n in vt], [vgrid[l][(n["topic"], n["node_id"])] for n in vt])["macro_f1"]
                    lab, chosen = {}, {}
                    if proto == "valid_shared":       # 各话题共用一个 λ：在全部 valid 上按话题平均 MacroF1 选
                        shared = max(lams, key=lambda l: np.mean([vscore(l, tt) for tt in topics]))
                        chosen = {t: shared for t in topics}
                    else:                             # 每个话题在自己的 valid 上单独选
                        chosen = {t: max(lams, key=lambda l: vscore(l, t)) for t in topics}
                    for n in ev:
                        k = (n["topic"], n["node_id"])
                        lab[k] = grid[chosen[n["topic"]]][k]
                    desc = f"{chosen[topics[0]]:g}" if proto == "valid_shared" else "/".join(f"{chosen[t]:g}" for t in topics)
                else:
                    lab, desc = cal.select(ev, grid, lams, topics, "crossfit" if proto == "fold" else "loto")
                rows.append(dd.summarize(ev, lab, base, {"model": mname, "mode": mode_name[mode], "emission": ekind,
                                                         "protocol": proto, "lambda": desc}))
                safe = f"{mname}_{mode}_{ekind}_{proto}".replace(":", "_").replace("+", "_")
                with open(out / f"pred_{safe}.jsonl", "w", encoding="utf-8") as f:
                    for n in ev:
                        k = (n["topic"], n["node_id"])
                        f.write(json.dumps({"topic": n["topic"], "node_id": n["node_id"], "depth": n["depth"],
                                            "stance_label": n["y"], "predicted_stance": lab[k], "base_prediction": base[k],
                                            "confidence": conf.get(k)}, ensure_ascii=False) + "\n")
          print(f"  完成 {mname} / {ekind}")
    dd.write_csv(out / "kb_confidence_results.csv", rows)

    dcols = [c for c in rows[0] if c.startswith("depth") and c.endswith("_F_avg")]
    print(f"\n{'模型':18s} {'用法':10s} {'观测':6s} {'协议':12s} {'λ':>20s} {'话题平均MacroF1':>14s} {'话题平均F_avg':>12s} {'合并MacroF1':>10s} {'合并F_avg':>9s} "
          f"{'改对':>5s} {'改错':>5s} | " + " ".join(f"{c.replace('_F_avg', ''):>8s}" for c in dcols))
    for r in rows:
        print(f"{r['model'][:18]:18s} {r['mode']:10s} {r['emission']:6s} {r['protocol']:12s} {str(r['lambda'])[:20]:>20s} {r['话题平均MacroF1']:14.4f} "
              f"{r['话题平均F_avg']:12.4f} {r['合并MacroF1']:10.4f} {r['合并F_avg']:9.4f} {r['改对']:5d} {r['改错']:5d} | "
              + " ".join(f"{r.get(c, float('nan')):8.4f}" for c in dcols))
    print("  （fold/loto 的 λ 分别为两折 / 各话题；valid 协议的 λ 按话题列出；按深度的列为各深度合并口径 F_avg）")
    print(f"\n结果已保存到 {out.resolve()}")


if __name__ == "__main__":
    main()