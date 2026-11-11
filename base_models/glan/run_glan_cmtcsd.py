# -*- coding: utf-8 -*-
"""
run_glan_cmtcsd.py
在 C-MTCSD 上复现 GLAN（Niu et al., LREC-COLING 2024，MT-CSD 仓库 code/models/GLAN.py），
导出逐条预测，供知识库会话树推断叠加使用。

不修改原仓库：直接导入 datasets/MT-CSD/code/models/GLAN.py 的 Model 和 code/util.py 的 build_dataset。
训练流程按原 train.py：每个话题单独训练；Adam(lr=1e-5, weight_decay=1e-4)，batch 32，max_seq_len 512，
每 20 个 batch 在 dev 上按 F_avg（favor 与 against 的 F1 平均）评测，刷新最佳则保存；
连续 3 个 epoch 无提升早停，最多 100 个 epoch；训练集不打乱（与原代码一致）。

为适配中文 C-MTCSD 并保证评测干净，做了以下改动（论文中需说明）：
  1. 选模只看 dev：最终用 dev 最佳的模型输出 valid / test 预测。
     原代码保存预测时以 test 分数是否更高为准，等于在 test 上挑选，本脚本不这样做。
  2. 截断：原代码按空格计词保留最后 450 词，中文无空格时失效，tokenizer 会从右侧截掉当前评论；
     改为从左侧截断（truncation_side='left'），保留当前评论与最近的上文。
  3. 句子数：模型的邻接矩阵固定为 8×8，只保留最近的 8 句（当前评论 + 7 个最近祖先，含原帖时原帖算一句）。
  4. 目标文本用中文目标词；预训练模型默认 chinese-roberta-wwm-ext（BERT 结构，可用 BertModel 加载）。
  5. 不使用 wandb。

输出（--out_dir/seed<seed>/）：
  <话题>/preds_valid.jsonl, preds_test.jsonl, metrics.json, train.log
  predictions_glan_valid.jsonl / predictions_glan_test.jsonl  五个话题合并，字段
      {topic, node_id, predicted_stance, confidence, probs, stance_label}，可直接给 analyze_combined_prior.py
  summary.json  各话题与话题平均的 F_avg / Macro-F1

用法（make_rag_data 目录下，用 2 号卡）：
python -u cmtcsd_sota/run_glan_cmtcsd.py --glan_code datasets/MT-CSD/code --data_dir C-MTCSD/data \
  --bert_dir ZS-CSD/plm/chinese-roberta-wwm-ext --out_dir cmtcsd_sota/outputs/glan --cuda_index 2 --seed 0
调试：加 --debug（每个划分只取 64 条，1 个 epoch）；只跑部分话题：--topics "Apollo Go,iPhone 15"
只合并已有话题结果：--merge_only
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

TOPICS = ["Apollo Go", "Naked Resignation", "Non-marriage Doctrine", "Pre-made Meals", "iPhone 15"]
TARGET_ZH = {"Apollo Go": "萝卜快跑", "Naked Resignation": "裸辞", "Non-marriage Doctrine": "不婚主义",
             "Pre-made Meals": "预制菜", "iPhone 15": "iPhone 15"}
LABEL_MAP = {"none": 0, "favor": 1, "against": 2}           # 与原 util.build_dataset 一致
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
DEFAULTS = dict(model_name="GLAN", sub_model="Model", lr=1e-5, dropout=0.2, max_seq_len=512, batch_size=32,
                hop=3, lambdaa=0.2, bert_dim=768, polarities_dim=3, hidden_dim=768, num_epoch=100)


def f1_scores(y, p):
    from sklearn import metrics
    f_avg = metrics.f1_score(y, p, average="macro", labels=[1, 2])
    macro3 = metrics.f1_score(y, p, average="macro", labels=[0, 1, 2])
    return float(f_avg), float(macro3)


def load_split(data_dir, topic, split, max_utts):
    import csv
    texts = {}
    with open(Path(data_dir) / "data" / f"{topic}.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            texts[str(r["id"]).strip()] = str(r.get("text") or "")
    recs = json.load(open(Path(data_dir) / topic / f"{split}.json", encoding="utf-8"))
    convs, labels, ids, missing = [], [], [], 0
    for r in recs:
        idx = [str(x).strip() for x in r["index"]]
        lab = str(r["stance"]).strip().lower()
        if lab not in LABEL_MAP:
            continue
        utts = []
        for i in idx[-max_utts:]:
            if i not in texts:
                missing += 1
            utts.append(texts.get(i, "").lower().strip() or "（空）")
        convs.append(" [SEP] ".join(utts))
        labels.append(lab)
        ids.append(idx[-1])
    return {"data": convs, "label": labels, "targets": [TARGET_ZH[topic]] * len(convs)}, ids, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glan_code", required=True, help="MT-CSD 仓库的 code 目录（含 models/GLAN.py、util.py）")
    ap.add_argument("--data_dir", required=True, help="C-MTCSD/data（含 data/<话题>.csv 与 <话题>/{train,valid,test}.json）")
    ap.add_argument("--bert_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cuda_index", type=int, default=0)
    ap.add_argument("--topics", default=",".join(TOPICS))
    ap.add_argument("--max_utts", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--patience_epochs", type=int, default=3)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--merge_only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir).resolve() / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    topics = [t.strip() for t in a.topics.split(",") if t.strip()]

    if not a.merge_only:
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, BertModel, RobertaModel
        sys.path.insert(0, str(Path(a.glan_code).resolve()))
        from models import GLAN                          # 原仓库模型，未改动
        from util import build_dataset                   # 原仓库的分词与 token_type 构造，未改动
        from torch.utils.data import DataLoader

        dev = torch.device(f"cuda:{a.cuda_index}" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(a.bert_dir)
        tok.truncation_side = "left"                     # 改动 2：保留当前评论
        for topic in topics:
            random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed(a.seed)
            torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
            args = argparse.Namespace(**DEFAULTS, pretrained_bert_name=a.bert_dir, device=dev)
            tdir = out / topic.replace(" ", "_"); tdir.mkdir(parents=True, exist_ok=True)
            log = open(tdir / "train.log", "w", encoding="utf-8")

            def P(msg):
                print(msg, flush=True); log.write(msg + "\n"); log.flush()

            data, ids = {}, {}
            for sp in ("train", "valid", "test"):
                d, i, miss = load_split(a.data_dir, topic, sp, a.max_utts)
                if a.debug:
                    d = {k: v[:64] for k, v in d.items()}; i = i[:64]
                data[sp], ids[sp] = build_dataset(args, d, tok), i
                P(f"[{topic}] {sp}: {len(i)} 条" + (f"（CSV 中缺文本 {miss} 处）" if miss else ""))
            loaders = {sp: DataLoader(data[sp], batch_size=args.batch_size, shuffle=False) for sp in data}

            model = GLAN.Model(args).to(dev)
            for child in model.children():               # 与原 train.reset_params 相同
                if type(child) != BertModel and type(child) != RobertaModel:
                    for p in child.parameters():
                        if p.requires_grad:
                            if len(p.shape) > 1:
                                torch.nn.init.xavier_uniform_(p)
                            else:
                                stdv = 1. / math.sqrt(p.shape[0]); torch.nn.init.uniform_(p, a=-stdv, b=stdv)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

            def predict(sp):
                model.eval()
                probs, ys = [], []
                with torch.no_grad():
                    for batch in loaders[sp]:
                        inputs = [batch[c].to(dev) for c in batch]
                        probs.append(torch.softmax(model(inputs), -1).cpu().numpy())
                        ys.append(inputs[3].cpu().numpy())
                model.train()
                pr = np.concatenate(probs); y = np.concatenate(ys)
                return pr, y

            best, best_state, best_epoch, total = -1.0, None, 0, 0
            n_epoch = 1 if a.debug else args.num_epoch
            t0 = time.time()
            model.train()
            for epoch in range(n_epoch):
                for batch in loaders["train"]:
                    inputs = [batch[c].to(dev) for c in batch]
                    loss = F.cross_entropy(model(inputs), inputs[3])
                    model.zero_grad(); loss.backward(); opt.step()
                    if total % a.eval_every == 0:
                        pr, y = predict("valid")
                        fa, m3 = f1_scores(y, pr.argmax(-1))
                        mark = ""
                        if fa > best:
                            best, best_epoch, mark = fa, epoch, " *"
                            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                        P(f"[{topic}] epoch {epoch + 1} step {total}  loss {loss.item():.3f}  dev F_avg {100 * fa:.2f}  "
                          f"Macro-F1 {100 * m3:.2f}{mark}  ({time.time() - t0:.0f}s)")
                    total += 1
                if epoch - best_epoch >= a.patience_epochs:
                    P(f"[{topic}] 早停于 epoch {epoch + 1}（最佳在 epoch {best_epoch + 1}）"); break
            model.load_state_dict(best_state)
            res = {"topic": topic, "seed": a.seed, "best_epoch": best_epoch + 1}
            for sp in ("valid", "test"):
                pr, y = predict(sp)
                assert len(pr) == len(ids[sp])
                fa, m3 = f1_scores(y, pr.argmax(-1))
                res[sp] = {"f_avg": fa, "macro_f1": m3, "n": len(y)}
                with open(tdir / f"preds_{sp}.jsonl", "w", encoding="utf-8") as f:
                    for nid, p, t in zip(ids[sp], pr, y):
                        f.write(json.dumps({"topic": topic, "node_id": nid, "predicted_stance": ID2LABEL[int(p.argmax())],
                                            "confidence": float(p.max()), "probs": [round(float(x), 6) for x in p],
                                            "stance_label": ID2LABEL[int(t)]}, ensure_ascii=False) + "\n")
            json.dump(res, open(tdir / "metrics.json", "w"), ensure_ascii=False, indent=2)
            P(f"[{topic}] 完成：best epoch {res['best_epoch']}  valid F_avg {100 * res['valid']['f_avg']:.2f}  "
              f"test F_avg {100 * res['test']['f_avg']:.2f}  test Macro-F1 {100 * res['test']['macro_f1']:.2f}")
            log.close()
            del model, opt, best_state
            torch.cuda.empty_cache()

    # ---- 合并五个话题
    summ, done = {}, []
    for sp in ("valid", "test"):
        rows = []
        for topic in TOPICS:
            p = out / topic.replace(" ", "_") / f"preds_{sp}.jsonl"
            if p.exists():
                rows += [l for l in open(p, encoding="utf-8") if l.strip()]
                if sp == "test":
                    done.append(topic)
        with open(out / f"predictions_glan_{sp}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(rows)
    for topic in done:
        summ[topic] = json.load(open(out / topic.replace(" ", "_") / "metrics.json"))["test"]
    if summ:
        fa = np.mean([v["f_avg"] for v in summ.values()]); m3 = np.mean([v["macro_f1"] for v in summ.values()])
        print("\n【test】" + "  ".join(f"{t} {100 * v['f_avg']:.2f}" for t, v in summ.items()) +
              f"\n  话题平均 F_avg {100 * fa:.2f}（C-MTCSD 论文 GLAN 66.10）  话题平均 Macro-F1 {100 * m3:.2f}"
              f"  已完成 {len(done)}/5 个话题")
        json.dump({"per_topic": summ, "topic_avg_f_avg": fa, "topic_avg_macro_f1": m3},
                  open(out / "summary.json", "w"), ensure_ascii=False, indent=2)
    print(f"合并预测 → {out}/predictions_glan_{{valid,test}}.jsonl")


if __name__ == "__main__":
    main()
