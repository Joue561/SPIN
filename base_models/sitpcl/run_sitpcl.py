# -*- coding: utf-8 -*-
"""
run_sitpcl.py
复现 ZS-CSD 原论文的 SITPCL，并导出逐条预测，供知识库会话树推断叠加使用。

不修改原仓库：直接导入 ZS-CSD/main.py 中的 Main 和模型，训练流程、超参数、选模规则
（按 dev F1-macro，取 >= 的最新 epoch）与原代码完全一致。唯一的区别是：
每当 dev 刷新最佳成绩时，同时保存该 epoch 在 dev 和 test 上的逐条预测与概率
（原代码只保存 test 预测，而知识库推断需要同一模型在 valid 上的预测来估计混淆矩阵、选 λ）。

输出（--out_dir/seed<seed>/）：
  sitpcl_dev_raw.jsonl / sitpcl_test_raw.jsonl   每行 {split, zscsd_id, true, pred, probs}
  metrics.json                                   最佳 epoch 及 dev / test 指标
  train.log

用法（make_rag_data 目录下，用 2 号卡）：
python -u spin/base_models/sitpcl/run_sitpcl.py --repo ZS-CSD --out_dir zscsd_exp/outputs/sitpcl \
  --cuda_index 2 --seed 1234
调试（各划分只取 100 条，1 个 epoch）：加 --debug --epoch_size 1
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

DEFAULTS = dict(bert_lr=1e-5, other_lr=1e-5, weight_decay=1e-6, hidden_size=768, num_classes=3, alpha=1.0,
                tau=0.1, cuda_index=0, seed=1234, debug=False, time="", gru_layer=1, gru_hidden=768,
                model_type="SITPCL", plm="roberta", bert_dir="./plm/chinese-roberta-wwm-ext/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="ZS-CSD 仓库目录（含 main.py、src/、data/）")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cuda_index", type=int, default=0)
    ap.add_argument("--bert_dir", default=None, help="默认用原代码的 ./plm/chinese-roberta-wwm-ext/（相对仓库目录）")
    ap.add_argument("--epoch_size", type=int, default=None, help="默认用 config.yaml 中的 20")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--save_model", action="store_true", help="同时保存最佳模型参数（约 400MB）")
    a = ap.parse_args()

    out = (Path(a.out_dir) / f"seed{a.seed}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(a.repo).resolve()
    os.chdir(repo)                      # 原代码用相对路径读 src/config.yaml、data/、plm/
    sys.path.insert(0, str(repo))
    import torch
    import main as zs                   # ZS-CSD/main.py

    args = argparse.Namespace(**DEFAULTS)
    args.seed, args.cuda_index, args.debug = a.seed, a.cuda_index, a.debug
    args.time = f"export_seed{a.seed}_{time.strftime('%m%d%H%M')}"
    if a.bert_dir:
        args.bert_dir = a.bert_dir

    class MainExport(zs.Main):
        def collect(self, mode):
            self.model.eval()
            loader = self.devLoader if mode == "dev" else self.testLoader
            ids, trues, preds, probs, loss_sum = [], [], [], [], 0.0
            with torch.no_grad():
                for data in loader:
                    loss, logits, labels = self.model(**data)
                    loss_sum += loss.item()
                    p = torch.softmax(logits, -1).detach().cpu().numpy()
                    probs.extend(p.tolist())
                    preds.extend(np.argmax(p, -1).tolist())
                    trues.extend(labels.detach().cpu().numpy().reshape(-1).tolist())
                    ids.extend(list(data["doc_id"]))
            m = self.get_metrices(trues, preds)
            return dict(zip(["macro_f1", "favor", "against", "neutral", "f_avg", "acc"], map(float, m))), \
                ids, trues, preds, probs, loss_sum / len(loader)

        def dump(self, mode, ids, trues, preds, probs):
            with open(out / f"sitpcl_{mode}_raw.jsonl", "w", encoding="utf-8") as f:
                for i, t, p, pr in zip(ids, trues, preds, probs):
                    f.write(json.dumps({"split": mode, "zscsd_id": int(i), "true": int(t), "pred": int(p),
                                        "probs": [round(x, 6) for x in pr]}) + "\n")

        def train(self):                # 与原代码相同的选模规则：dev F1 >= 历史最佳时更新
            best_dev = 0.0
            for epoch in range(self.config.epoch_size):
                train_loss = self.train_iter()
                dm, *dev_rest = self.collect("dev")
                msg = f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Dev Macro F1: {100 * dm['macro_f1']:.2f}"
                logging.info(msg); print(msg, flush=True)
                if dm["macro_f1"] >= best_dev:
                    best_dev = dm["macro_f1"]
                    tm, *test_rest = self.collect("test")
                    self.dump("dev", *dev_rest[:4]); self.dump("test", *test_rest[:4])
                    self.best_epoch = epoch + 1
                    json.dump({"seed": a.seed, "best_epoch": self.best_epoch, "dev": dm, "test": tm,
                               "epoch_size": self.config.epoch_size, "debug": a.debug},
                              open(out / "metrics.json", "w"), indent=2)
                    msg = (f"  ↳ 新的最佳 dev，test Macro F1 {100 * tm['macro_f1']:.2f}  F_avg {100 * tm['f_avg']:.2f}"
                           f"（已保存 dev/test 预测）")
                    logging.info(msg); print(msg, flush=True)
                    if a.save_model:
                        torch.save(self.model.state_dict(), out / "best_model.pth")

    m = MainExport(args)
    if a.epoch_size:
        m.config.epoch_size = a.epoch_size
    fh = logging.FileHandler(out / "train.log", encoding="utf-8")
    logging.getLogger().addHandler(fh)
    print(f"设备 {m.config.device}，seed {a.seed}，epoch {m.config.epoch_size}，输出 {out}", flush=True)
    m.forward()
    r = json.load(open(out / "metrics.json"))
    print(f"\n完成：最佳 epoch {r['best_epoch']}，dev F1-macro {100 * r['dev']['macro_f1']:.2f}，"
          f"test F1-macro {100 * r['test']['macro_f1']:.2f}（原论文 43.81）")


if __name__ == "__main__":
    main()