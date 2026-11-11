#!/usr/bin/env python3
"""
train_plm_stance_model.py

用法：
python train_plm_stance_model.py \
  --data_dir plm_experiment_data/plm_semantic_rag_feature \
  --output_dir plm_model_outputs/plm_semantic_rag_feature_macbert \
  --base_model hfl/chinese-macbert-base \
  --use_rag_features true \
  --max_length 512 \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --seed 42
"""

import json
import os
import argparse
import random
from pathlib import Path
from collections import Counter
from typing import Optional

import numpy as np

LABEL_NAMES = {0: "favor", 1: "against", 2: "none"}
LABEL_LIST  = ["favor", "against", "none"]

RAG_FEATURE_KEYS = [
    "rag_favor_prob", "rag_against_prob", "rag_none_prob",
    "rag_dominant_label_id", "rag_dominant_label_probability",
    "rag_avg_memory_confidence", "rag_avg_memory_weighted_count",
    "rag_high_reliability_count", "rag_medium_reliability_count",
    "rag_low_reliability_count", "rag_memory_count",
    "rag_pattern_match_score", "rag_semantic_match_score",
    "rag_target_relation_match_score", "rag_reply_relation_match_score",
    "rag_strong_usage_count", "rag_weak_usage_count",
]


# ============ IO ============

def load_jsonl(filepath: str) -> list:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list, filepath: Path):
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  ✅ {filepath.name}（{len(records)} 条）")


def write_json(data, filepath: Path):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {filepath.name}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ============ Dataset ============

def build_dataset_class():
    """延迟导入 torch，避免脚本在无 GPU 环境下 import 失败"""
    import torch
    from torch.utils.data import Dataset

    class StanceDataset(Dataset):
        def __init__(self, records, tokenizer, max_length, use_rag_features):
            self.records = records
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.use_rag_features = use_rag_features

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            rec = self.records[idx]
            enc = self.tokenizer(
                rec["input_text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            item = {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels":         torch.tensor(rec["label"], dtype=torch.long),
            }
            if self.use_rag_features:
                feats = rec.get("rag_features", {}) or {}
                vec = [float(feats.get(k, 0.0)) for k in RAG_FEATURE_KEYS]
                item["rag_features"] = torch.tensor(vec, dtype=torch.float)
            return item

    return StanceDataset


# ============ 模型 ============

def build_fusion_model_class():
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class TextRagFusionModel(nn.Module):
        def __init__(self, base_model_name, rag_dim, num_labels=3, dropout=0.1):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(base_model_name)
            hidden_size = self.encoder.config.hidden_size
            self.rag_dim = rag_dim

            self.classifier = nn.Sequential(
                nn.Linear(hidden_size + rag_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_labels),
            )
            self.num_labels = num_labels

        def forward(self, input_ids, attention_mask, rag_features, labels=None):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            # 优先用 pooler_output，没有则取 [CLS] 对应的 last_hidden_state
            if getattr(outputs, "pooler_output", None) is not None:
                text_vec = outputs.pooler_output
            else:
                text_vec = outputs.last_hidden_state[:, 0, :]

            combined = torch.cat([text_vec, rag_features], dim=-1)
            logits = self.classifier(combined)

            loss = None
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)

            return {"loss": loss, "logits": logits}

    return TextRagFusionModel


# ============ 评估指标 ============

def compute_metrics_from_preds(preds: list, labels: list) -> dict:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
    )

    acc    = accuracy_score(labels, preds)
    mac_f1 = f1_score(labels, preds, labels=[0, 1, 2], average="macro", zero_division=0)
    wgt_f1 = f1_score(labels, preds, labels=[0, 1, 2], average="weighted", zero_division=0)

    per_class = {}
    for lid, lname in LABEL_NAMES.items():
        per_class[lname] = {
            "precision": round(precision_score(labels, preds, labels=[lid], average="micro", zero_division=0), 4),
            "recall":    round(recall_score(labels, preds, labels=[lid], average="micro", zero_division=0), 4),
            "f1":        round(f1_score(labels, preds, labels=[lid], average="micro", zero_division=0), 4),
            "support":   labels.count(lid) if isinstance(labels, list) else int((np.array(labels) == lid).sum()),
        }

    cm = confusion_matrix(labels, preds, labels=[0, 1, 2]).tolist()

    return {
        "accuracy":    round(float(acc), 4),
        "macro_f1":    round(float(mac_f1), 4),
        "weighted_f1": round(float(wgt_f1), 4),
        "per_class":   per_class,
        "confusion_matrix": {"labels": LABEL_LIST, "matrix": cm},
    }


def group_metrics(records: list, preds: list, labels: list, group_fn) -> dict:
    groups = {}
    for rec, p, l in zip(records, preds, labels):
        key = group_fn(rec)
        if key is None:
            continue
        groups.setdefault(str(key), {"preds": [], "labels": []})
        groups[str(key)]["preds"].append(p)
        groups[str(key)]["labels"].append(l)

    result = {}
    for key, d in groups.items():
        if len(d["labels"]) == 0:
            continue
        result[key] = compute_metrics_from_preds(d["preds"], d["labels"])
        result[key]["support"] = len(d["labels"])
    return result


def depth_group(rec):
    d = rec.get("depth", 0) or 0
    if d == 1: return "depth=1"
    if d == 2: return "depth=2"
    return "depth>=3"


def short_text_group(rec):
    ct = str(rec.get("current_text", "")).replace(" ", "")
    if len(ct) <= 4:
        return "short_text_le4"
    if rec.get("is_ambiguous_current"):
        return "is_ambiguous"
    return None


# ============ 训练（纯文本分类模型，用 Trainer） ============

def train_text_classification_model(args, train_recs, valid_recs, test_recs):
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, EarlyStoppingCallback,
    )
    from torch.utils.data import Dataset

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=3
    )

    class SimpleDataset(Dataset):
        def __init__(self, records):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            rec = self.records[idx]
            enc = tokenizer(
                rec["input_text"], truncation=True,
                max_length=args.max_length, padding="max_length",
            )
            enc["labels"] = rec["label"]
            return enc

    train_ds = SimpleDataset(train_recs)
    valid_ds = SimpleDataset(valid_recs)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        m = compute_metrics_from_preds(preds.tolist(), labels.tolist())
        return {
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
        }

    use_fp16 = torch.cuda.is_available() and args.fp16

    training_args = TrainingArguments(
        output_dir=str(Path(args.output_dir) / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=use_fp16,
        seed=args.seed,
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    train_result = trainer.train()
    best_model_dir = Path(args.output_dir) / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    training_log = {
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_loss":    train_result.metrics.get("train_loss"),
        "epochs":        args.epochs,
        "base_model":    args.base_model,
        "learning_rate": args.learning_rate,
        "batch_size":    args.batch_size,
        "seed":          args.seed,
    }

    def predict_split(records):
        ds = SimpleDataset(records)
        pred_out = trainer.predict(ds)
        logits = pred_out.predictions
        preds = np.argmax(logits, axis=-1).tolist()
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        return preds, probs

    return trainer, training_log, predict_split


# ============ 训练（文本+RAG特征融合模型，手写训练循环） ============

def train_fusion_model(args, train_recs, valid_recs, test_recs):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  使用设备：{device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    DatasetClass = build_dataset_class()
    ModelClass = build_fusion_model_class()

    train_ds = DatasetClass(train_recs, tokenizer, args.max_length, use_rag_features=True)
    valid_ds = DatasetClass(valid_recs, tokenizer, args.max_length, use_rag_features=True)
    test_ds  = DatasetClass(test_recs,  tokenizer, args.max_length, use_rag_features=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    model = ModelClass(args.base_model, rag_dim=len(RAG_FEATURE_KEYS), num_labels=3).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_fp16 = torch.cuda.is_available() and args.fp16
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    best_macro_f1 = -1.0
    patience_counter = 0
    training_log = {"epochs_log": []}
    best_model_dir = Path(args.output_dir) / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            rag_features   = batch["rag_features"].to(device)
            labels         = batch["labels"].to(device)

            if use_fp16:
                with torch.cuda.amp.autocast():
                    out = model(input_ids, attention_mask, rag_features, labels)
                    loss = out["loss"] / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                out = model(input_ids, attention_mask, rag_features, labels)
                loss = out["loss"] / args.gradient_accumulation_steps
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            total_loss += out["loss"].item()

        avg_loss = total_loss / max(len(train_loader), 1)

        # ---- 验证 ----
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                rag_features   = batch["rag_features"].to(device)
                labels         = batch["labels"].to(device)
                out = model(input_ids, attention_mask, rag_features)
                preds = torch.argmax(out["logits"], dim=-1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        val_metrics = compute_metrics_from_preds(val_preds, val_labels)
        print(f"  Epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}  "
              f"val_acc={val_metrics['accuracy']}  val_macro_f1={val_metrics['macro_f1']}")
        training_log["epochs_log"].append({
            "epoch": epoch + 1, "train_loss": avg_loss,
            "val_accuracy": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"],
        })

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            torch.save(model.state_dict(), str(best_model_dir / "model.pt"))
            tokenizer.save_pretrained(str(best_model_dir))
            with open(best_model_dir / "model_config.json", "w") as f:
                json.dump({
                    "base_model": args.base_model,
                    "rag_dim": len(RAG_FEATURE_KEYS),
                    "num_labels": 3,
                }, f)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"  早停触发（patience={args.early_stopping_patience}）")
                break

    training_log["best_macro_f1"] = best_macro_f1
    training_log["base_model"] = args.base_model
    training_log["learning_rate"] = args.learning_rate
    training_log["batch_size"] = args.batch_size
    training_log["seed"] = args.seed

    # 加载 best model 用于预测
    model.load_state_dict(torch.load(str(best_model_dir / "model.pt"), map_location=device))
    model.eval()

    def predict_split(records):
        ds = DatasetClass(records, tokenizer, args.max_length, use_rag_features=True)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        all_preds, all_probs = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                rag_features   = batch["rag_features"].to(device)
                out = model(input_ids, attention_mask, rag_features)
                probs = torch.softmax(out["logits"], dim=-1)
                preds = torch.argmax(probs, dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())
        return all_preds, np.array(all_probs)

    return model, training_log, predict_split


# ============ 预测结果输出 ============

def write_predictions(records, preds, probs, filepath: Path):
    out_records = []
    for rec, pred, prob in zip(records, preds, probs):
        out = {
            "node_id":      rec.get("node_id"),
            "topic":        rec.get("topic"),
            "split":        rec.get("split"),
            "current_text": rec.get("current_text"),
            "parent_text":  rec.get("parent_text"),
            "depth":        rec.get("depth"),
            "is_ambiguous_current": rec.get("is_ambiguous_current"),
            "llm_reply_relation":       rec.get("llm_reply_relation"),
            "llm_semantic_propagation": rec.get("llm_semantic_propagation"),
            "llm_target_relation":      rec.get("llm_target_relation"),
            "llm_candidate_patterns":   rec.get("llm_candidate_patterns"),
            "stance_label":     LABEL_NAMES[rec["label"]],
            "predicted_label":  LABEL_NAMES[int(pred)],
            "predicted_probs":  {LABEL_NAMES[i]: round(float(p), 4) for i, p in enumerate(prob)},
        }
        out_records.append(out)
    write_jsonl(out_records, filepath)
    return out_records


def build_full_evaluation(records, preds, labels, filepath: Path):
    overall = compute_metrics_from_preds(preds, labels)
    overall["by_topic"]               = group_metrics(records, preds, labels, lambda r: r.get("topic"))
    overall["by_depth"]                = group_metrics(records, preds, labels, depth_group)
    overall["by_short_text"]           = group_metrics(records, preds, labels, short_text_group)
    overall["by_semantic_propagation"] = group_metrics(
        records, preds, labels, lambda r: r.get("llm_semantic_propagation") or None
    )
    overall["by_target_relation"]      = group_metrics(
        records, preds, labels, lambda r: r.get("llm_target_relation") or None
    )
    overall["by_reply_relation"]       = group_metrics(
        records, preds, labels, lambda r: r.get("llm_reply_relation") or None
    )
    write_json(overall, filepath)
    return overall


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="训练 PLM 立场检测模型")
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model", default="hfl/chinese-macbert-base")
    parser.add_argument("--use_rag_features", default="false")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs",     type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay",  type=float, default=0.01)
    parser.add_argument("--warmup_ratio",  type=float, default=0.1)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", default="true")
    args = parser.parse_args()

    args.use_rag_features = args.use_rag_features.lower() == "true"
    args.fp16 = args.fp16.lower() == "true"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    print(f"📂 读取数据：{args.data_dir}")
    train_recs = load_jsonl(str(Path(args.data_dir) / "train.jsonl"))
    valid_recs = load_jsonl(str(Path(args.data_dir) / "valid.jsonl"))
    test_recs  = load_jsonl(str(Path(args.data_dir) / "test.jsonl"))
    print(f"  train={len(train_recs)}  valid={len(valid_recs)}  test={len(test_recs)}")

    print(f"\n🏋️ 开始训练（base_model={args.base_model}，use_rag_features={args.use_rag_features}）...")

    if args.use_rag_features:
        model, training_log, predict_fn = train_fusion_model(args, train_recs, valid_recs, test_recs)
    else:
        trainer, training_log, predict_fn = train_text_classification_model(
            args, train_recs, valid_recs, test_recs
        )

    write_json(training_log, output_dir / "training_log.json")

    # ---- valid 预测与评估 ----
    print("\n📊 在 valid 上预测...")
    valid_preds, valid_probs = predict_fn(valid_recs)
    valid_labels = [r["label"] for r in valid_recs]
    write_predictions(valid_recs, valid_preds, valid_probs, output_dir / "predictions_valid.jsonl")
    eval_valid = build_full_evaluation(valid_recs, valid_preds, valid_labels, output_dir / "evaluation_valid.json")
    print(f"  valid: accuracy={eval_valid['accuracy']}  macro_f1={eval_valid['macro_f1']}")

    # ---- test 预测与评估 ----
    print("\n📊 在 test 上预测...")
    test_preds, test_probs = predict_fn(test_recs)
    test_labels = [r["label"] for r in test_recs]
    write_predictions(test_recs, test_preds, test_probs, output_dir / "predictions_test.jsonl")
    eval_test = build_full_evaluation(test_recs, test_preds, test_labels, output_dir / "evaluation_test.json")
    print(f"  test:  accuracy={eval_test['accuracy']}  macro_f1={eval_test['macro_f1']}")

    print(f"""
✅ 训练完成！
  输出目录：{output_dir}
  best_model/
  predictions_valid.jsonl  predictions_test.jsonl
  evaluation_valid.json    evaluation_test.json
  training_log.json
    """)


if __name__ == "__main__":
    main()
