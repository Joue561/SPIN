"""LoRA SFT of Qwen2.5-7B-Instruct on ZS-CSD (StanceAgent-R1 data/default format), then
export dev/test predictions for SPIN.

Data: StanceAgent-R1-main/data/default/{train,dev,test}_data.json
      fields: id, target, sentences (list, last = current utterance), speakers (list), label (支持/反对/中立)

Usage (single A100-40GB, GPU chosen with CUDA_VISIBLE_DEVICES):
  # 1) train; keeps the adapter with the best dev macro-F1 (dev only, never test)
  CUDA_VISIBLE_DEVICES=1 python -u qwen_sft_zscsd.py train \
      --data_dir StanceAgent-R1-main/data/default --model Qwen/Qwen2.5-7B-Instruct --out qwen7b_sft
  # 2) predict dev + test with the best adapter
  CUDA_VISIBLE_DEVICES=1 python -u qwen_sft_zscsd.py predict \
      --data_dir StanceAgent-R1-main/data/default --model Qwen/Qwen2.5-7B-Instruct --out qwen7b_sft

  # low-resource runs (A1): add --train_frac 0.1 (the ids used are saved to <out>/train_ids_used.json)

Prediction scores the first token of the three labels, so every node gets a valid label.
"""
import argparse, json, math, os, random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

LABELS = ["支持", "反对", "中立"]
SYSTEM = "你是一个立场检测助手。根据对话上下文，判断当前发言对目标的立场，只回答“支持”“反对”或“中立”之一。"


_NAME2CN = {"favor": "支持", "支持": "支持", "against": "反对", "反对": "反对",
            "neutral": "中立", "none": "中立", "中立": "中立"}


def load(data_dir, split):
    """Labels in data/default are integers; label.json gives their names. Convert to 支持/反对/中立."""
    lm = json.load(open(Path(data_dir) / "label.json", encoding="utf-8"))
    i2cn = {int(v): _NAME2CN[str(k).lower()] for k, v in lm.items()} if not isinstance(lm, list) \
        else {i: _NAME2CN[str(k).lower()] for i, k in enumerate(lm)}
    data = json.load(open(Path(data_dir) / f"{split}_data.json", encoding="utf-8"))
    for ex in data:
        if not isinstance(ex["label"], str):
            ex["label"] = i2cn[int(ex["label"])]
        ex["speakers"] = [str(x) for x in ex.get("speakers") or [""] * len(ex["sentences"])]
    return data


def build_messages(ex, max_hist_chars):
    sents, spk = ex["sentences"], ex.get("speakers") or [""] * len(ex["sentences"])
    hist = [f"[{spk[i]}] {sents[i]}" for i in range(len(sents) - 1)]
    hist_text = "\n".join(hist)
    if len(hist_text) > max_hist_chars:  # keep the most recent context (left truncation)
        hist_text = "……" + hist_text[-max_hist_chars:]
    user = (f"目标：{ex['target']}\n\n对话历史：\n{hist_text or '（无）'}\n\n"
            f"当前发言：[{spk[-1]}] {sents[-1]}\n\n当前发言对目标的立场是？")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


class SFTData(Dataset):
    def __init__(self, data, tok, max_len, max_hist_chars):
        self.items = []
        for ex in data:
            prompt = tok.apply_chat_template(build_messages(ex, max_hist_chars), tokenize=False,
                                             add_generation_prompt=True)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            a_ids = tok(ex["label"] + tok.eos_token, add_special_tokens=False)["input_ids"]
            p_ids = p_ids[-(max_len - len(a_ids)):]
            self.items.append((p_ids + a_ids, [-100] * len(p_ids) + a_ids))

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate(batch, pad_id):
    L = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), L), pad_id); lab = torch.full((len(batch), L), -100)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x); lab[i, :len(y)] = torch.tensor(y); att[i, :len(x)] = 1
    return ids, att, lab


@torch.no_grad()
def predict(model, tok, data, max_hist_chars, bs=16):
    first = [tok(l, add_special_tokens=False)["input_ids"][0] for l in LABELS]
    assert len(set(first)) == 3, "label first tokens collide; switch to full-sequence scoring"
    tok.padding_side = "left"
    preds = []
    model.eval()
    for i in range(0, len(data), bs):
        chunk = data[i:i + bs]
        prompts = [tok.apply_chat_template(build_messages(ex, max_hist_chars), tokenize=False,
                                           add_generation_prompt=True) for ex in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048,
                  add_special_tokens=False).to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        probs = torch.softmax(logits[:, first], dim=-1).cpu().tolist()
        for ex, p in zip(chunk, probs):
            preds.append({"id": ex["id"], "pred": LABELS[max(range(3), key=lambda k: p[k])],
                          "probs": dict(zip(LABELS, [round(v, 4) for v in p])), "label": ex["label"]})
    return preds


def macro_f1(preds):
    f1s = []
    for c in LABELS:
        tp = sum(p["pred"] == c and p["label"] == c for p in preds)
        fp = sum(p["pred"] == c and p["label"] != c for p in preds)
        fn = sum(p["pred"] != c and p["label"] == c for p in preds)
        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    return 100 * sum(f1s) / 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "predict"])
    ap.add_argument("--data_dir", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=4); ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024); ap.add_argument("--max_hist_chars", type=int, default=1200)
    ap.add_argument("--lora_r", type=int, default=16); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--evals_per_epoch", type=int, default=2)
    ap.add_argument("--train_frac", type=float, default=1.0,
                    help="fraction of the training split to use (random sample, fixed --sample_seed)")
    ap.add_argument("--sample_seed", type=int, default=13)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print("loading model ...", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map={"": 0},
                                                attn_implementation="sdpa")
    dev = load(a.data_dir, "dev")

    if a.mode == "train":
        base.gradient_checkpointing_enable(); base.enable_input_require_grads()
        model = get_peft_model(base, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                               task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
        tok.padding_side = "right"
        print("tokenizing train ...", flush=True)
        train = load(a.data_dir, "train")
        if a.train_frac < 1.0:
            train = random.Random(a.sample_seed).sample(train, int(round(len(train) * a.train_frac)))
        json.dump(sorted(ex["id"] for ex in train), open(out / "train_ids_used.json", "w"))
        ds = SFTData(train, tok, a.max_len, a.max_hist_chars)
        print(f"train samples {len(ds)}", flush=True)
        dl = DataLoader(ds, batch_size=a.bs, shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id))
        steps = math.ceil(len(dl) / a.grad_accum) * a.epochs
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
        sch = get_cosine_schedule_with_warmup(opt, int(0.03 * steps), steps)
        eval_every = max(1, math.ceil(len(dl) / a.grad_accum) // a.evals_per_epoch)
        best, step = -1, 0
        log = open(out / "train_log.txt", "a", encoding="utf-8")
        for ep in range(a.epochs):
            model.train()
            for it, (ids, att, lab) in enumerate(dl):
                loss = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda()).loss / a.grad_accum
                loss.backward()
                if (it + 1) % a.grad_accum == 0 or it + 1 == len(dl):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); sch.step(); opt.zero_grad(); step += 1
                    if step % 20 == 0:
                        print(f"ep {ep} step {step}/{steps} loss {loss.item() * a.grad_accum:.4f}", flush=True)
                    if step % eval_every == 0 or step == steps:
                        f1 = macro_f1(predict(model, tok, dev, a.max_hist_chars)); tok.padding_side = "right"
                        model.train()
                        msg = f"step {step}: dev macro-F1 {f1:.2f}" + ("  *best*" if f1 > best else "")
                        print(msg, flush=True); log.write(msg + "\n"); log.flush()
                        if f1 > best:
                            best = f1; model.save_pretrained(out / "best_adapter")
        print(f"best dev macro-F1 {best:.2f}; adapter saved to {out / 'best_adapter'}")
    else:
        model = PeftModel.from_pretrained(base, out / "best_adapter")
        for split in ["dev", "test"]:
            preds = predict(model, tok, load(a.data_dir, split), a.max_hist_chars)
            with open(out / f"preds_{split}.jsonl", "w", encoding="utf-8") as f:
                for p in preds: f.write(json.dumps(p, ensure_ascii=False) + "\n")
            print(f"{split}: n={len(preds)}  macro-F1 {macro_f1(preds):.2f}  -> {out / f'preds_{split}.jsonl'}")


if __name__ == "__main__":
    main()