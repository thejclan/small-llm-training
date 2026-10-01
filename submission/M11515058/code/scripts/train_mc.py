"""方向 3：選擇題格式對齊（只對答案字母計算 loss 的 SFT）。

從一個 Hugging Face 格式的模型出發，用「與評估腳本完全相同的 prompt」加上正確答案字母
做訓練，輸出新的模型資料夾（tokenizer 與設定檔沿用起點模型）。

資料只用允許的部分：TMMLU+ train split、data/val_train.jsonl；
保留集（--exclude）裡出現過的題目會自動排除。

用法：
  python scripts/train_mc.py --base_model models/baseline --out_dir models/mc_rotate --augment rotate
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict

import torch

HW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ~/hw1
sys.path.insert(0, HW_DIR)
from eval_tmmluplus import (FINE_TO_CATEGORY, LABELS, SUBJECTS, build_prompt,
                            load_hf_split, load_local_file, question_key)
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = os.path.join(HW_DIR, "data")


# ---------------- 資料 ----------------
def load_sources(sources):
    rows = []
    for src in sources:
        part = load_hf_split("train", list(SUBJECTS)) if src == "train" else load_local_file(src)
        print(f"  {src}: {len(part)} 題")
        rows += part
    return rows


def rotations(row):
    """正確答案輪流放在 A、B、C、D，其餘選項維持原本的相對順序。"""
    opts = [row[l] for l in LABELS]
    k = LABELS.index(row["answer"])
    correct, others = opts[k], opts[:k] + opts[k + 1:]
    out = []
    for pos in range(4):
        new = others[:pos] + [correct] + others[pos:]
        r = dict(row, answer=LABELS[pos])
        r.update({l: new[i] for i, l in enumerate(LABELS)})
        out.append(r)
    return out


def encode(row, tok, label_ids):
    """與評估腳本相同：prompt 單獨切字（不加特殊 token），答案字母另外接在後面。"""
    ctx = tok(build_prompt(row, [], tok, False), add_special_tokens=False)["input_ids"]
    return ctx, label_ids[LABELS.index(row["answer"])]


def pad_batch(seqs, pad_id, device):
    n = max(map(len, seqs))
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids.to(device), mask.to(device)


# ---------------- 評估（與評估腳本相同的計分方式）----------------
@torch.no_grad()
def evaluate(model, tok, rows, label_ids, pad_id, device, bs=32):
    """比較 prompt 最後一個位置上 A/B/C/D 的機率，再依評估腳本的方式彙整成 macro。"""
    model.eval()
    correct, total = defaultdict(int), defaultdict(int)
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        ctxs = [encode(r, tok, label_ids)[0] for r in batch]
        ids, mask = pad_batch(ctxs, pad_id, device)
        logits = model(input_ids=ids, attention_mask=mask).logits.float()
        last = torch.tensor([len(c) - 1 for c in ctxs], device=device)
        scores = logits[torch.arange(len(batch), device=device), last][:, label_ids]
        for r, pred in zip(batch, scores.argmax(-1).tolist()):
            correct[r["subject"]] += LABELS[pred] == r["answer"]
            total[r["subject"]] += 1
    model.train()
    by_cat = defaultdict(list)
    for s in total:
        by_cat[FINE_TO_CATEGORY[SUBJECTS.get(s, (None, "other"))[1]]].append(correct[s] / total[s])
    macro = sum(sum(v) / len(v) for v in by_cat.values()) / len(by_cat)
    micro = sum(correct.values()) / sum(total.values())
    return round(macro, 4), round(micro, 4)


def report(title, evals, model, tok, label_ids, pad_id, device):
    res = {name: evaluate(model, tok, rows, label_ids, pad_id, device) for name, rows in evals.items()}
    print(f"== {title} ==")
    for name, (macro, micro) in res.items():
        print(f"  {name:<30} MACRO {macro:.4f}  micro {micro:.4f}")
    return {name: {"macro": m, "micro": u} for name, (m, u) in res.items()}


# ---------------- 主程式 ----------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_model", required=True, help="起點模型（HF 格式資料夾）")
    p.add_argument("--out_dir", required=True, help="輸出的新模型資料夾")
    p.add_argument("--train_sources", nargs="+", default=["train", os.path.join(DATA, "val_train.jsonl")],
                   help="'train' 代表 TMMLU+ train split，其餘為本地 .jsonl")
    p.add_argument("--exclude", default=os.path.join(DATA, "val_holdout.jsonl"), help="要排除的保留集")
    p.add_argument("--eval_data", nargs="*", default=[os.path.join(DATA, "val_holdout.jsonl"),
                                                      os.path.join(DATA, "val_holdout_shuffled.jsonl")])
    p.add_argument("--augment", choices=["none", "rotate"], default="rotate")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=50)
    args = p.parse_args()

    base, out_dir = os.path.abspath(args.base_model), os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        sys.exit(f"輸出資料夾已存在，為避免覆蓋而停止：{out_dir}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(base)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    label_ids = []
    for l in LABELS:
        ids = tok(l, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"選項字母 {l} 不是單一 token"
        label_ids.append(ids[0])

    # 資料
    print("訓練資料：")
    rows = load_sources(args.train_sources)
    held = {question_key(r) for r in load_local_file(args.exclude)}
    n_before = len(rows)
    rows = [r for r in rows if question_key(r) not in held]
    print(f"  排除與保留集重複的題目：{n_before - len(rows)} 題，剩 {len(rows)} 題")

    # rotate：每題 4 個版本；none：原始順序重複 4 次，讓兩者的訓練步數相同，只差在選項順序
    examples = [v for r in rows for v in (rotations(r) if args.augment == "rotate" else [r] * 4)]
    data = [encode(r, tok, label_ids) for r in examples]
    dist = Counter(r["answer"] for r in examples)
    print(f"  訓練筆數 {len(data)}（augment={args.augment}），正確答案分布 {dict(sorted(dist.items()))}")

    ctx, y = data[0]
    print("---- 訓練資料範例（模型實際看到的內容，【】內是要學的答案）----")
    print(tok.decode(ctx) + "【" + tok.decode([y]) + "】")
    print("-" * 40)

    evals = {os.path.basename(x): load_local_file(x) for x in args.eval_data}
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).to(device)
    history = [{"epoch": 0, **report("訓練前（應與官方評估的起點模型分數一致）", evals, model, tok, label_ids, pad_id, device)}]

    # 訓練
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(data) / args.batch_size) * args.epochs
    warm = max(1, int(total_steps * args.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total_steps - s) / max(1, total_steps - warm))))
    rng = random.Random(args.seed)
    step, run_loss, start = 0, 0.0, time.time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(data)))
        rng.shuffle(order)
        for b in range(0, len(order), args.batch_size):
            items = [data[i] for i in order[b:b + args.batch_size]]
            ids, mask = pad_batch([c + [y] for c, y in items], pad_id, device)
            labels = torch.full_like(ids, -100)
            for i, (c, y) in enumerate(items):
                labels[i, len(c)] = y  # 只有答案字母計算 loss
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run_loss += loss.item()
            if step % args.log_interval == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {run_loss / args.log_interval:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e}")
                run_loss = 0.0
        title = f"epoch {epoch} 結束（累計 {(time.time() - start) / 60:.1f} 分鐘）"
        history.append({"epoch": epoch, **report(title, evals, model, tok, label_ids, pad_id, device)})

    # 儲存：權重用新的，tokenizer 與設定檔沿用起點模型（保留相容性調整）
    model.save_pretrained(out_dir)
    for f in os.listdir(base):
        src = os.path.join(base, f)
        if os.path.isfile(src) and not f.endswith((".safetensors", ".bin")) and f != "train_config.json":
            shutil.copy2(src, os.path.join(out_dir, f))
    minutes = round((time.time() - start) / 60, 1)
    with open(os.path.join(out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "n_questions": len(rows), "n_examples": len(data),
                   "minutes": minutes, "history": history}, f, ensure_ascii=False, indent=2)

    reloaded = AutoModelForCausalLM.from_pretrained(out_dir)
    n = sum(q.numel() for q in reloaded.parameters())
    print(f"\n完成：{out_dir}（重新載入成功，參數量 {n / 1e6:.2f}M，訓練 {minutes} 分鐘）")


if __name__ == "__main__":
    main()