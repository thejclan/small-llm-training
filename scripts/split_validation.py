"""依科目分層，從 TMMLU+ validation 切出保留集。

輸出（預設在 ~/hw1/data/）：
  val_holdout.jsonl           保留集，只用來評估（原始選項順序）
  val_holdout_shuffled.jsonl  同一批題目，選項順序打亂（模擬私人評分集）
  val_train.jsonl             其餘題目，可用於訓練

同樣的 --seed 與 --fraction 一定產生完全相同的檔案（可用檢查碼驗證）。
"""
import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict

HW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ~/hw1
sys.path.insert(0, HW_DIR)
# 沿用評估腳本的載入與抽樣工具，確保資料版本（ikala/tmmluplus v1.1）一致
from eval_tmmluplus import LABELS, SUBJECTS, load_hf_split, question_key, shuffle_options, subject_rng

FIELDS = ["subject", "question", *LABELS, "answer"]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in FIELDS}, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    print(f"  {os.path.basename(path):<28} {len(rows):>5} 題  sha256[:16]={digest}")


def answer_dist(rows):
    c = Counter(r["answer"] for r in rows)
    return "  ".join(f"{l}:{c[l] / len(rows):.1%}" for l in LABELS)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fraction", type=float, default=0.2, help="每個科目保留的比例")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default=os.path.join(HW_DIR, "data"))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_hf_split("validation", list(SUBJECTS))
    print(f"validation 共 {len(rows)} 題，{len({r['subject'] for r in rows})} 個科目")

    # 題目文字相同的視為同一題，整組只會進到同一邊，避免洩漏
    groups = defaultdict(list)
    for r in rows:
        groups[question_key(r)].append(r)
    n_dup = sum(len(g) - 1 for g in groups.values() if len(g) > 1)
    print(f"題目文字重複：{n_dup} 題（會與其重複題分在同一邊）")

    # 每組歸屬於一個科目，再依科目分層抽樣
    keys_by_subject = defaultdict(list)
    for key, g in groups.items():
        keys_by_subject[min(r["subject"] for r in g)].append(key)

    holdout_keys = set()
    for subject in sorted(keys_by_subject):
        keys = sorted(keys_by_subject[subject])  # 排序，讓結果不受資料原始順序影響
        k = math.ceil(len(keys) * args.fraction)
        rng = subject_rng(args.seed, subject)
        holdout_keys.update(rng.sample(keys, k))

    order = lambda r: (r["subject"], r["question"], r["answer"])
    holdout = sorted((r for r in rows if question_key(r) in holdout_keys), key=order)
    train = sorted((r for r in rows if question_key(r) not in holdout_keys), key=order)

    # 打亂選項版：每個科目用獨立、固定的亂數
    shuffled = []
    for subject in sorted({r["subject"] for r in holdout}):
        rng = subject_rng(f"{args.seed}-shuffle", subject)
        shuffled += [shuffle_options(r, rng) for r in holdout if r["subject"] == subject]

    # 最後再確認一次：保留集和訓練集沒有任何相同題目
    assert not ({question_key(r) for r in holdout} & {question_key(r) for r in train})

    print(f"\n輸出（fraction={args.fraction}, seed={args.seed}）：")
    write_jsonl(os.path.join(args.out_dir, "val_holdout.jsonl"), holdout)
    write_jsonl(os.path.join(args.out_dir, "val_holdout_shuffled.jsonl"), shuffled)
    write_jsonl(os.path.join(args.out_dir, "val_train.jsonl"), train)

    print("\n正確答案的字母分布：")
    print(f"  原始順序  {answer_dist(holdout)}")
    print(f"  打亂選項  {answer_dist(shuffled)}")


if __name__ == "__main__":
    main()