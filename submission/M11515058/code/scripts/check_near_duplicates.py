"""檢查保留集與訓練資料之間「幾乎重複」的題目，並計算排除後的準確率。

相似度：題目加選項的文字切成連續 3 個字的片段（3-gram），
計算兩題相同片段的比例（Jaccard，0~1，1 代表完全相同）。
"""
import argparse
import json
import os
import sys

HW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ~/hw1
sys.path.insert(0, HW_DIR)
from eval_tmmluplus import LABELS, SUBJECTS, load_hf_split, load_local_file, question_key

DATA = os.path.join(HW_DIR, "data")
RESULTS = os.path.join(HW_DIR, "results")


def grams(row, n=3):
    text = "".join((row["question"] + "".join(str(row[l]) for l in LABELS)).split())  # 去掉空白
    return {text[i:i + n] for i in range(max(1, len(text) - n + 1))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threshold", type=float, default=0.7, help="視為幾乎重複的相似度門檻")
    p.add_argument("--models", nargs="+", default=["baseline", "mc_ep10"], help="results/ 中預測檔的名稱前綴")
    args = p.parse_args()

    holdout = load_local_file(os.path.join(DATA, "val_holdout.jsonl"))
    train = load_hf_split("train", list(SUBJECTS)) + load_local_file(os.path.join(DATA, "val_train.jsonl"))
    held_keys = {question_key(r) for r in holdout}
    train = [r for r in train if question_key(r) not in held_keys]  # 與訓練腳本相同，先排除完全重複

    # 每個保留集題目，找出最相似的訓練題
    train_grams = [grams(r) for r in train]
    best = []
    for h in holdout:
        hg = grams(h)
        sims = [jaccard(hg, g) for g in train_grams]
        j = max(range(len(sims)), key=sims.__getitem__)
        best.append((sims[j], j))

    print(f"保留集 {len(holdout)} 題 vs 訓練資料 {len(train)} 題")
    for t in (0.5, 0.7, 0.9):
        print(f"  最高相似度 ≥ {t}：{sum(s >= t for s, _ in best)} 題")

    print("\n最相似的 10 組：")
    for i in sorted(range(len(holdout)), key=lambda i: -best[i][0])[:10]:
        s, j = best[i]
        print(f"  {s:.2f} [{holdout[i]['subject']}]")
        print(f"     保留集：{holdout[i]['question'][:60]}")
        print(f"     訓練題：{train[j]['question'][:60]}")

    # 排除幾乎重複的題目後重新計算準確率
    # 打亂版保留集與原始版的題目順序相同，所以用題號對應
    flagged = {i for i, (s, _) in enumerate(best) if s >= args.threshold}
    print(f"\n排除相似度 ≥ {args.threshold} 的 {len(flagged)} 題後的準確率（micro）：")
    for d in ("holdout", "holdout_shuffled"):
        rows = load_local_file(os.path.join(DATA, f"val_{d}.jsonl"))
        assert [r["question"] for r in rows] == [r["question"] for r in holdout], "題目順序不一致"
        for m in args.models:
            with open(os.path.join(RESULTS, f"{m}_{d}_pred.jsonl")) as f:
                preds = [json.loads(l)["pred"] for l in f]
            assert len(preds) == len(rows)
            ok = [pr == r["answer"] for pr, r in zip(preds, rows)]
            kept = [o for i, o in enumerate(ok) if i not in flagged]
            print(f"  {d:<17} {m:<9} 全部 {sum(ok) / len(ok):.4f}  排除後 {sum(kept) / len(kept):.4f}（{len(kept)} 題）")


if __name__ == "__main__":
    main()