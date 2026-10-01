"""分析 MiniMind tokenizer 處理 TMMLU+（繁體中文）的狀況。

只使用 TMMLU+ validation（不碰 test），prompt 格式與評估腳本預設相同
（無 chat template、無範例題）。

1. 題目長度分布：token 數，以及超過訓練長度（pretrain 340、SFT 768）的比例
2. 繁簡對比：同一批題目轉成簡體後的 token 數，量化 tokenizer 對繁體的額外成本
3. 被拆碎的字：無法用單一 token 表示的中文字比例與常見例子
"""
import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter

HW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ~/hw1
sys.path.insert(0, HW_DIR)
from eval_tmmluplus import LABELS, SUBJECTS, build_prompt, load_hf_split

from opencc import OpenCC
from transformers import AutoTokenizer

CJK = re.compile(r"[\u4e00-\u9fff]")  # 常用中文字的 Unicode 範圍


def length_stats(lengths, limits):
    s = sorted(lengths)
    pct = lambda q: s[min(len(s) - 1, int(q * len(s)))]
    out = {"mean": round(statistics.mean(s), 1), "median": statistics.median(s),
           "p90": pct(0.9), "p99": pct(0.99), "max": s[-1]}
    for limit in limits:
        out[f"over_{limit}"] = round(sum(x > limit for x in s) / len(s), 4)
    return out


def char_stats(texts, ntok):
    """統計中文字：哪些字無法用單一 token 表示。"""
    counts = Counter(c for t in texts for c in CJK.findall(t))
    fragmented = {}
    for c in counts:
        n = ntok(c)
        if n > 1:
            fragmented[c] = n
    total = sum(counts.values())
    summary = {
        "unique_chars": len(counts),
        "unique_fragmented_ratio": round(len(fragmented) / len(counts), 4),
        "occurrence_fragmented_ratio": round(sum(counts[c] for c in fragmented) / total, 4),
    }
    return counts, fragmented, summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", default=os.path.join(HW_DIR, "models", "baseline"))
    p.add_argument("--output", default=os.path.join(HW_DIR, "results", "tokenizer_analysis.json"))
    p.add_argument("--limits", type=int, nargs="+", default=[340, 768], help="要比較的訓練長度")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # 與評估腳本相同的切法
    cc = OpenCC("t2s")  # 繁體轉簡體

    rows = load_hf_split("validation", list(SUBJECTS))

    # 1、2：題目長度，繁體原文 vs 轉成簡體
    prompts_t = [build_prompt(r, [], tok, False) for r in rows]
    prompts_s = [cc.convert(x) for x in prompts_t]
    len_t = [ntok(x) for x in prompts_t]
    len_s = [ntok(x) for x in prompts_s]
    stats_t = length_stats(len_t, args.limits)
    stats_s = length_stats(len_s, args.limits)
    extra = sum(len_t) / sum(len_s) - 1
    cpt_t = sum(map(len, prompts_t)) / sum(len_t)
    cpt_s = sum(map(len, prompts_s)) / sum(len_s)

    # 3：被拆碎的字（只看題目和選項，不含固定的指示文字）
    texts_t = [r["question"] + "".join(str(r[l]) for l in LABELS) for r in rows]
    texts_s = [cc.convert(x) for x in texts_t]
    counts_t, frag_t, chars_t = char_stats(texts_t, ntok)
    _, _, chars_s = char_stats(texts_s, ntok)
    top = sorted(frag_t, key=lambda c: -counts_t[c])[:20]

    labels = {l: ntok(l) for l in LABELS}

    # 印出摘要
    cols = ["mean", "median", "p90", "p99", "max"] + [f"over_{x}" for x in args.limits]
    print(f"=== 題目長度（token 數，validation {len(rows)} 題）===")
    print(f"{'':<8}" + "".join(f"{c:>10}" for c in cols))
    for name, st in [("繁體", stats_t), ("簡體", stats_s)]:
        cells = [f"{st[c]:>10}" if not c.startswith("over") else f"{st[c]:>10.1%}" for c in cols]
        print(f"{name:<8}" + "".join(cells))
    print(f"繁體比簡體多用 {extra:.1%} 的 token；每個 token 平均 {cpt_t:.2f} 字（繁）vs {cpt_s:.2f} 字（簡）")

    print("\n=== 中文字拆碎 ===")
    for name, cs in [("繁體", chars_t), ("簡體", chars_s)]:
        print(f"{name}：不重複 {cs['unique_chars']} 字，{cs['unique_fragmented_ratio']:.1%} 無法用單一 token 表示；"
              f"以出現次數計 {cs['occurrence_fragmented_ratio']:.1%}")
    print("最常被拆碎的繁體字（字:出現次數/token 數）：")
    print("  " + "  ".join(f"{c}:{counts_t[c]}/{frag_t[c]}" for c in top))

    print(f"\n選項字母的 token 數：{labels}")

    result = {
        "tokenizer": args.tokenizer, "n_questions": len(rows),
        "prompt_length_traditional": stats_t, "prompt_length_simplified": stats_s,
        "traditional_extra_tokens": round(extra, 4),
        "chars_per_token": {"traditional": round(cpt_t, 3), "simplified": round(cpt_s, 3)},
        "fragmented_chars_traditional": chars_t, "fragmented_chars_simplified": chars_s,
        "top_fragmented_traditional": [{"char": c, "count": counts_t[c], "tokens": frag_t[c]} for c in top],
        "label_tokens": labels,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n已存到 {args.output}")


if __name__ == "__main__":
    main()