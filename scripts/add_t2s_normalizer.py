"""替模型資料夾的 tokenizer 加上「繁體轉簡體」的正規化規則。

模型權重不動，只修改 tokenizer.json：切字之前先把繁體字逐字換成簡體字，
讓以簡體資料訓練的模型讀到它熟悉的字。對照表只由 OpenCC（t2s）產生，
不使用任何 TMMLU+ 資料。

用法：
  python scripts/add_t2s_normalizer.py --src models/baseline --dst models/baseline_t2s
"""
import argparse
import os
import shutil
import statistics
import sys
import time

HW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ~/hw1
sys.path.insert(0, HW_DIR)
from eval_tmmluplus import LABELS, SUBJECTS, build_prompt, load_hf_split

from opencc import OpenCC
from tokenizers import Tokenizer, normalizers
from transformers import AutoTokenizer


def build_mapping():
    """逐字丟給 OpenCC，保留轉換後不同的字（CJK 基本區 U+4E00–U+9FFF）。"""
    cc = OpenCC("t2s")
    mapping = {}
    for cp in range(0x4E00, 0xA000):
        t = chr(cp)
        s = cc.convert(t)
        if s != t and len(s) == 1:
            mapping[t] = s
    return mapping


def ordered_rules(mapping):
    """規則會依序套用。若某條規則轉出來的字，又是另一條規則的來源字，
    就把後者排在前面，避免同一個字被連續轉換兩次。"""
    targets = set(mapping.values())
    first = [(t, s) for t, s in mapping.items() if t in targets]
    rest = [(t, s) for t, s in mapping.items() if t not in targets]
    return first + rest, len(first)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="原本的模型資料夾")
    p.add_argument("--dst", required=True, help="輸出的新模型資料夾")
    args = p.parse_args()
    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    if os.path.exists(dst):
        sys.exit(f"輸出資料夾已存在，為避免覆蓋而停止：{dst}")

    # 1. 建立對照表
    rules, n_chain = ordered_rules(build_mapping())
    print(f"繁→簡對照：{len(rules)} 字（其中 {n_chain} 條調整順序，避免連鎖轉換）")

    # 2. 複製模型資料夾，只修改 tokenizer.json
    shutil.copytree(src, dst)
    path = os.path.join(dst, "tokenizer.json")
    tk = Tokenizer.from_file(path)
    steps = [tk.normalizer] if tk.normalizer is not None else []  # 保留原本的正規化規則
    steps += [normalizers.Replace(t, s) for t, s in rules]
    tk.normalizer = normalizers.Sequence(steps)
    tk.save(path)

    # 3. 檢查
    old = AutoTokenizer.from_pretrained(src)
    new = AutoTokenizer.from_pretrained(dst)  # 用標準方式重新載入，確認規則有生效
    ntok = lambda tok, x: len(tok(x, add_special_tokens=False)["input_ids"])

    labels = {l: ntok(new, l) for l in LABELS}
    assert all(n == 1 for n in labels.values()), f"選項字母不是單一 token：{labels}"
    print(f"選項字母的 token 數：{labels}")

    sample = "為什麼關於國學的會議時間與地點還沒確定？"
    ids = new(sample, add_special_tokens=False)["input_ids"]
    print(f"\n範例：{sample}")
    print(f"  token 數：{ntok(old, sample)} → {len(ids)}")
    print(f"  模型實際看到：{new.decode(ids)}")

    rows = load_hf_split("validation", list(SUBJECTS))
    prompts = [build_prompt(r, [], new, False) for r in rows]

    def measure(tok):
        start = time.perf_counter()
        lens = [ntok(tok, x) for x in prompts]
        return lens, (time.perf_counter() - start) / len(prompts) * 1000

    len_old, ms_old = measure(old)
    len_new, ms_new = measure(new)
    print(f"\nvalidation {len(prompts)} 題的平均 token 數：{statistics.mean(len_old):.1f} → {statistics.mean(len_new):.1f}")
    print(f"切字速度（每題）：{ms_old:.2f} ms → {ms_new:.2f} ms")
    print(f"\n完成：{dst}")


if __name__ == "__main__":
    main()