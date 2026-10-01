"""把 MiniMind 的 .pth 權重轉成 Hugging Face（Qwen3）格式。

包裝 minimind/scripts/convert_model.py 的 convert_torch2transformers，
讓輸入、輸出路徑和模型大小可以用參數指定。
"""
import argparse
import os
import sys

HW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ~/hw1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--torch_path", required=True, help="要轉換的 .pth 檔")
    p.add_argument("--out_dir", required=True, help="輸出資料夾")
    p.add_argument("--minimind_dir", default=os.path.join(HW_DIR, "minimind"))
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    args = p.parse_args()

    # 細節 3：先轉成完整路徑，因為下面會切換資料夾
    torch_path = os.path.abspath(args.torch_path)
    out_dir = os.path.abspath(args.out_dir)
    minimind_dir = os.path.abspath(args.minimind_dir)
    scripts_dir = os.path.join(minimind_dir, "scripts")

    if not os.path.isfile(torch_path):
        sys.exit(f"找不到檔案：{torch_path}")
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        sys.exit(f"輸出資料夾已存在且不是空的，為避免覆蓋而停止：{out_dir}")

    # 讓 Python 找得到 MiniMind 的程式
    sys.path.insert(0, minimind_dir)
    sys.path.insert(0, scripts_dir)
    # 細節 1：作者用 '../model/' 讀 tokenizer，所以切到 scripts/ 執行
    os.chdir(scripts_dir)

    import convert_model
    from model.model_minimind import MiniMindConfig

    # 細節 2：替作者的函式準備好模型設定
    convert_model.lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=8192,  # 與作者腳本相同
        use_moe=bool(args.use_moe),
    )
    convert_model.convert_torch2transformers(torch_path, out_dir)

    # 轉完自動檢查：用標準方式重新載入
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(out_dir)
    tok = AutoTokenizer.from_pretrained(out_dir)
    n = sum(q.numel() for q in model.parameters())
    print(f"[check] {type(model).__name__} 載入成功，參數量 {n / 1e6:.2f}M，詞表大小 {len(tok)}")


if __name__ == "__main__":
    main()