#!/usr/bin/env bash
# HW1（M11515058）：從零重現最終模型
# 用法：依 README.md 建好環境後，在 code/ 中執行 bash run.sh
# 輸出：output/model（最終模型）、output/baseline、output/logs
set -euo pipefail  # 任何一步失敗就立刻停止

cd "$(dirname "$0")"  # 一律以 code/ 為工作目錄，所有路徑都是相對路徑
OUT_DIR="output"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

MINIMIND_REPO="https://github.com/jingyaogong/minimind"
MINIMIND_COMMIT="f659b55761b754d306bd140573493a6543cafd7f"
DATA_REPO="jingyaogong/minimind_dataset"
DATA_REVISION="312afb4f76391145c6902f765bb51691c09a12f5"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/run_time.log"; }

check_prefix() {  # 比對檔案 SHA-256 的前 16 碼
  local actual
  actual=$(sha256sum "$1" | cut -c1-16)
  [ "$actual" = "$2" ] || { echo "檢查碼不符：$1（預期 $2，實際 $actual）"; exit 1; }
}

# 1. MiniMind 原始碼（固定版本）
log "步驟 1：下載 MiniMind 原始碼"
[ -d minimind ] || git clone "$MINIMIND_REPO" minimind
git -C minimind checkout --quiet "$MINIMIND_COMMIT"

# 2. 訓練資料（固定版本，並比對檢查碼）
log "步驟 2：下載 MiniMind 資料"
hf download "$DATA_REPO" --repo-type dataset --revision "$DATA_REVISION" \
  --include "pretrain_t2t_mini.jsonl" "sft_t2t_mini.jsonl" --local-dir minimind/dataset
(cd minimind/dataset && sha256sum -c - << 'EOF'
6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c  pretrain_t2t_mini.jsonl
abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518  sft_t2t_mini.jsonl
EOF
)

# 3. Stage 1：pretrain（MiniMind 預設設定）
log "步驟 3：Stage 1 pretrain"
(cd minimind/trainer && python -u train_pretrain.py) 2>&1 | tee "$LOG_DIR/pretrain.log"

# 4. Stage 2：SFT（MiniMind 預設設定）
log "步驟 4：Stage 2 SFT"
(cd minimind/trainer && python -u train_full_sft.py) 2>&1 | tee "$LOG_DIR/sft.log"

# 5. 轉成 Hugging Face 格式（= baseline）
log "步驟 5：轉換 baseline"
rm -rf "$OUT_DIR/baseline"
python scripts/convert.py --torch_path minimind/out/full_sft_768.pth --out_dir "$OUT_DIR/baseline"

# 6. 切分 TMMLU+ validation（依科目分層 20%，seed 42）
log "步驟 6：切分 validation"
python scripts/split_validation.py --fraction 0.2 --seed 42 --out_dir data
check_prefix data/val_holdout.jsonl 0545cf5ce7709f51
check_prefix data/val_holdout_shuffled.jsonl 015bf7ce64310d15
check_prefix data/val_train.jsonl 6f1593cdafb02d81

# 7. Stage 3：選擇題格式訓練 → 最終模型
log "步驟 7：Stage 3 格式訓練"
rm -rf "$OUT_DIR/model"
python -u scripts/train_mc.py --base_model "$OUT_DIR/baseline" --out_dir "$OUT_DIR/model" \
  --augment rotate --lr 1e-4 --epochs 10 --batch_size 16 --seed 42 2>&1 | tee "$LOG_DIR/mc_sft.log"

log "完成：最終模型位於 $OUT_DIR/model"