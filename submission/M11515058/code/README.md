# HW1 重現說明（M11515058）

## 環境
開發與測試環境：Windows 11 + WSL2（Ubuntu 24.04）、NVIDIA RTX 4070 系列 16GB、
驅動 591.86、Python 3.11、PyTorch 2.14.0（CUDA 13.0）

```bash
conda create -n minigpt -c conda-forge --override-channels python=3.11 -y
conda activate minigpt
# PyTorch 請依 https://pytorch.org 選擇與驅動相符的 CUDA 版本安裝
# 開發時使用：pip install torch==2.14.0（PyPI 預設為 CUDA 13.0 版本）
pip install -r requirements.txt
```

## 執行
```bash
cd code
bash run.sh
```
在 RTX 4070 上約需 8.5 小時（pretrain 約 3 小時、SFT 約 5 小時、格式訓練約 15 分鐘），
建議在 tmux 中執行。最終模型輸出至 `output/model/`。

## 流程
1. 下載 MiniMind（commit f659b55）與訓練資料（revision 312afb4f），並比對檢查碼
2. Stage 1、2：MiniMind 預設設定的 pretrain 與 SFT（= baseline）
3. 以 `scripts/convert.py` 轉為 Hugging Face 格式
4. 以 `scripts/split_validation.py` 將 TMMLU+ validation 依科目分層切出 20% 保留集（seed 42）
5. Stage 3：以 `scripts/train_mc.py` 做選擇題格式訓練（選項輪替擴增、lr 1e-4、10 輪）

資料來源與處理細節見 `../data_manifest.md`。

## 檔案
- `run.sh`：完整重現流程
- `eval_tmmluplus.py`：助教提供的評估腳本（訓練腳本沿用其 prompt 與資料載入函式）
- `scripts/convert.py`、`split_validation.py`、`train_mc.py`：最終模型使用
- `scripts/analyze_tokenizer.py`、`add_t2s_normalizer.py`、`check_near_duplicates.py`：報告中的分析與實驗

## 預期結果與注意事項
GPU 運算有些微不確定性，即使隨機種子相同，重跑結果也不會完全一致。
我們以本 run.sh 實際完整重跑過一次（記錄見 `../logs/reproduction/`）：

| 模型 | 保留集 micro（原始 / 打亂） | practice set MACRO |
| --- | --- | --- |
| baseline（原始訓練） | 24.46 / 26.18 | 21.91 |
| 最終模型（繳交的 `model/`） | 28.54 / 27.90 | 26.24 |
| 重跑的 baseline | 24.89 / 26.39 | — |
| 重跑的最終模型 | 24.25 / 26.61 | 25.36 |

- baseline 重現良好：最後 loss 與保留集分數的差距都在誤差範圍內
- Stage 3 格式訓練的單次結果波動較大：重跑時 loss 走勢相同，但保留集未出現原本約 3 個百分點的進步；
  兩次結果的差異在統計上不顯著，詳見報告第 5 節
- 繳交的 `model/` 是原始訓練的最終模型，其訓練記錄為 `../logs/pretrain.log`、`sft.log`、`mc_sft.log`