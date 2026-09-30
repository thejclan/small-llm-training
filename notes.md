# HW1 實驗記錄

## 環境
- 主機：Windows + WSL2（Ubuntu 24.04）
- GPU：RTX 4070 系列 16GB，驅動 591.86
- Python 3.11（conda 環境 minigpt）
- PyTorch 2.14.0+cu130
- MiniMind commit：f659b55761b754d306bd140573493a6543cafd7f
- 額外安裝 opencc-python-reimplemented（tokenizer 分析用）

## 待確認事項
- 9/29 寄信問助教 → 9/30 教授已回覆（見下方）
- 仍未確認：繳交檔名 hw_X 的 X

## 實驗
### smoke test（9/29）
- 指令：python eval_tmmluplus.py eval --model_path ./minimind-3 --limit 20 --output results/smoke.json
- 結果：macro 0.0848，truncated 0，流程正常

## 環境建置過程（9/29）
- 主機是 Windows，改用 WSL2 + Ubuntu 24.04
- WSL DNS 無法解析（與 Tailscale 衝突），改用 8.8.8.8 / 1.1.1.1 並停用自動產生 resolv.conf
- conda 環境用 conda-forge 建立（避開 Anaconda 使用條款），未裝 torchvision
- MiniMind 的 requirements.txt 沒有指定 PyTorch 版本，直接安裝

## 資料下載（9/29）
- 來源：Hugging Face jingyaogong/minimind_dataset
- pretrain_t2t_mini.jsonl（1.2G）、sft_t2t_mini.jsonl（1.7G）

## AI 使用記錄
- 使用 Claude 協助：環境建置（WSL、DNS、conda）、閱讀作業文件、訓練指令與記錄檔設定、模型轉換腳本（scripts/convert.py）、validation 切分腳本（scripts/split_validation.py）

## Baseline 訓練（9/29 23:03 開始）
- 設定：MiniMind 預設（hidden 768、8 層、dense、pretrain 與 SFT 各 2 輪、seed 42）
- 指令：在 minimind/trainer 下執行
  python -u train_pretrain.py 2>&1 | tee ~/hw1/logs/baseline_pretrain.log && python -u train_full_sft.py 2>&1 | tee ~/hw1/logs/baseline_sft.log
- 模型參數：63.91M
- 每輪 pretrain 約 1.5 小時；GPU 使用率 99%、記憶體 7.3GB
- 結束時間：9/30 07:04（總計約 8.0 GPU 小時）
  - Pretrain：23:03 → 01:57（約 2.9 小時），loss 7.52 → 約 1.7
  - SFT：01:57 → 07:04（約 5.1 小時），最後 loss 約 1.7
- 產出：out/pretrain_768.pth、out/full_sft_768.pth（各 132M）
- 記錄檔：logs/baseline_pretrain.log、logs/baseline_sft.log、logs/baseline_time.txt
- 已備份至 models/baseline_pth/

## Baseline 評估（9/30）
- 轉換：python scripts/convert.py --torch_path models/baseline_pth/full_sft_768.pth --out_dir models/baseline
  - 轉成 Qwen3ForCausalLM 格式，63.91M 參數，詞表 6400
- practice set：MACRO 0.2191，micro 0.2359 ± 0.0095（18.2 秒，truncated 0）
  - results/baseline_practice.json、logs/baseline_eval_practice.log
- 確認流程無誤：
  - SFT 最初 loss 約 2.1 → 有接在 pretrain 之後訓練
  - 作者的 minimind-3 在 practice set：MACRO 0.2473，micro 0.2429 ± 0.0096
    → 這個大小的模型本來就接近亂猜，baseline 在正常範圍（可放進報告）

## 重要發現
- 評估腳本產生私人評分集時會打亂選項順序，practice set 則沒有打亂
  → 靠偏好某個字母得到的分數，在排行榜上不會成立
- HW1 第 7 節：擊敗自己的 baseline，排行榜部分至少得 35 分（該項滿分）
- 助教評分應使用預設選項（無 chat template、無範例題）
  → 選擇題格式訓練要對齊評估腳本的預設 prompt

## 教授回覆（9/30）
1. 不能用 test set 訓練。practice set（test 的 10%）用來產生統一的 baseline 分數；
   兩種分數都可以在報告中說明；擊敗自己的第一份 baseline 可得 15 分。
2. 可以自行調整 validation 保留集來觀察模型與 baseline 的變化；
   對照 validation 保留集與 practice set 是有趣的題目。不要直接用 test set 做評估。
3. 20% 不強制，比例可自由選擇。

## 評估規劃
- 官方 baseline 分數：practice set（評估腳本預設）
- 開發時的比較與決策：只看 validation 保留集
- 最終模型：兩者都評估並報告
- 保留比例：20%（依科目分層、固定隨機種子）
- 報告加分方向：比較各模型在保留集與 practice set 的分數是否同步變化
- 保留集檔案：data/val_holdout.jsonl、data/val_holdout_shuffled.jsonl（打亂選項，模擬私人評分集）
- 可訓練：data/val_train.jsonl
- 切分腳本：scripts/split_validation.py（seed 42）

## Validation 切分（9/30）
- 指令：python scripts/split_validation.py（fraction 0.2、seed 42）
- validation 實際 2,197 題、66 科（HW1.docx 寫 2,202，以實際為準）；重複題文字 6 題
- val_holdout.jsonl 466 題（sha256[:16]=0545cf5ce7709f51）
- val_holdout_shuffled.jsonl 466 題（015bf7ce64310d15）
- val_train.jsonl 1,731 題（6f1593cdafb02d81）
- 正確答案分布：原始 A 24.5 / B 26.8 / C 27.5 / D 21.2；打亂後 A 26.2 / B 24.5 / C 22.5 / D 26.8

## Baseline 保留集評估（9/30）
- 原始順序：MACRO 0.2249，micro 0.2446 ± 0.0199
- 打亂選項：MACRO 0.2521，micro 0.2618 ± 0.0204
- 保留集誤差約 ±2 個百分點，4 點以內的差距難以和雜訊區分

