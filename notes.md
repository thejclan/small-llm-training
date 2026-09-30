# HW1 實驗記錄

## 環境
- 主機：Windows + WSL2（Ubuntu 24.04）
- GPU：RTX 4070 系列 16GB，驅動 591.86
- Python 3.11（conda 環境 minigpt）
- PyTorch 2.14.0+cu130
- MiniMind commit：f659b55761b754d306bd140573493a6543cafd7f
- 額外安裝 opencc-python-reimplemented（tokenizer 分析用）；套件版本清單存於 requirements.txt

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

## Tokenizer 分析（9/30）
- 指令：python scripts/analyze_tokenizer.py（只用 validation 2,197 題，預設 prompt 格式）
- 結果：results/tokenizer_analysis.json
- 題目長度（token）：繁體 平均 194.7、中位數 168、p90 296、最長 1726
  - 超過 340：5.3%；超過 768：1.1%（簡體對照為 2.2%、0.3%）
- 繁體比簡體多用 40.9% 的 token；每個 token 平均 0.84 字（繁）vs 1.19 字（簡）
- 被拆碎的中文字（以出現次數計）：繁體 38.6% vs 簡體 8.2%
  - 不重複字：繁體 3445 字中 76.1% 被拆碎；簡體 3323 字中 61.5%
  - 最常被拆碎：為、於、時、關、與、會、險、對、動、體、應、國、發、業、敘、經、產、學、確、種
- 選項字母 A/B/C/D 都是單一 token
- 結論：
  - 序列長度不是主要問題（僅約 5% 題目超過 pretrain 長度）→ 方向 4 調序列長度效益有限
  - 繁體字才是瓶頸：常用字的繁體版本被拆成陌生碎片，模型等於讀到近四成亂碼
- 下一步：A. tokenizer 加入繁轉簡（免訓練）→ 視結果決定是否 B. 繁體化 MiniMind 資料續訓 → 在最佳版本上做方向 3

## 方向 2：tokenizer 繁轉簡（9/30）
- 指令：python scripts/add_t2s_normalizer.py --src models/baseline --dst models/baseline_t2s
- 對照表 3,553 字（OpenCC t2s 逐字產生，不用 TMMLU+ 資料）；A/B/C/D 仍為單一 token
- 平均 token 數 194.7 → 138.2；切字每題 0.11 → 1.96 ms
- 保留集：原始 MACRO 0.2589 / micro 0.2468；打亂 0.2299 / 0.2425
- 結論：無可觀察的進步（誤差 ±2）。模型本來就不會答題，看懂題目也無法利用
- 決定：今晚不跑 B（繁體化資料續訓），效益低

## 方向 3：選擇題格式訓練（9/30）
- 腳本：scripts/train_mc.py（prompt 與計分直接沿用評估腳本；只對答案字母計算 loss）
- 資料：TMMLU+ train 317 題（HW1.docx 寫 321）+ val_train 1,731 題，排除與保留集重複 2 題 → 2,046 題
- 驗證：訓練前分數與官方評估完全一致 → 格式與計分已對齊
- 實驗 2（baseline + rotate，lr 2e-5，2 輪，2.6 分鐘）
  - 原始順序 micro 24.46 → 28.11 → 29.61；打亂選項 26.18 → 24.25 → 24.25
  - loss 1.62 → 1.39 ≈ ln(4)：學會格式，但沒學會分辨選項內容
  - 原始順序的進步未出現在打亂選項 → 可能只是字母偏好的運氣

### 2×2 實驗結果（epoch 2，保留集 micro：原始 / 打亂，最後 loss）
- 實驗 1 不打亂：20.39 / 21.24，loss 0.0015 → 背下訓練題答案，過度擬合
- 實驗 2 打亂：29.61 / 24.25，loss 1.39 → 連訓練題都無法擬合
- 實驗 3 繁轉簡 + 不打亂：25.54 / 22.75，loss 0.0021
- 實驗 4 繁轉簡 + 打亂：27.90 / 21.24，loss 1.39
- 結論：
  - 打亂選項上皆未超越 baseline（26.18）
  - 瓶頸：模型無法把答案內容對應到選項位置
  - 打亂版本「原始升、打亂降」推測為字母偏好（待驗證）
  - 繁轉簡無一致效果 → 後續不採用
- 下一步：提高學習率（1e-4、3e-4，3 輪）
- 判斷標準（事先訂定）：原始與打亂的平均，需超過 baseline 平均 25.32 至少 2 個百分點

### 學習率實驗（rotate，3 輪，保留集 micro：原始 / 打亂）
- lr 1e-4：ep1 26.82 / 24.46；ep2 27.47 / 22.53；ep3 27.47 / 24.68，loss 約 1.39
- lr 3e-4：ep1、ep2 與 lr 1e-4 完全相同；ep3 24.46 / 26.39，loss 約 1.39
- 皆未通過標準（平均需 > 27.32）

### 關鍵發現：模型每題都選同一個字母
- 保留集答案題數：原始 A114 / B125 / C128 / D99；打亂 A122 / B114 / C105 / D125
- baseline 分數（114、122）= A 的題數 → baseline 每題都選 A
- lr 實驗 ep1（125、114）= B；ep2（128、105）= C → 訓練後改為固定選 B 或 C
- 「原始順序上升」只是 C 在原始保留集比較常是答案
- 排行榜打亂選項 → 固定選一個字母的模型期望值 25%