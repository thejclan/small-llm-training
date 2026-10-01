# Data Manifest

## 1. 訓練資料

最終模型分三階段訓練：Stage 1 pretrain → Stage 2 SFT（兩者合為 baseline）→ Stage 3 選擇題格式訓練。

### 1.1 MiniMind pretrain 資料（Stage 1）
- 來源：https://huggingface.co/datasets/jingyaogong/minimind_dataset
- 版本（commit）：312afb4f76391145c6902f765bb51691c09a12f5
- 檔案：pretrain_t2t_mini.jsonl
- 大小：1,241,043,656 bytes，1,270,238 筆
- SHA-256：6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c
- 語言：以簡體中文為主
- 處理：未修改，依 MiniMind 預設流程使用（train_pretrain.py，max_seq_len 340）

### 1.2 MiniMind SFT 資料（Stage 2）
- 來源：同上，版本相同
- 檔案：sft_t2t_mini.jsonl
- 大小：1,739,201,170 bytes，905,718 筆
- SHA-256：abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518
- 語言：以簡體中文為主
- 處理：未修改，依 MiniMind 預設流程使用（train_full_sft.py，max_seq_len 768）

### 1.3 TMMLU+ train split（Stage 3）
- 來源：https://huggingface.co/datasets/ikala/tmmluplus
- 版本：v1.1（與 eval_tmmluplus.py 相同）
- 範圍：train split，66 科，共 317 題

### 1.4 TMMLU+ validation 的訓練部分（Stage 3）
- 來源與版本：同 1.3
- 範圍：validation split 共 2,197 題，以 scripts/split_validation.py
  依科目分層切出 80%（seed 42），共 1,731 題
- 檔案：data/val_train.jsonl（由腳本產生，SHA-256 前 16 碼 6f1593cdafb02d81）

### 1.3 與 1.4 的處理方式（scripts/train_mc.py）
1. 合併 1.3 與 1.4 共 2,048 題，排除與保留集（2.1）文字完全相同的 2 題，剩 2,046 題
2. 以 eval_tmmluplus.py 的 build_prompt 組成與評估完全相同的 prompt（0-shot、不使用 chat template）
3. 選項輪替擴增：每題產生 4 個版本，正確答案依序位於 A、B、C、D，共 8,184 筆
4. 只對答案字母計算 loss

## 2. 評估資料（未用於訓練）

### 2.1 TMMLU+ validation 保留集
- 來源與版本：同 1.3
- 範圍：validation 依科目分層切出 20%（seed 42），共 466 題，66 科
- 檔案：data/val_holdout.jsonl（SHA-256 前 16 碼 0545cf5ce7709f51）
- 打亂選項版：data/val_holdout_shuffled.jsonl（015bf7ce64310d15），題目相同、選項順序打亂，模擬私人評分集
- 用途：開發過程中的所有比較與決策

### 2.2 TMMLU+ practice set
- 來源：eval_tmmluplus.py 預設的 practice set（test split 固定 10% 樣本，2,001 題）
- 用途：僅用於評估 baseline 與最終模型各一次；未用於訓練、調參或模型選擇，也未查看題目內容

## 3. 汙染防範
- 未使用 TMMLU+ test split 的任何資料訓練；未使用其他考題資料集
- 保留集與訓練資料：排除文字完全相同的題目；另以 3-gram 相似度檢查，
  相似度 ≥ 0.7 者 5 題（標點差異或共用閱讀文章），排除後評估結果不變
  （scripts/check_near_duplicates.py）
- 除上述資料外，未使用任何其他外部資料

## 4. 分析用工具（不屬於訓練資料）
- opencc-python-reimplemented：僅用於 tokenizer 分析與繁轉簡實驗，未用於最終模型