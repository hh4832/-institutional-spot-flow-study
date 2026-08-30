# 三大法人現貨買賣金額對 0050 未來報酬影響

本專案研究 `d0` 盤後公布的外資、投信、自營商現貨買進、賣出及買賣超金額，是否與 0050 自 `d1 open` 後的未來報酬相關。

## 核心定義

- Predictor：法人 1／5／10 交易日累積金額 ÷ 同期市場成交金額。
- 市場範圍：上市、上櫃及上市＋上櫃，分子與分母嚴格配對。
- 隔夜報酬：`C0→O1`，僅供統計解釋。
- 可交易報酬：`O1→C1/C2/C3/C5/C10`。
- 主要正規化：只使用 d0 及以前資料的 252 日 rolling PR 與 Z-score。
- Global PR/Z：標記為 `lookahead_descriptive_only`，不得解釋為歷史即時訊號。
- 推論：Newey–West HAC；多日報酬使用 `maxlags=horizon-1`。
- 價格：收盤使用 `etl:adj_close`；若 `etl:adj_open` 不存在，開盤使用
  `原始開盤 × 還原收盤／原始收盤` 推導，以保持相同還原尺度。

## 安裝

```bash
python -m pip install -r requirements.txt
```

## 先執行測試

```bash
pytest -q
```

## 合成資料完整流程

不需要 FinLab Token，可驗證整個計算與輸出管線：

```bash
python run_synthetic.py
```

合成資料只用於程式驗證，不是研究結果。

## 真實資料

```bash
python main.py
```

程式會安全要求輸入 FinLab API Token。也可使用瀏覽器登入：

```bash
python main.py --browser-login
```

## Google Colab

開啟：

```text
notebooks/institutional_spot_flow_study_colab.ipynb
```

Notebook 流程：GitHub clone／安全更新、安裝套件、測試、FinLab 登入、執行研究、複製時間戳記輸出至 Google Drive。

## 重要限制

- 本研究是預測關係分析，不是完整資金配置回測。
- 不計算 CAGR 或策略 Sharpe ratio。
- 不把重疊 forward returns 視為獨立交易。
- 統計顯著不等於具有經濟意義或可交易性。
