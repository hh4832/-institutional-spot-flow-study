# 三大法人現貨買賣金額對 0050 未來報酬影響

本專案研究 `d0` 盤後公布的外資、投信、自營商現貨買進、賣出及買賣超金額，是否與 0050 自 `d1 open` 後的未來報酬相關。

## 核心定義

- Predictor：法人 1／5／10 交易日累積金額 ÷ 同期市場成交金額。
- 市場範圍：上市、上櫃及上市＋上櫃，分子與分母嚴格配對。
- 隔夜報酬：`C0→O1`，僅供統計解釋。
- 可交易報酬：`O1→C1/C2/C3/C5/C10`。
- 主要正規化：只使用 d0 及以前資料的 252／504／756 日 rolling PR 與 Z-score。
- 標準化輸出欄位明確標示視窗，例如 `rolling_252d_pr`、`rolling_504d_z`。
- 同時輸出各視窗完整可用樣本，以及自 756 日視窗可用日起算的共同樣本。
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

## 多滾動視窗輸出

- `analysis_dataset_rolling_252d_full.parquet`
- `analysis_dataset_rolling_504d_full.parquet`
- `analysis_dataset_rolling_756d_full.parquet`
- `analysis_dataset_common_window_start.parquet`
- `sample_availability.csv`

`full_available` 保留各標準化視窗本身所有可用日期，各 Predictor 依其
1／5／10 日累積期間完成暖機；`common_window_start` 則固定由最長的
756 日視窗下所有 Predictor 都完成暖機之日算起，
用來避免不同視窗因樣本起點不同而產生不公平比較。

## Phase 2：交易活動、轉折與機制驗證

Phase 2 是獨立的精簡研究模式，不會重跑或覆蓋舊版完整 grid。主要分析
756 日、以 504 日作穩健性驗證；252 日舊結果只作短期狀態比較。

新增特徵：

- `Gross = (Buy + Sell) / MarketTurnover`：法人雙邊交易活躍度。
- `DirectionalBalance = (Buy - Sell) / (Buy + Sell)`：活動中的買賣方向。
- `NetIntensity = abs(Buy - Sell) / (Buy + Sell)`：方向集中程度。
- Buy／Sell／Net／Gross 的 1／5／10 日非重疊區間變化。
- Net 正負轉換、Sell 連續下降、Buy 連續上升及高分位回落訊號。
- d0 已知的 0050 前期 1／5／10 日報酬、10／20 日波動率及成交額變化。

統計分析同時比較未控制與控制後 HAC 迴歸、分組平均、正報酬率、
中心化二次項及固定自由度 cubic spline。Confirmatory 與 Exploratory
假設分開進行 BH FDR；樣本數不足門檻的結果會在 FDR 前排除並另行輸出。

### Phase 2 合成資料測試

```bash
python run_synthetic_phase2.py
```

### Phase 2 真實資料

```bash
python main.py --phase2
```

主要輸出位於新的時間戳記資料夾，包括：

- `phase2_confirmatory_results.csv`
- `phase2_controlled_regressions.csv`
- `phase2_nonlinear_results.csv`
- `phase2_gross_activity_results.csv`
- `phase2_flow_change_results.csv`
- `phase2_turning_point_results.csv`
- `phase2_hit_rate_signals.csv`
- `phase2_subperiod_results.csv`
- `phase2_market_regime_results.csv`
- `phase2_data_regime_audit.csv`
- `phase2_reconstruction_discrepancies.csv`
- `phase2_excluded_results.csv`
- `phase2_summary.md`
- `phase2_run_metadata.json`

合成資料僅驗證計算流程，不可解讀為真實研究結果。真實資料仍須透過
FinLab登入後執行。
