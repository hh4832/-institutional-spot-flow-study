from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from config import PR_LABELS, Z_LABELS, StudyConfig
from data_loader import RawData
from features import (
    add_normalizations,
    build_nonoverlapping_changes,
    build_phase2_activity_predictors,
    build_prior_market_controls,
    build_turning_point_signals,
    build_turnover,
    reconstruct_institution_flows,
)
from grouping import assign_pr_group, assign_z_group
from phase2_statistics import (
    add_separate_fdr,
    hac_signal_regression,
    nonlinear_hac_regressions,
    vif_table,
)
from reporting import build_manifest, timestamped_output_directory, write_json
from returns import build_forward_returns
from study_statistics import summarize_group


CONFIRMATORY_SPECS = (
    ("combined__foreign__sell__10d", "z", (504, 756), "Z_GE_P2_5", (10,)),
    ("otc__total_institutional__sell__1d", "pr", (504,), "PR_05_20", (10,)),
    ("otc__total_institutional__sell__1d", "z", (504,), "Z_M1_5_M0_5", (10,)),
    ("otc__total_institutional__sell__5d", "pr", (504,), "PR_05_20", (5, 10)),
    ("otc__dealer__sell__10d", "z", (504, 756), "Z_M2_5_M1_5", (10,)),
    ("otc__foreign__buy__5d", "z", (756,), "Z_P0_5_P1_5", (10,)),
    ("otc__foreign__buy__10d", "z", (756,), "Z_P0_5_P1_5", (10,)),
    ("otc__total_institutional__sell__5d", "pr", (756,), "PR_60_80", (10,)),
    ("otc__total_institutional__sell__10d", "pr", (756,), "PR_60_80", (10,)),
    ("listed__foreign__net__5d", "pr", (504, 756), "PR_95_100", (10,)),
    ("listed__dealer__net__10d", "pr", (504, 756), "PR_95_100", (10,)),
    ("listed__dealer__net__5d", "z", (504,), "Z_P1_5_P2_5", (10,)),
)


def _return_column(horizon: int) -> str:
    return f"return_o1_c{horizon}"


def _hac_lag(horizon: int) -> int:
    return max(horizon - 1, 0)


def _feature_dictionary(columns: list[str]) -> pd.DataFrame:
    rows = []
    definitions = {
        "buy": "sum(BuyAmount) / sum(MarketTurnover)",
        "sell": "sum(SellAmount) / sum(MarketTurnover)",
        "net": "sum(BuyAmount-SellAmount) / sum(MarketTurnover)",
        "gross": "sum(BuyAmount+SellAmount) / sum(MarketTurnover)",
        "directional_balance": "sum(Buy-Sell) / sum(Buy+Sell)",
        "net_intensity": "abs(sum(Buy-Sell)) / sum(Buy+Sell)",
    }
    for column in columns:
        scope, institution, feature, window = column.split("__")
        base = feature.removesuffix("_change")
        rows.append({
            "feature": column,
            "market_scope": scope,
            "institution": institution,
            "feature_type": feature,
            "accumulation_window": int(window.removesuffix("d")),
            "formula": definitions.get(base, "pre-specified transformation"),
            "change_definition": (
                "current k-day level minus prior non-overlapping k-day level"
                if feature.endswith("_change") else "level"
            ),
            "known_at": "d0_after_close",
        })
    return pd.DataFrame(rows)


def _controls_for_scope(controls: pd.DataFrame, scope: str) -> pd.DataFrame:
    return controls[[
        "prior_return_10d",
        "prior_volatility_20d",
        f"{scope}__market_turnover_change_10d",
    ]].rename(columns={f"{scope}__market_turnover_change_10d": "market_turnover_change_10d"})


def _candidate_memberships(
    normalized: pd.DataFrame,
    available_windows: tuple[int, ...],
    available_horizons: tuple[int, ...] = (1, 5, 10),
    include_252_sensitivity: bool = False,
) -> list[dict]:
    rows = []
    window_map = {504: min(available_windows), 756: max(available_windows)}
    for predictor, method, windows, label, horizons in CONFIRMATORY_SPECS:
        requested_windows = tuple(dict.fromkeys(
            ((252,) if include_252_sensitivity and 252 in available_windows else ())
            + tuple(windows)
        ))
        for requested_window in requested_windows:
            window = (
                requested_window
                if requested_window in available_windows
                else window_map[requested_window]
            )
            column = f"{predictor}__rolling_{window}d_{method}"
            if column not in normalized:
                continue
            groups = assign_pr_group(normalized[column]) if method == "pr" else assign_z_group(normalized[column])
            for horizon in horizons:
                if horizon not in available_horizons:
                    continue
                rows.append({
                    "candidate_id": f"{predictor}__rolling_{window}d_{method}__{label}__O1_C{horizon}",
                    "predictor": predictor,
                    "normalization_type": f"rolling_{method}",
                    "normalization_window": window,
                    "requested_normalization_window": requested_window,
                    "group": label,
                    "horizon": horizon,
                    "membership": groups.eq(label),
                    "continuous": normalized[column],
                    "market_scope": predictor.split("__")[0],
                })
    return rows


def _confirmatory_results(
    candidates: list[dict], returns: pd.DataFrame, controls: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for item in candidates:
        outcome = returns[_return_column(item["horizon"])]
        lag = _hac_lag(item["horizon"])
        group_summary = summarize_group(outcome, item["membership"], lag)
        if group_summary:
            rows.append({
                "hypothesis_family": "confirmatory_group_mean",
                "candidate_id": item["candidate_id"],
                "predictor": item["predictor"],
                "normalization_type": item["normalization_type"],
                "normalization_window": item["normalization_window"],
                "requested_normalization_window": item["requested_normalization_window"],
                "group": item["group"],
                "return_type": f"O1_to_C{item['horizon']}",
                "model": "group_uncontrolled",
                "raw_p_value": group_summary["unconditional_p_value"],
                **group_summary,
            })
        scope_controls = _controls_for_scope(controls, item["market_scope"])
        for signal_type, signal in (
            ("group_dummy", item["membership"].astype(float)),
            ("continuous", item["continuous"]),
        ):
            uncontrolled = hac_signal_regression(signal, outcome, lag)
            controlled = hac_signal_regression(signal, outcome, lag, scope_controls)
            for model, result in (("uncontrolled", uncontrolled), ("controlled", controlled)):
                if result:
                    rows.append({
                        "hypothesis_family": f"confirmatory_{signal_type}_{model}",
                        "candidate_id": item["candidate_id"],
                        "predictor": item["predictor"],
                        "normalization_type": item["normalization_type"],
                        "normalization_window": item["normalization_window"],
                        "requested_normalization_window": item["requested_normalization_window"],
                        "group": item["group"],
                        "return_type": f"O1_to_C{item['horizon']}",
                        "model": f"{signal_type}_{model}",
                        **result,
                    })
    return pd.DataFrame(rows)


def _nonlinear_results(
    normalized: pd.DataFrame,
    returns: pd.DataFrame,
    windows: tuple[int, ...],
    horizons: tuple[int, ...],
    spline_df: int,
) -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"__(gross|directional_balance|net_intensity|buy_change|sell_change|net_change|gross_change)__")
    columns = [
        c for c in normalized
        if pattern.search(c) and re.search(r"__rolling_(?:" + "|".join(map(str, windows)) + r")d_z$", c)
    ]
    for column in columns:
        window = int(re.search(r"__rolling_(\d+)d_z$", column).group(1))
        for horizon in horizons:
            outcome = returns[_return_column(horizon)]
            for result in nonlinear_hac_regressions(
                normalized[column], outcome, _hac_lag(horizon), spline_df
            ):
                rows.append({
                    "hypothesis_family": "exploratory_nonlinear",
                    "predictor": re.sub(r"__rolling_\d+d_z$", "", column),
                    "normalization_type": "rolling_z",
                    "normalization_window": window,
                    "return_type": f"O1_to_C{horizon}",
                    **result,
                })
    return pd.DataFrame(rows)


def _nonlinear_curve_data(
    normalized: pd.DataFrame,
    returns: pd.DataFrame,
    windows: tuple[int, ...],
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """Export decile points for auditable visualization of nonlinear shapes."""
    rows = []
    wanted = re.compile(r"__(gross|directional_balance|net_intensity|buy_change|sell_change|net_change|gross_change)__")
    for column in normalized:
        match = re.search(r"__rolling_(\d+)d_z$", column)
        if not wanted.search(column) or not match or int(match.group(1)) not in windows:
            continue
        for horizon in horizons:
            frame = pd.concat([
                normalized[column].rename("predictor_value"),
                returns[_return_column(horizon)].rename("return"),
            ], axis=1).dropna()
            if len(frame) < 20 or frame.predictor_value.nunique() < 10:
                continue
            frame["decile"] = pd.qcut(frame.predictor_value, 10, labels=False, duplicates="drop")
            summary = frame.groupby("decile", observed=True).agg(
                observation_count=("return", "size"),
                predictor_mean=("predictor_value", "mean"),
                mean_return=("return", "mean"),
                positive_rate=("return", lambda x: x.gt(0).mean()),
            ).reset_index()
            summary["predictor"] = re.sub(r"__rolling_\d+d_z$", "", column)
            summary["normalization_window"] = int(match.group(1))
            summary["return_type"] = f"O1_to_C{horizon}"
            rows.extend(summary.to_dict("records"))
    return pd.DataFrame(rows)


def _group_feature_results(
    normalized: pd.DataFrame,
    returns: pd.DataFrame,
    windows: tuple[int, ...],
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    rows = []
    wanted = re.compile(r"__(gross|directional_balance|net_intensity|buy_change|sell_change|net_change|gross_change)__")
    for column in normalized:
        if not wanted.search(column):
            continue
        match = re.search(r"__rolling_(\d+)d_(pr|z)$", column)
        if not match or int(match.group(1)) not in windows:
            continue
        window, method = int(match.group(1)), match.group(2)
        is_change = "_change__" in column
        if (is_change and method != "z") or (not is_change and method != "pr"):
            continue
        groups = assign_pr_group(normalized[column]) if method == "pr" else assign_z_group(normalized[column])
        labels = (
            ["PR_00_05", "PR_05_20", "PR_80_95", "PR_95_100"]
            if method == "pr"
            else ["Z_LT_M2_5", "Z_M2_5_M1_5", "Z_P1_5_P2_5", "Z_GE_P2_5"]
        )
        base = re.sub(r"__rolling_\d+d_(?:pr|z)$", "", column)
        for horizon in horizons:
            outcome = returns[_return_column(horizon)]
            for label in labels:
                summary = summarize_group(outcome, groups.eq(label), _hac_lag(horizon))
                if summary:
                    rows.append({
                        "hypothesis_family": "exploratory_group",
                        "predictor": base,
                        "normalization_type": f"rolling_{method}",
                        "normalization_window": window,
                        "group": label,
                        "return_type": f"O1_to_C{horizon}",
                        "raw_p_value": summary["unconditional_p_value"],
                        **summary,
                    })
    return pd.DataFrame(rows)


def _turning_results(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    rows = []
    for column in signals:
        for horizon in horizons:
            summary = summarize_group(
                returns[_return_column(horizon)], signals[column], _hac_lag(horizon)
            )
            if summary:
                rows.append({
                    "hypothesis_family": "exploratory_turning_point",
                    "predictor": column,
                    "return_type": f"O1_to_C{horizon}",
                    "raw_p_value": summary["unconditional_p_value"],
                    **summary,
                })
    return pd.DataFrame(rows)


def _temporal_breakdowns(candidates: list[dict], returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods = {
        "2009_2013": (2009, 2013), "2014_2017": (2014, 2017),
        "2018_2019": (2018, 2019), "2020_2022": (2020, 2022),
        "2023_latest": (2023, 9999),
    }
    period_rows, loo_rows = [], []
    for item in candidates:
        frame = pd.concat([
            item["membership"].rename("membership"),
            returns[_return_column(item["horizon"])].rename("return"),
        ], axis=1).dropna()
        frame = frame.loc[frame.membership]
        for period, (start, end) in periods.items():
            years = frame.index.year
            values = frame.loc[(years >= start) & (years <= end), "return"]
            period_rows.append({
                "candidate_id": item["candidate_id"], "period": period,
                "observation_count": len(values), "mean_return": values.mean(),
                "positive_rate": values.gt(0).mean() if len(values) else np.nan,
            })
        for year in sorted(frame.index.year.unique()):
            values = frame.loc[frame.index.year != year, "return"]
            loo_rows.append({
                "candidate_id": item["candidate_id"], "excluded_year": year,
                "observation_count": len(values), "mean_return": values.mean(),
                "positive_rate": values.gt(0).mean() if len(values) else np.nan,
            })
    return pd.DataFrame(period_rows), pd.DataFrame(loo_rows)


def _market_regime_breakdown(
    candidates: list[dict], returns: pd.DataFrame, close: pd.Series
) -> pd.DataFrame:
    """Use only d0-known prices to define trend and expanding volatility regimes."""
    close = pd.to_numeric(close, errors="coerce").reindex(returns.index)
    daily = close.pct_change(fill_method=None)
    ma200 = close.rolling(200, min_periods=200).mean()
    distance = close.div(ma200) - 1
    trend = pd.Series("sideways", index=returns.index, dtype=object)
    trend.loc[distance.gt(0.05)] = "bull"
    trend.loc[distance.lt(-0.05)] = "bear"
    vol20 = daily.rolling(20, min_periods=20).std()
    historical_median = vol20.expanding(min_periods=252).median().shift(1)
    volatility = pd.Series("low_volatility", index=returns.index, dtype=object)
    volatility.loc[vol20.gt(historical_median)] = "high_volatility"
    rows = []
    for item in candidates:
        outcome = returns[_return_column(item["horizon"])]
        base = pd.concat([
            item["membership"].rename("membership"), outcome.rename("return"),
            trend.rename("trend_regime"), volatility.rename("volatility_regime"),
        ], axis=1).dropna()
        base = base.loc[base.membership]
        for dimensions in (("trend_regime",), ("volatility_regime",)):
            for key, frame in base.groupby(list(dimensions), observed=True):
                key = key[0] if isinstance(key, tuple) else key
                rows.append({
                    "candidate_id": item["candidate_id"],
                    "regime_type": dimensions[0], "regime": key,
                    "observation_count": len(frame),
                    "mean_return": frame["return"].mean(),
                    "positive_rate": frame["return"].gt(0).mean(),
                })
    return pd.DataFrame(rows)


def _add_control_interpretation(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["interpretation_flag"] = ""
    if result.empty or "model" not in result:
        return result
    uncontrolled = result.loc[result.model.eq("group_dummy_uncontrolled"), [
        "candidate_id", "beta", "raw_p_value"
    ]].rename(columns={"beta": "uncontrolled_beta", "raw_p_value": "uncontrolled_p_value"})
    controlled_index = result.model.eq("group_dummy_controlled")
    merged = result.loc[controlled_index, ["candidate_id", "beta", "raw_p_value"]].merge(
        uncontrolled, on="candidate_id", how="left"
    )
    for row_index, (_, row) in zip(result.index[controlled_index], merged.iterrows()):
        same_direction = np.sign(row["beta"]) == np.sign(row["uncontrolled_beta"])
        if same_direction and row["raw_p_value"] < 0.05:
            flag = "incremental_predictive_information"
        elif not same_direction:
            flag = "unstable_after_controls"
        else:
            flag = "likely_prior_return_volatility_or_turnover_proxy"
        result.loc[row_index, "interpretation_flag"] = flag
        result.loc[row_index, "uncontrolled_beta"] = row["uncontrolled_beta"]
        result.loc[row_index, "uncontrolled_p_value"] = row["uncontrolled_p_value"]
    return result


def _data_regime_audit(raw: RawData) -> pd.DataFrame:
    rows = []
    regimes = {
        "dealer_definition_regime": "2014-12-01",
        "foreign_definition_regime": "2018-01-15",
    }
    for dataset, frame in {
        "buy": raw.institutional_buy, "sell": raw.institutional_sell,
        "net": raw.institutional_net,
    }.items():
        for column in frame:
            series = pd.to_numeric(frame[column], errors="coerce")
            rows.append({
                "dataset": dataset, "column": column,
                "start_date": series.first_valid_index(), "end_date": series.last_valid_index(),
                "observation_count": int(series.notna().sum()),
                "missing_count": int(series.isna().sum()),
                "dealer_definition_regime": regimes["dealer_definition_regime"],
                "foreign_definition_regime": regimes["foreign_definition_regime"],
            })
    return pd.DataFrame(rows)


def _reconstruction_discrepancies(raw: RawData, tolerance: float = 1.0) -> pd.DataFrame:
    rows = []
    for flow_type, frame in {
        "buy": raw.institutional_buy, "sell": raw.institutional_sell,
        "net": raw.institutional_net,
    }.items():
        rebuilt = reconstruct_institution_flows(frame)
        for scope, official_column in (("listed", "上市合計"), ("otc", "上櫃三大法人合計*")):
            official = pd.to_numeric(frame[official_column], errors="coerce")
            reconstructed = rebuilt[(scope, "total_institutional")]
            diff = reconstructed - official
            for date in diff.index[diff.abs().gt(tolerance).fillna(False)]:
                rows.append({
                    "date": date, "market_scope": scope, "flow_type": flow_type,
                    "official_total": official.loc[date],
                    "reconstructed_total": reconstructed.loc[date],
                    "absolute_difference": abs(diff.loc[date]),
                    "relative_difference": diff.loc[date] / official.loc[date] if official.loc[date] else np.nan,
                })
    return pd.DataFrame(rows, columns=[
        "date", "market_scope", "flow_type", "official_total",
        "reconstructed_total", "absolute_difference", "relative_difference",
    ])


def _mark_discrepancy_candidate_impact(
    discrepancies: pd.DataFrame, candidates: list[dict]
) -> pd.DataFrame:
    result = discrepancies.copy()
    if result.empty:
        result["affects_candidate_signal"] = pd.Series(dtype=bool)
        result["affected_candidate_ids"] = pd.Series(dtype=str)
        return result
    affected = []
    for date in pd.to_datetime(result["date"]):
        ids = [
            item["candidate_id"] for item in candidates
            if date in item["membership"].index and bool(item["membership"].loc[date])
        ]
        affected.append(";".join(ids))
    result["affected_candidate_ids"] = affected
    result["affects_candidate_signal"] = result["affected_candidate_ids"].ne("")
    return result


def _write_summary(path: Path, confirmatory: pd.DataFrame, nonlinear: pd.DataFrame, excluded: pd.DataFrame) -> None:
    controlled = confirmatory.loc[confirmatory.get("model", pd.Series(dtype=str)).eq("group_dummy_controlled")]
    significant = int(controlled.get("fdr_global", pd.Series(dtype=float)).lt(0.05).sum())
    nonlinear_sig = int(nonlinear.get("fdr_global", pd.Series(dtype=float)).lt(0.05).sum())
    path.write_text(f"""# Phase 2 法人現貨機制驗證摘要

## 執行範圍

- 既有候選訊號控制後驗證，不重跑舊版完整 grid。
- 新增 Gross、Directional Balance、Net Intensity、非重疊變化與轉折訊號。
- 主要報酬為 O1→C1／C5／C10，主要標準化為 756 日、504 日作穩健性分析。

## 程式產出概況

- Confirmatory 結果：{len(confirmatory)} 列
- 控制後 Global FDR 顯著：{significant} 列
- 非線性 Global FDR 顯著：{nonlinear_sig} 列
- FDR 前因樣本不足或無法估計而排除：{len(excluded)} 列

## 解讀規則

真實資料完成前不自動宣稱訊號有效。請優先檢查控制後方向、504／756 一致性、子期間穩定性、勝率與左尾風險，以及資料制度斷點。
""", encoding="utf-8")


def run_phase2_study(raw: RawData, config: StudyConfig | None = None) -> Path:
    config = config or StudyConfig(study_mode="phase2_flow_mechanism")
    windows = tuple(config.phase2_normalization_windows)
    normalization_windows = tuple(dict.fromkeys(
        ((252,) if config.phase2_include_252_sensitivity else ()) + windows
    ))
    horizons = tuple(config.phase2_return_horizons)
    turnover = build_turnover(raw.market_amount)
    levels = build_phase2_activity_predictors(
        raw.institutional_buy, raw.institutional_sell, raw.institutional_net,
        turnover, config.accumulation_windows,
    )
    changes = build_nonoverlapping_changes(levels)
    features = pd.concat([levels, changes], axis=1)
    normalized = add_normalizations(
        features, windows=normalization_windows, include_global=False
    )
    returns = build_forward_returns(
        raw.adjusted_open, raw.adjusted_close, features.index, horizons=horizons
    )
    controls = build_prior_market_controls(
        raw.adjusted_close, turnover, features.index, config.phase2_turnover_epsilon
    )
    first_valid = [
        normalized[c].first_valid_index() for c in normalized
        if any(f"__rolling_{w}d_" in c for w in windows)
    ]
    common_start = max(date for date in first_valid if date is not None)
    normalized_common = normalized.loc[common_start:]
    returns_common = returns.loc[common_start:]
    controls_common = controls.loc[common_start:]
    features_common = features.loc[common_start:]

    candidates = _candidate_memberships(
        normalized_common,
        normalization_windows,
        horizons,
        include_252_sensitivity=config.phase2_include_252_sensitivity,
    )
    confirmatory_raw = _confirmatory_results(candidates, returns_common, controls_common)
    confirmatory, excluded_confirmatory = add_separate_fdr(
        confirmatory_raw, config.phase2_min_group_n
    )
    confirmatory = _add_control_interpretation(confirmatory)
    distribution_results = confirmatory.loc[
        confirmatory.get("model", pd.Series(dtype=str)).eq("group_uncontrolled")
    ].copy()
    hit_rate_raw = confirmatory_raw.loc[
        confirmatory_raw.get("model", pd.Series(dtype=str)).eq("group_uncontrolled")
        & confirmatory_raw.get("positive_vs_unconditional_p_value", pd.Series(dtype=float)).notna()
    ].copy()
    if not hit_rate_raw.empty:
        hit_rate_raw["raw_p_value"] = hit_rate_raw["positive_vs_unconditional_p_value"]
        hit_rate_raw["hypothesis_family"] = "confirmatory_hit_rate"
    hit_rate_results, excluded_hit_rate = add_separate_fdr(
        hit_rate_raw, config.phase2_min_group_n
    )
    nonlinear_raw = _nonlinear_results(
        normalized_common, returns_common, windows, horizons, config.phase2_spline_df
    )
    nonlinear, excluded_nonlinear = add_separate_fdr(
        nonlinear_raw, config.phase2_min_group_n
    )
    curve_data = _nonlinear_curve_data(
        normalized_common, returns_common, windows, horizons
    )
    group_raw = _group_feature_results(normalized_common, returns_common, windows, horizons)
    group_results, excluded_groups = add_separate_fdr(group_raw, config.phase2_min_group_n)
    turning = build_turning_point_signals(features, normalized, windows).loc[common_start:]
    turning_raw = _turning_results(turning, returns_common, horizons)
    turning_results, excluded_turning = add_separate_fdr(
        turning_raw, config.phase2_min_group_n
    )
    excluded = pd.concat(
        [excluded_confirmatory, excluded_hit_rate, excluded_nonlinear, excluded_groups, excluded_turning],
        ignore_index=True, sort=False,
    )
    subperiod, leave_one_year_out = _temporal_breakdowns(candidates, returns_common)
    market_regime = _market_regime_breakdown(
        candidates, returns_common, raw.adjusted_close
    )
    regime_audit = _data_regime_audit(raw)
    discrepancies = _mark_discrepancy_candidate_impact(
        _reconstruction_discrepancies(raw), candidates
    )

    output = timestamped_output_directory(config.output_root)
    _feature_dictionary(list(features.columns)).to_csv(output / "phase2_feature_dictionary.csv", index=False)
    confirmatory.to_csv(output / "phase2_confirmatory_results.csv", index=False)
    confirmatory.loc[
        confirmatory.get("model", pd.Series(dtype=str)).ne("group_uncontrolled")
    ].to_csv(output / "phase2_controlled_regressions.csv", index=False)
    confirmatory.to_csv(output / "phase2_signal_comparison.csv", index=False)
    nonlinear.to_csv(output / "phase2_nonlinear_results.csv", index=False)
    curve_data.to_csv(output / "phase2_nonlinear_curve_data.csv", index=False)
    group_results.loc[group_results.get("predictor", pd.Series(dtype=str)).str.contains("gross|directional_balance|net_intensity", regex=True, na=False)].to_csv(
        output / "phase2_gross_activity_results.csv", index=False
    )
    group_results.loc[group_results.get("predictor", pd.Series(dtype=str)).str.contains("_change__", regex=False, na=False)].to_csv(
        output / "phase2_flow_change_results.csv", index=False
    )
    turning_results.to_csv(output / "phase2_turning_point_results.csv", index=False)
    distribution_results.to_csv(output / "phase2_return_distribution_results.csv", index=False)
    hit_rate_results.to_csv(output / "phase2_hit_rate_signals.csv", index=False)
    subperiod.to_csv(output / "phase2_subperiod_results.csv", index=False)
    market_regime.to_csv(output / "phase2_market_regime_results.csv", index=False)
    leave_one_year_out.to_csv(output / "phase2_leave_one_year_out.csv", index=False)
    regime_audit.to_csv(output / "phase2_data_regime_audit.csv", index=False)
    discrepancies.to_csv(output / "phase2_reconstruction_discrepancies.csv", index=False)
    excluded.to_csv(output / "phase2_excluded_results.csv", index=False)
    pd.concat([confirmatory, nonlinear, group_results, turning_results], ignore_index=True, sort=False).loc[
        lambda x: x.get("fdr_global", pd.Series(dtype=float)).lt(0.05)
    ].to_csv(output / "phase2_significant_results.csv", index=False)
    vif_table(controls_common[["prior_return_1d", "prior_return_5d", "prior_return_10d", "prior_volatility_20d"]]).to_csv(
        output / "phase2_control_vif.csv", index=False
    )
    _write_summary(output / "phase2_summary.md", confirmatory, nonlinear, excluded)
    metadata = {
        "study_mode": "phase2_flow_mechanism",
        "data_start": str(features.index.min()), "data_end": str(features.index.max()),
        "common_sample_start": str(common_start),
        "minimum_group_n_before_fdr": config.phase2_min_group_n,
        "confirmatory_hypotheses": len(candidates),
        "exploratory_search_restricted": True,
        "legacy_full_grid_executed": False,
        "price_dataset_names": raw.price_dataset_names,
    }
    write_json(output / "phase2_run_metadata.json", build_manifest(config.to_dict(), metadata))
    return output
