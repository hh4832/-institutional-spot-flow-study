from __future__ import annotations

import numpy as np
import pandas as pd

from config import INSTITUTION_COLUMNS


def _required_columns() -> set[str]:
    return {
        column
        for institution in INSTITUTION_COLUMNS.values()
        for scope_columns in institution.values()
        for column in scope_columns
    }


def reconstruct_institution_flows(frame: pd.DataFrame) -> pd.DataFrame:
    """Build listed, OTC and combined flows from primitive FinLab columns."""
    missing = _required_columns().difference(frame.columns)
    if missing:
        raise KeyError(f"法人資料缺少必要欄位：{sorted(missing)}")

    frame = pd.DataFrame(frame).sort_index()
    result: dict[tuple[str, str], pd.Series] = {}
    for institution, scopes in INSTITUTION_COLUMNS.items():
        for scope, columns in scopes.items():
            result[(scope, institution)] = frame[columns].sum(
                axis=1, min_count=len(columns)
            )
        result[("combined", institution)] = (
            result[("listed", institution)] + result[("otc", institution)]
        )

    for scope in ("listed", "otc", "combined"):
        result[(scope, "total_institutional")] = sum(
            result[(scope, institution)]
            for institution in INSTITUTION_COLUMNS
        )

    output = pd.DataFrame(result, index=frame.index)
    output.columns = pd.MultiIndex.from_tuples(
        output.columns, names=["market_scope", "institution"]
    )
    return output.sort_index(axis=1)


def build_turnover(market_amount: pd.DataFrame) -> pd.DataFrame:
    required = {"TAIEX", "OTC"}
    missing = required.difference(market_amount.columns)
    if missing:
        raise KeyError(f"市場成交金額缺少欄位：{sorted(missing)}")
    market_amount = pd.DataFrame(market_amount).sort_index()
    result = pd.DataFrame(index=market_amount.index)
    result["listed"] = pd.to_numeric(market_amount["TAIEX"], errors="coerce")
    result["otc"] = pd.to_numeric(market_amount["OTC"], errors="coerce")
    result["combined"] = result["listed"] + result["otc"]
    return result


def build_flow_predictors(
    buy: pd.DataFrame,
    sell: pd.DataFrame,
    official_net: pd.DataFrame,
    turnover: pd.DataFrame,
    windows: tuple[int, ...] = (1, 5, 10),
    include_gross_activity: bool = False,
) -> pd.DataFrame:
    reconstructed = {
        "buy": reconstruct_institution_flows(buy),
        "sell": reconstruct_institution_flows(sell),
        "net": reconstruct_institution_flows(official_net),
    }
    index = turnover.index
    for frame in reconstructed.values():
        index = index.intersection(frame.index)
    turnover = turnover.loc[index]

    records: dict[str, pd.Series] = {}
    flow_types = ["buy", "sell", "net"]
    if include_gross_activity:
        reconstructed["gross_activity"] = reconstructed["buy"] + reconstructed["sell"]
        flow_types.append("gross_activity")

    for flow_type in flow_types:
        flow_frame = reconstructed[flow_type].loc[index]
        for scope, institution in flow_frame.columns:
            amount = flow_frame[(scope, institution)]
            for window in windows:
                numerator = amount.rolling(window, min_periods=window).sum()
                denominator = turnover[scope].rolling(
                    window, min_periods=window
                ).sum()
                name = f"{scope}__{institution}__{flow_type}__{window}d"
                records[name] = numerator.div(denominator).replace(
                    [np.inf, -np.inf], np.nan
                )
    return pd.DataFrame(records, index=index)


def rolling_percentile_rank(
    series: pd.Series, window: int = 252, min_periods: int = 252
) -> pd.Series:
    def last_rank(values: np.ndarray) -> float:
        last = values[-1]
        valid = values[~np.isnan(values)]
        if np.isnan(last) or valid.size == 0:
            return np.nan
        return float(np.mean(valid <= last) * 100.0)

    return series.rolling(window, min_periods=min_periods).apply(
        last_rank, raw=True
    )


def rolling_zscore(
    series: pd.Series, window: int = 252, min_periods: int = 252
) -> pd.Series:
    rolling = series.rolling(window, min_periods=min_periods)
    mean = rolling.mean()
    std = rolling.std(ddof=0).replace(0, np.nan)
    return (series - mean) / std


def add_normalizations(
    predictors: pd.DataFrame,
    windows: tuple[int, ...] = (252, 504, 756),
    include_global: bool = True,
) -> pd.DataFrame:
    if not windows:
        raise ValueError("標準化視窗不可為空")
    if any(window <= 1 for window in windows):
        raise ValueError("標準化視窗必須大於 1")
    if len(set(windows)) != len(windows):
        raise ValueError("標準化視窗不可重複")

    output: dict[str, pd.Series] = {}
    for name in predictors.columns:
        series = predictors[name]
        output[f"{name}__raw"] = series
        for window in windows:
            output[f"{name}__rolling_{window}d_pr"] = rolling_percentile_rank(
                series, window, window
            )
            output[f"{name}__rolling_{window}d_z"] = rolling_zscore(
                series, window, window
            )
        if include_global:
            output[f"{name}__global_pr"] = series.rank(
                method="average", pct=True
            ) * 100
            std = series.std(ddof=0)
            output[f"{name}__global_z"] = (
                (series - series.mean()) / std if std != 0 else np.nan
            )
    return pd.DataFrame(output, index=predictors.index)
