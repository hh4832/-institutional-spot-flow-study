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


def build_phase2_activity_predictors(
    buy: pd.DataFrame,
    sell: pd.DataFrame,
    official_net: pd.DataFrame,
    turnover: pd.DataFrame,
    windows: tuple[int, ...] = (1, 5, 10),
) -> pd.DataFrame:
    """Build Phase-2 levels that separate activity from direction.

    ``gross`` measures two-sided activity, ``directional_balance`` measures
    buy-versus-sell direction within institutional activity, and
    ``net_intensity`` measures the absolute directional concentration.
    """
    reconstructed_buy = reconstruct_institution_flows(buy)
    reconstructed_sell = reconstruct_institution_flows(sell)
    reconstructed_net = reconstruct_institution_flows(official_net)
    index = turnover.index
    for frame in (reconstructed_buy, reconstructed_sell, reconstructed_net):
        index = index.intersection(frame.index)
    turnover = turnover.loc[index]
    records: dict[str, pd.Series] = {}
    for scope, institution in reconstructed_buy.columns:
        buy_amount = reconstructed_buy.loc[index, (scope, institution)]
        sell_amount = reconstructed_sell.loc[index, (scope, institution)]
        net_amount = reconstructed_net.loc[index, (scope, institution)]
        for window in windows:
            buy_sum = buy_amount.rolling(window, min_periods=window).sum()
            sell_sum = sell_amount.rolling(window, min_periods=window).sum()
            net_sum = net_amount.rolling(window, min_periods=window).sum()
            activity_sum = buy_sum + sell_sum
            market_sum = turnover[scope].rolling(window, min_periods=window).sum()
            prefix = f"{scope}__{institution}"
            values = {
                "buy": buy_sum.div(market_sum),
                "sell": sell_sum.div(market_sum),
                "net": net_sum.div(market_sum),
                "gross": activity_sum.div(market_sum),
                "directional_balance": net_sum.div(activity_sum),
                "net_intensity": net_sum.abs().div(activity_sum),
            }
            for flow_type, series in values.items():
                records[f"{prefix}__{flow_type}__{window}d"] = series.replace(
                    [np.inf, -np.inf], np.nan
                )
    return pd.DataFrame(records, index=index)


def build_nonoverlapping_changes(
    predictors: pd.DataFrame,
    flow_types: tuple[str, ...] = (
        "buy",
        "sell",
        "net",
        "gross",
        "directional_balance",
    ),
) -> pd.DataFrame:
    """Compare each k-day level with the preceding non-overlapping k-day block."""
    records: dict[str, pd.Series] = {}
    for column in predictors.columns:
        parts = column.split("__")
        if len(parts) != 4 or parts[2] not in flow_types:
            continue
        window = int(parts[3].removesuffix("d"))
        name = f"{parts[0]}__{parts[1]}__{parts[2]}_change__{window}d"
        records[name] = predictors[column] - predictors[column].shift(window)
    return pd.DataFrame(records, index=predictors.index)


def build_prior_market_controls(
    close: pd.Series,
    turnover: pd.DataFrame,
    index: pd.Index,
    epsilon: float = 1.0,
) -> pd.DataFrame:
    """Create d0-known return, volatility and non-overlapping turnover controls."""
    close = pd.to_numeric(close, errors="coerce").reindex(index)
    daily_return = close.pct_change(fill_method=None)
    controls = pd.DataFrame(index=index)
    for window in (1, 5, 10):
        controls[f"prior_return_{window}d"] = close.div(close.shift(window)) - 1
    controls["prior_volatility_10d"] = daily_return.rolling(10, min_periods=10).std()
    controls["prior_volatility_20d"] = daily_return.rolling(20, min_periods=20).std()
    for scope in ("listed", "otc", "combined"):
        amount = pd.to_numeric(turnover[scope], errors="coerce").reindex(index)
        for window in (5, 10):
            current = amount.rolling(window, min_periods=window).sum()
            previous = current.shift(window)
            controls[f"{scope}__market_turnover_change_{window}d"] = np.log(
                (current + epsilon) / (previous + epsilon)
            ).replace([np.inf, -np.inf], np.nan)
    return controls


def build_turning_point_signals(
    predictors: pd.DataFrame,
    normalized: pd.DataFrame,
    rolling_windows: tuple[int, ...],
) -> pd.DataFrame:
    """Build pre-specified direction and sell-pressure turning points."""
    records: dict[str, pd.Series] = {}
    for column in predictors.columns:
        scope, institution, flow_type, window_label = column.split("__")
        window = int(window_label.removesuffix("d"))
        prefix = f"{scope}__{institution}"
        if flow_type == "net":
            previous = predictors[column].shift(window)
            records[f"{prefix}__net_negative_to_positive__{window}d"] = (
                previous.lt(0) & predictors[column].gt(0)
            )
            records[f"{prefix}__net_positive_to_negative__{window}d"] = (
                previous.gt(0) & predictors[column].lt(0)
            )
        if flow_type == "sell":
            one_day = f"{prefix}__sell__1d"
            if window == 1 and one_day in predictors:
                delta = predictors[one_day].diff()
                records[f"{prefix}__sell_declining_3d__1d"] = delta.lt(0).rolling(3).sum().eq(3)
                records[f"{prefix}__sell_declining_5d__1d"] = delta.lt(0).rolling(5).sum().eq(5)
                records[f"{prefix}__sell_change_positive_to_negative__1d"] = (
                    delta.shift(1).gt(0) & delta.lt(0)
                )
            for rolling_window in rolling_windows:
                pr_name = f"{column}__rolling_{rolling_window}d_pr"
                z_name = f"{column}__rolling_{rolling_window}d_z"
                if pr_name in normalized:
                    pr = normalized[pr_name]
                    records[f"{prefix}__sell_pr80_to_40_60__{window}d__rolling_{rolling_window}d"] = (
                        pr.shift(1).ge(80) & pr.between(40, 60, inclusive="both")
                    )
                    records[f"{prefix}__sell_pr80_to_lt40__{window}d__rolling_{rolling_window}d"] = (
                        pr.shift(1).ge(80) & pr.lt(40)
                    )
                if z_name in normalized:
                    z = normalized[z_name]
                    records[f"{prefix}__sell_z1_5_to_lt0_5__{window}d__rolling_{rolling_window}d"] = (
                        z.shift(1).ge(1.5) & z.lt(0.5)
                    )
        if flow_type == "buy" and window == 1:
            delta = predictors[column].diff()
            records[f"{prefix}__buy_rising_3d__1d"] = delta.gt(0).rolling(3).sum().eq(3)
            records[f"{prefix}__buy_rising_5d__1d"] = delta.gt(0).rolling(5).sum().eq(5)
            records[f"{prefix}__buy_change_negative_to_positive__1d"] = (
                delta.shift(1).lt(0) & delta.gt(0)
            )
        if flow_type == "gross":
            for rolling_window in rolling_windows:
                pr_name = f"{column}__rolling_{rolling_window}d_pr"
                if pr_name in normalized:
                    pr = normalized[pr_name]
                    records[f"{prefix}__gross_high_turn_down__{window}d__rolling_{rolling_window}d"] = (
                        pr.shift(1).ge(80) & pr.lt(pr.shift(1))
                    )
    result = pd.DataFrame(records, index=predictors.index)
    return result.astype("boolean")
