import numpy as np
import pandas as pd

from features import (
    add_normalizations,
    build_nonoverlapping_changes,
    build_phase2_activity_predictors,
    build_prior_market_controls,
    build_turning_point_signals,
    build_turnover,
)
from synthetic_data import make_synthetic_raw_data


def test_phase2_activity_formulas_and_zero_denominator():
    raw = make_synthetic_raw_data(periods=30)
    turnover = build_turnover(raw.market_amount)
    result = build_phase2_activity_predictors(
        raw.institutional_buy, raw.institutional_sell, raw.institutional_net,
        turnover, windows=(1,),
    )
    date = result.index[5]
    buy = raw.institutional_buy.loc[date, "上市外資及陸資(不含外資自營商)"]
    sell = raw.institutional_sell.loc[date, "上市外資及陸資(不含外資自營商)"]
    market = raw.market_amount.loc[date, "TAIEX"]
    assert np.isclose(result.loc[date, "listed__foreign__gross__1d"], (buy + sell) / market)
    assert np.isclose(result.loc[date, "listed__foreign__directional_balance__1d"], (buy - sell) / (buy + sell))
    assert np.isclose(result.loc[date, "listed__foreign__net_intensity__1d"], abs(buy - sell) / (buy + sell))

    raw.institutional_buy.loc[date, "上市投信"] = 0
    raw.institutional_sell.loc[date, "上市投信"] = 0
    raw.institutional_net.loc[date, "上市投信"] = 0
    zero = build_phase2_activity_predictors(
        raw.institutional_buy, raw.institutional_sell, raw.institutional_net,
        turnover, windows=(1,),
    )
    assert np.isnan(zero.loc[date, "listed__investment_trust__directional_balance__1d"])
    assert np.isnan(zero.loc[date, "listed__investment_trust__net_intensity__1d"])


def test_nonoverlapping_changes_use_k_day_shift():
    index = pd.bdate_range("2024-01-01", periods=12)
    source = pd.DataFrame({
        "otc__foreign__sell__1d": np.arange(12.0),
        "otc__foreign__sell__5d": np.arange(12.0) * 10,
    }, index=index)
    result = build_nonoverlapping_changes(source)
    assert result.loc[index[6], "otc__foreign__sell_change__1d"] == 1
    assert result.loc[index[6], "otc__foreign__sell_change__5d"] == 50


def test_prior_returns_only_use_d0_and_past():
    index = pd.bdate_range("2024-01-01", periods=30)
    close = pd.Series(np.arange(100.0, 130.0), index=index)
    turnover = pd.DataFrame({"listed": 1.0, "otc": 1.0, "combined": 2.0}, index=index)
    controls = build_prior_market_controls(close, turnover, index)
    date = index[15]
    expected = close.loc[date] / close.loc[index[10]] - 1
    assert np.isclose(controls.loc[date, "prior_return_5d"], expected)
    changed_future = close.copy()
    changed_future.loc[index[16]:] *= 100
    controls_future = build_prior_market_controls(changed_future, turnover, index)
    assert controls.loc[date, "prior_return_5d"] == controls_future.loc[date, "prior_return_5d"]


def test_turning_points_do_not_look_ahead():
    index = pd.bdate_range("2024-01-01", periods=8)
    predictors = pd.DataFrame({
        "otc__foreign__net__1d": [-1, -1, 1, 1, -1, 1, 1, 1],
        "otc__foreign__sell__1d": [8, 7, 6, 5, 4, 3, 2, 1],
        "otc__foreign__buy__1d": [1, 2, 3, 4, 5, 6, 7, 8],
        "otc__foreign__gross__1d": np.arange(8.0),
    }, index=index)
    normalized = add_normalizations(predictors, windows=(3,), include_global=False)
    result = build_turning_point_signals(predictors, normalized, (3,))
    assert result.loc[index[2], "otc__foreign__net_negative_to_positive__1d"]
    assert result.loc[index[3], "otc__foreign__sell_declining_3d__1d"]
    assert result.loc[index[3], "otc__foreign__buy_rising_3d__1d"]
