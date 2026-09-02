import numpy as np

from features import (
    add_normalizations,
    build_flow_predictors,
    build_turnover,
)
from synthetic_data import make_synthetic_raw_data


def test_five_day_predictor_is_ratio_of_sums_not_mean_of_ratios():
    raw = make_synthetic_raw_data(periods=12)
    turnover = build_turnover(raw.market_amount)
    predictors = build_flow_predictors(
        raw.institutional_buy,
        raw.institutional_sell,
        raw.institutional_net,
        turnover,
        windows=(1, 5),
    )
    numerator = (
        raw.institutional_net["上市外資及陸資(不含外資自營商)"]
        + raw.institutional_net["上櫃外資及陸資(不含自營商)"]
    )
    expected = numerator.rolling(5).sum() / turnover["combined"].rolling(5).sum()
    actual = predictors["combined__foreign__net__5d"]
    assert np.allclose(actual.dropna(), expected.dropna())


def test_rolling_normalization_does_not_change_past_when_future_added():
    raw = make_synthetic_raw_data(periods=40)
    turnover = build_turnover(raw.market_amount)
    predictors = build_flow_predictors(
        raw.institutional_buy,
        raw.institutional_sell,
        raw.institutional_net,
        turnover,
        windows=(1,),
    )
    short = add_normalizations(
        predictors.iloc[:30], windows=(10,), include_global=False
    )
    full = add_normalizations(
        predictors, windows=(10,), include_global=False
    )
    assert np.allclose(short.to_numpy(), full.loc[short.index].to_numpy(), equal_nan=True)


def test_multiple_normalization_windows_are_named_and_aligned():
    raw = make_synthetic_raw_data(periods=40)
    turnover = build_turnover(raw.market_amount)
    predictors = build_flow_predictors(
        raw.institutional_buy,
        raw.institutional_sell,
        raw.institutional_net,
        turnover,
        windows=(1,),
    )
    normalized = add_normalizations(
        predictors[["combined__foreign__net__1d"]],
        windows=(10, 20, 30),
        include_global=False,
    )
    assert "combined__foreign__net__1d__rolling_10d_pr" in normalized
    assert "combined__foreign__net__1d__rolling_20d_z" in normalized
    assert "combined__foreign__net__1d__rolling_30d_pr" in normalized
    assert normalized["combined__foreign__net__1d__rolling_10d_z"].first_valid_index() == predictors.index[9]
    assert normalized["combined__foreign__net__1d__rolling_30d_z"].first_valid_index() == predictors.index[29]
