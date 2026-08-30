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
        predictors.iloc[:30], window=10, min_periods=10, include_global=False
    )
    full = add_normalizations(
        predictors, window=10, min_periods=10, include_global=False
    )
    assert np.allclose(short.to_numpy(), full.loc[short.index].to_numpy(), equal_nan=True)
