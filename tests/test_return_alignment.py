import numpy as np
import pandas as pd

from returns import build_forward_returns


def test_forward_return_alignment_uses_trading_days():
    index = pd.bdate_range("2024-01-01", periods=12)
    open_price = pd.Series(np.arange(100, 112), index=index, dtype=float)
    close = pd.Series(np.arange(101, 113), index=index, dtype=float)
    result = build_forward_returns(open_price, close, index, horizons=(1, 2, 5))
    d0 = index[0]
    assert result.loc[d0, "entry_date"] == index[1]
    assert result.loc[d0, "exit_date_c1"] == index[1]
    assert result.loc[d0, "exit_date_c2"] == index[2]
    assert result.loc[d0, "exit_date_c5"] == index[5]
    assert np.isclose(result.loc[d0, "return_c0_o1"], open_price.iloc[1] / close.iloc[0] - 1)
    assert np.isclose(result.loc[d0, "return_o1_c5"], close.iloc[5] / open_price.iloc[1] - 1)


def test_missing_future_price_is_not_forward_filled():
    index = pd.bdate_range("2024-01-01", periods=4)
    price = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)
    result = build_forward_returns(price, price, index, horizons=(5,))
    assert result["return_o1_c5"].isna().all()
