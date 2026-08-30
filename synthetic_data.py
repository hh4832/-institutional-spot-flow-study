from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import RawData


def make_synthetic_raw_data(periods: int = 420, seed: int = 42) -> RawData:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-02", periods=periods)
    columns = [
        "上市合計",
        "上市外資",
        "上市外資及陸資(不含外資自營商)",
        "上市外資自營商",
        "上市投信",
        "上市自營商(自行買賣)",
        "上市自營商(避險)",
        "上市自營商合計",
        "上櫃三大法人合計*",
        "上櫃外資及陸資(不含自營商)",
        "上櫃外資及陸資合計",
        "上櫃外資自營商",
        "上櫃投信",
        "上櫃自營商(自行買賣)",
        "上櫃自營商(避險)",
        "上櫃自營商合計",
    ]

    def make_direction(scale: float) -> pd.DataFrame:
        frame = pd.DataFrame(0.0, index=index, columns=columns)
        primitives = {
            "上市外資及陸資(不含外資自營商)": rng.lognormal(24, 0.25, periods) * scale,
            "上市投信": rng.lognormal(21.5, 0.25, periods) * scale,
            "上市自營商(自行買賣)": rng.lognormal(21, 0.25, periods) * scale,
            "上市自營商(避險)": rng.lognormal(22, 0.25, periods) * scale,
            "上櫃外資及陸資(不含自營商)": rng.lognormal(22.5, 0.25, periods) * scale,
            "上櫃投信": rng.lognormal(20.5, 0.25, periods) * scale,
            "上櫃自營商(自行買賣)": rng.lognormal(20, 0.25, periods) * scale,
            "上櫃自營商(避險)": rng.lognormal(20.5, 0.25, periods) * scale,
        }
        for column, values in primitives.items():
            frame[column] = values
        frame["上市外資自營商"] = 0.0
        frame["上櫃外資自營商"] = 0.0
        frame["上市外資"] = np.nan
        frame["上市自營商合計"] = np.nan
        frame["上櫃外資及陸資合計"] = frame[
            "上櫃外資及陸資(不含自營商)"
        ]
        frame["上櫃自營商合計"] = (
            frame["上櫃自營商(自行買賣)"] + frame["上櫃自營商(避險)"]
        )
        frame["上市合計"] = (
            frame["上市外資及陸資(不含外資自營商)"]
            + frame["上市投信"]
            + frame["上市自營商(自行買賣)"]
            + frame["上市自營商(避險)"]
        )
        frame["上櫃三大法人合計*"] = (
            frame["上櫃外資及陸資(不含自營商)"]
            + frame["上櫃投信"]
            + frame["上櫃自營商(自行買賣)"]
            + frame["上櫃自營商(避險)"]
        )
        return frame

    buy = make_direction(1.02)
    sell = make_direction(0.98)
    net = buy - sell
    market = pd.DataFrame(
        {
            "TAIEX": rng.lognormal(27.3, 0.15, periods),
            "OTC": rng.lognormal(25.8, 0.15, periods),
        },
        index=index,
    )
    close_returns = rng.normal(0.0003, 0.011, periods)
    close = pd.Series(100 * np.cumprod(1 + close_returns), index=index, name="close")
    open_price = close.shift(1).fillna(close.iloc[0]) * (
        1 + rng.normal(0, 0.003, periods)
    )
    open_price.name = "open"
    return RawData(
        market_amount=market,
        institutional_buy=buy,
        institutional_sell=sell,
        institutional_net=net,
        adjusted_open=open_price,
        adjusted_close=close,
        price_dataset_names={
            "open": "synthetic_adjusted_open",
            "close": "synthetic_adjusted_close",
        },
    )
