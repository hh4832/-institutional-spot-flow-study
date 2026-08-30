from __future__ import annotations

import numpy as np
import pandas as pd


def build_forward_returns(
    adjusted_open: pd.Series,
    adjusted_close: pd.Series,
    signal_index: pd.DatetimeIndex,
    horizons: tuple[int, ...] = (1, 2, 3, 5, 10),
) -> pd.DataFrame:
    prices = pd.concat(
        [adjusted_open.rename("open"), adjusted_close.rename("close")], axis=1
    ).sort_index()
    prices = prices.loc[~prices.index.duplicated(keep="last")]
    prices = prices.dropna(subset=["open", "close"])
    if (prices <= 0).any().any():
        raise ValueError("0050 開盤價或收盤價包含非正數")

    position = pd.Series(np.arange(len(prices)), index=prices.index)
    result = pd.DataFrame(index=pd.DatetimeIndex(signal_index).sort_values())
    result.index.name = "signal_date"
    result["entry_date"] = pd.NaT
    result["return_c0_o1"] = np.nan

    valid_signals = result.index.intersection(prices.index)
    signal_positions = position.loc[valid_signals].to_numpy()
    next_positions = signal_positions + 1
    valid_next = next_positions < len(prices)
    usable_signals = valid_signals[valid_next]
    usable_signal_pos = signal_positions[valid_next]
    usable_entry_pos = next_positions[valid_next]

    result.loc[usable_signals, "entry_date"] = prices.index[
        usable_entry_pos
    ].to_numpy()
    result.loc[usable_signals, "return_c0_o1"] = (
        prices["open"].to_numpy()[usable_entry_pos]
        / prices["close"].to_numpy()[usable_signal_pos]
        - 1
    )

    for horizon in horizons:
        column = f"return_o1_c{horizon}"
        exit_date_column = f"exit_date_c{horizon}"
        result[column] = np.nan
        result[exit_date_column] = pd.NaT
        exit_positions = usable_signal_pos + horizon
        valid_exit = exit_positions < len(prices)
        signals = usable_signals[valid_exit]
        entries = usable_entry_pos[valid_exit]
        exits = exit_positions[valid_exit]
        result.loc[signals, column] = (
            prices["close"].to_numpy()[exits]
            / prices["open"].to_numpy()[entries]
            - 1
        )
        result.loc[signals, exit_date_column] = prices.index[exits].to_numpy()

    return result
