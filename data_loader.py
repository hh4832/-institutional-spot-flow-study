from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from config import PRICE_DATASET_CANDIDATES


@dataclass
class RawData:
    market_amount: pd.DataFrame
    institutional_buy: pd.DataFrame
    institutional_sell: pd.DataFrame
    institutional_net: pd.DataFrame
    adjusted_open: pd.Series
    adjusted_close: pd.Series
    price_dataset_names: dict[str, str]


def authenticate_finlab(api_token: str | None = None) -> None:
    import finlab

    finlab.login(api_token)


def _as_plain_dataframe(obj) -> pd.DataFrame:
    result = pd.DataFrame(obj).copy()
    result.index = pd.to_datetime(result.index)
    return result.sort_index()


def _load_price_field(
    getter: Callable[[str], object],
    ticker: str,
    field: str,
) -> tuple[pd.Series, str]:
    errors: list[str] = []
    for dataset_name in PRICE_DATASET_CANDIDATES[field]:
        try:
            frame = _as_plain_dataframe(getter(dataset_name))
            if ticker not in frame.columns:
                errors.append(f"{dataset_name}: 找不到 {ticker} 欄位")
                continue
            series = pd.to_numeric(frame[ticker], errors="coerce").rename(field)
            if (series.dropna() <= 0).any():
                errors.append(f"{dataset_name}: 包含非正價格")
                continue
            return series, dataset_name
        except Exception as exc:  # pragma: no cover - depends on remote API
            errors.append(f"{dataset_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"無法取得 {ticker} 的 {field} 價格。候選資料集結果：" + " | ".join(errors)
    )


def _load_derived_adjusted_open(
    getter: Callable[[str], object],
    ticker: str,
    adjusted_close: pd.Series,
) -> tuple[pd.Series, str]:
    """Derive adjusted open with the close adjustment factor when needed."""
    raw_open = _as_plain_dataframe(getter("price:開盤價"))
    raw_close = _as_plain_dataframe(getter("price:收盤價"))
    if ticker not in raw_open or ticker not in raw_close:
        raise RuntimeError(f"原始開／收盤價資料找不到 {ticker}")
    frame = pd.concat(
        [
            pd.to_numeric(raw_open[ticker], errors="coerce").rename("raw_open"),
            pd.to_numeric(raw_close[ticker], errors="coerce").rename("raw_close"),
            adjusted_close.rename("adjusted_close"),
        ],
        axis=1,
    )
    factor = frame["adjusted_close"].div(frame["raw_close"].replace(0, pd.NA))
    adjusted_open = frame["raw_open"].mul(factor).rename("open")
    if (adjusted_open.dropna() <= 0).any():
        raise RuntimeError("推導的還原開盤價包含非正數")
    return adjusted_open, "derived:price:開盤價*(etl:adj_close/price:收盤價)"


def load_finlab_data(ticker: str = "0050", getter=None) -> RawData:
    if getter is None:
        from finlab import data

        getter = data.get

    market_amount = _as_plain_dataframe(
        getter("market_transaction_info:成交金額")
    )
    institutional_buy = _as_plain_dataframe(
        getter("institutional_investors_trading_all_market_summary:買進金額")
    )
    institutional_sell = _as_plain_dataframe(
        getter("institutional_investors_trading_all_market_summary:賣出金額")
    )
    institutional_net = _as_plain_dataframe(
        getter("institutional_investors_trading_all_market_summary:買賣超")
    )
    adjusted_close, close_name = _load_price_field(getter, ticker, "close")
    try:
        adjusted_open, open_name = _load_price_field(getter, ticker, "open")
    except RuntimeError:
        adjusted_open, open_name = _load_derived_adjusted_open(
            getter, ticker, adjusted_close
        )

    return RawData(
        market_amount=market_amount,
        institutional_buy=institutional_buy,
        institutional_sell=institutional_sell,
        institutional_net=institutional_net,
        adjusted_open=adjusted_open,
        adjusted_close=adjusted_close,
        price_dataset_names={"open": open_name, "close": close_name},
    )
