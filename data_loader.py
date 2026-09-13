from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from config import (
    OTC_INDEX_DATASET,
    OTC_PRICE_INDEX_SERIES,
    OTC_TOTAL_RETURN_SERIES,
    PRICE_DATASET_CANDIDATES,
)


@dataclass
class RawData:
    market_amount: pd.DataFrame
    institutional_buy: pd.DataFrame
    institutional_sell: pd.DataFrame
    institutional_net: pd.DataFrame
    adjusted_open: pd.Series
    adjusted_close: pd.Series
    price_dataset_names: dict[str, str]
    otc_total_return_index: pd.Series | None = None
    otc_price_index: pd.Series | None = None
    otc_index_dataset_name: str | None = None


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


def load_finlab_data(
    ticker: str = "0050", getter=None, include_otc_indices: bool = False
) -> RawData:
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
    adjusted_open, open_name = _load_price_field(getter, ticker, "open")
    otc_total_return = None
    otc_price_index = None
    otc_dataset_name = None
    if include_otc_indices:
        otc_index_frame = _as_plain_dataframe(getter(OTC_INDEX_DATASET))
        missing_otc = {
            OTC_TOTAL_RETURN_SERIES,
            OTC_PRICE_INDEX_SERIES,
        }.difference(otc_index_frame.columns)
        if missing_otc:
            raise RuntimeError(f"OTC 指數資料缺少必要欄位：{sorted(missing_otc)}")
        otc_total_return = pd.to_numeric(
            otc_index_frame[OTC_TOTAL_RETURN_SERIES], errors="coerce"
        ).rename("otc_total_return_index")
        otc_price_index = pd.to_numeric(
            otc_index_frame[OTC_PRICE_INDEX_SERIES], errors="coerce"
        ).rename("otc_price_index")
        if (otc_total_return.dropna() <= 0).any() or (otc_price_index.dropna() <= 0).any():
            raise RuntimeError("OTC 指數包含非正數")
        otc_dataset_name = OTC_INDEX_DATASET

    return RawData(
        market_amount=market_amount,
        institutional_buy=institutional_buy,
        institutional_sell=institutional_sell,
        institutional_net=institutional_net,
        adjusted_open=adjusted_open,
        adjusted_close=adjusted_close,
        price_dataset_names={"open": open_name, "close": close_name},
        otc_total_return_index=otc_total_return,
        otc_price_index=otc_price_index,
        otc_index_dataset_name=otc_dataset_name,
    )
