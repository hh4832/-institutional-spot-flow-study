import pandas as pd
import pytest

from data_loader import load_finlab_data


def _base_datasets(index):
    return {
        "market_transaction_info:成交金額": pd.DataFrame(index=index),
        "institutional_investors_trading_all_market_summary:買進金額": pd.DataFrame(index=index),
        "institutional_investors_trading_all_market_summary:賣出金額": pd.DataFrame(index=index),
        "institutional_investors_trading_all_market_summary:買賣超": pd.DataFrame(index=index),
        "etl:adj_close": pd.DataFrame({"0050": [100.0, 101.0, 102.0]}, index=index),
    }


def test_adjusted_price_fields_are_required_and_recorded():
    index = pd.bdate_range("2024-01-01", periods=3)
    datasets = _base_datasets(index)
    datasets["etl:adj_open"] = pd.DataFrame({"0050": [100.0, 100.0, 101.0]}, index=index)
    raw = load_finlab_data(getter=datasets.__getitem__)
    assert raw.price_dataset_names == {"open": "etl:adj_open", "close": "etl:adj_close"}


def test_missing_adjusted_open_does_not_fallback_to_raw_price():
    index = pd.bdate_range("2024-01-01", periods=3)
    datasets = _base_datasets(index)
    datasets["price:開盤價"] = pd.DataFrame({"0050": [100.0, 50.0, 51.0]}, index=index)
    datasets["price:收盤價"] = pd.DataFrame({"0050": [100.0, 50.0, 50.0]}, index=index)
    with pytest.raises(RuntimeError, match="etl:adj_open"):
        load_finlab_data(getter=datasets.__getitem__)
