import numpy as np
import pandas as pd

from data_loader import _load_derived_adjusted_open


def test_adjusted_open_is_derived_on_same_scale_as_adjusted_close():
    index = pd.bdate_range("2024-01-01", periods=3)
    datasets = {
        "price:開盤價": pd.DataFrame({"0050": [100.0, 50.0, 51.0]}, index=index),
        "price:收盤價": pd.DataFrame({"0050": [100.0, 50.0, 50.0]}, index=index),
    }
    adjusted_close = pd.Series([50.0, 50.0, 50.0], index=index)
    adjusted_open, name = _load_derived_adjusted_open(
        datasets.__getitem__, "0050", adjusted_close
    )
    assert np.allclose(adjusted_open, [50.0, 50.0, 51.0])
    assert name.startswith("derived:")
