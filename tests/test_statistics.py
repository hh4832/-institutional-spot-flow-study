import numpy as np
import pandas as pd

from study_statistics import summarize_group


def test_hac_summary_has_expected_mean_difference():
    index = pd.bdate_range("2020-01-01", periods=100)
    membership = pd.Series([True] * 50 + [False] * 50, index=index)
    returns = pd.Series([0.02] * 50 + [-0.01] * 50, index=index)
    result = summarize_group(returns, membership, maxlags=0)
    assert np.isclose(result["mean_return"], 0.02)
    assert np.isclose(result["nongroup_mean"], -0.01)
    assert np.isclose(result["difference_vs_nongroup"], 0.03)
    assert np.isclose(result["unconditional_mean"], 0.005)
    assert np.isclose(result["difference_vs_unconditional"], 0.015)
