import numpy as np
import pandas as pd

from phase2_pipeline import _membership_for_label
from phase2_statistics import add_separate_fdr, hac_signal_regression, nonlinear_hac_regressions
from study_statistics import summarize_group


def test_minimum_sample_filter_is_applied_before_fdr():
    frame = pd.DataFrame({
        "hypothesis_family": ["confirmatory", "confirmatory", "exploratory"],
        "observation_count": [3, 60, 80],
        "raw_p_value": [1e-12, 0.02, 0.03],
    })
    eligible, excluded = add_separate_fdr(frame, min_n=50)
    assert len(eligible) == 2
    assert len(excluded) == 1
    assert excluded.iloc[0].exclusion_reason == "observation_count_below_minimum"
    assert eligible.fdr_global.notna().all()


def test_quadratic_uses_centered_predictor_and_spline_runs():
    rng = np.random.default_rng(4)
    index = pd.bdate_range("2020-01-01", periods=300)
    x = pd.Series(np.linspace(-2, 2, len(index)), index=index)
    y = x.pow(2) + pd.Series(rng.normal(0, 0.2, len(index)), index=index)
    rows = nonlinear_hac_regressions(x, y, maxlags=4, spline_df=4)
    types = {row["model_type"] for row in rows}
    assert types == {"linear", "quadratic", "restricted_cubic_spline"}
    quadratic = next(row for row in rows if row["model_type"] == "quadratic")
    assert quadratic["quadratic_beta"] > 0
    assert quadratic["nonlinear_p_value"] < 0.05


def test_missing_group_values_are_excluded_instead_of_becoming_nongroup():
    index = pd.bdate_range("2020-01-01", periods=8)
    groups = pd.Series(
        pd.Categorical([None, None, "target", "other", "target", "other", "target", "other"]),
        index=index,
    )
    membership = _membership_for_label(groups, "target")
    returns = pd.Series(np.arange(8, dtype=float) / 100, index=index)

    assert membership.iloc[:2].isna().all()
    summary = summarize_group(returns, membership, maxlags=0)
    regression = hac_signal_regression(
        membership.astype("Float64"), returns, maxlags=0
    )
    assert summary["unconditional_count"] == 6
    assert summary["observation_count"] == 3
    assert summary["nongroup_count"] == 3
    assert regression == {}  # Regression requires at least 10 complete cases.
