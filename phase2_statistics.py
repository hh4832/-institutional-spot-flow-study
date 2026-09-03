from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix
from scipy import stats
from statsmodels.stats.multitest import multipletests


def _fit(y: pd.Series, x: pd.DataFrame, maxlags: int):
    return sm.OLS(y.astype(float), x.astype(float)).fit(
        cov_type="HAC", cov_kwds={"maxlags": int(maxlags)}
    )


def _prepare(
    signal: pd.Series,
    returns: pd.Series,
    controls: pd.DataFrame | None = None,
) -> pd.DataFrame:
    parts = [signal.rename("signal"), returns.rename("return")]
    if controls is not None:
        parts.append(controls)
    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan).dropna()


def hac_signal_regression(
    signal: pd.Series,
    returns: pd.Series,
    maxlags: int,
    controls: pd.DataFrame | None = None,
) -> dict:
    frame = _prepare(signal, returns, controls)
    if len(frame) < 10 or frame["signal"].nunique() < 2:
        return {}
    columns = ["signal"] + ([] if controls is None else list(controls.columns))
    x = sm.add_constant(frame[columns], has_constant="add")
    result = _fit(frame["return"], x, maxlags)
    beta = float(result.params["signal"])
    se = float(result.bse["signal"])
    z = stats.norm.ppf(0.975)
    return {
        "observation_count": len(frame),
        "beta": beta,
        "hac_standard_error": se,
        "t_statistic": beta / se if se else np.nan,
        "ci_low": beta - z * se,
        "ci_high": beta + z * se,
        "raw_p_value": float(result.pvalues["signal"]),
        "r_squared": float(result.rsquared),
        "adjusted_r_squared": float(result.rsquared_adj),
    }


def nonlinear_hac_regressions(
    predictor: pd.Series,
    returns: pd.Series,
    maxlags: int,
    spline_df: int = 4,
) -> list[dict]:
    frame = _prepare(predictor, returns)
    if len(frame) < 20 or frame["signal"].nunique() < 5:
        return []
    centered = frame["signal"] - frame["signal"].mean()
    linear_x = sm.add_constant(centered.rename("linear"), has_constant="add")
    linear = _fit(frame["return"], linear_x, maxlags)
    rows = [{
        "model_type": "linear",
        "observation_count": len(frame),
        "linear_beta": float(linear.params["linear"]),
        "quadratic_beta": np.nan,
        "raw_p_value": float(linear.pvalues["linear"]),
        "nonlinear_p_value": np.nan,
        "r_squared": float(linear.rsquared),
        "adjusted_r_squared": float(linear.rsquared_adj),
        "aic": float(linear.aic),
        "bic": float(linear.bic),
    }]

    quadratic_x = pd.DataFrame(
        {"linear": centered, "quadratic": centered.pow(2)}, index=frame.index
    )
    quadratic_x = sm.add_constant(quadratic_x, has_constant="add")
    quadratic = _fit(frame["return"], quadratic_x, maxlags)
    rows.append({
        "model_type": "quadratic",
        "observation_count": len(frame),
        "linear_beta": float(quadratic.params["linear"]),
        "quadratic_beta": float(quadratic.params["quadratic"]),
        "raw_p_value": float(quadratic.pvalues["linear"]),
        "nonlinear_p_value": float(quadratic.pvalues["quadratic"]),
        "r_squared": float(quadratic.rsquared),
        "adjusted_r_squared": float(quadratic.rsquared_adj),
        "aic": float(quadratic.aic),
        "bic": float(quadratic.bic),
    })

    spline = dmatrix(
        f"bs(x, df={int(spline_df)}, degree=3, include_intercept=False)",
        {"x": centered.to_numpy()}, return_type="dataframe"
    )
    spline.index = frame.index
    spline = spline.drop(columns=["Intercept"], errors="ignore")
    spline_x = sm.add_constant(spline, has_constant="add")
    spline_result = _fit(frame["return"], spline_x, maxlags)
    restriction = np.zeros((spline.shape[1], spline_x.shape[1]))
    restriction[:, 1:] = np.eye(spline.shape[1])
    joint = spline_result.wald_test(restriction, scalar=True)
    rows.append({
        "model_type": "restricted_cubic_spline",
        "observation_count": len(frame),
        "linear_beta": np.nan,
        "quadratic_beta": np.nan,
        "raw_p_value": float(joint.pvalue),
        "nonlinear_p_value": float(joint.pvalue),
        "r_squared": float(spline_result.rsquared),
        "adjusted_r_squared": float(spline_result.rsquared_adj),
        "aic": float(spline_result.aic),
        "bic": float(spline_result.bic),
    })
    return rows


def add_separate_fdr(
    frame: pd.DataFrame,
    min_n: int,
    family_column: str = "hypothesis_family",
    p_column: str = "raw_p_value",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the sample-size rule before BH correction and retain exclusions."""
    if frame.empty:
        return frame.copy(), frame.copy()
    result = frame.copy()
    reason = pd.Series("", index=result.index, dtype=object)
    reason.loc[result["observation_count"].lt(min_n)] = "observation_count_below_minimum"
    reason.loc[result[p_column].isna()] = "p_value_not_estimable"
    excluded = result.loc[reason.ne("")].copy()
    excluded["exclusion_reason"] = reason.loc[excluded.index]
    eligible = result.loc[reason.eq("")].copy()
    eligible["fdr_global"] = np.nan
    eligible["fdr_family"] = np.nan
    if eligible.empty:
        return eligible, excluded
    eligible["fdr_global"] = multipletests(eligible[p_column], method="fdr_bh")[1]
    for _, indices in eligible.groupby(family_column, dropna=False).groups.items():
        eligible.loc[indices, "fdr_family"] = multipletests(
            eligible.loc[indices, p_column], method="fdr_bh"
        )[1]
    return eligible, excluded


def vif_table(controls: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    frame = controls.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return pd.DataFrame(columns=["variable", "vif"])
    x = sm.add_constant(frame, has_constant="add")
    return pd.DataFrame({
        "variable": x.columns,
        "vif": [variance_inflation_factor(x.to_numpy(), i) for i in range(x.shape[1])],
    })
