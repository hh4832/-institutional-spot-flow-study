from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests


@dataclass
class HACEstimate:
    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    p_value: float


def _fit_hac(y: pd.Series, x: pd.DataFrame, maxlags: int):
    model = sm.OLS(y.astype(float), x.astype(float), missing="drop")
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": int(maxlags)})


def _estimate_from_result(result, index: int, scale: float = 1.0) -> HACEstimate:
    estimate = float(result.params.iloc[index] * scale)
    standard_error = float(result.bse.iloc[index] * abs(scale))
    z = stats.norm.ppf(0.975)
    if standard_error == 0 or np.isnan(standard_error):
        p_value = np.nan
    else:
        p_value = float(2 * stats.norm.sf(abs(estimate / standard_error)))
    return HACEstimate(
        estimate=estimate,
        standard_error=standard_error,
        ci_low=estimate - z * standard_error,
        ci_high=estimate + z * standard_error,
        p_value=p_value,
    )


def hac_mean(series: pd.Series, maxlags: int, null_value: float = 0.0) -> HACEstimate:
    y = series.dropna().astype(float)
    if len(y) < 3:
        return HACEstimate(*(np.nan,) * 5)
    x = pd.DataFrame({"const": 1.0}, index=y.index)
    result = _fit_hac(y - null_value, x, maxlags)
    return _estimate_from_result(result, 0)


def pooled_cohens_d(group: pd.Series, nongroup: pd.Series) -> float:
    group = group.dropna().astype(float)
    nongroup = nongroup.dropna().astype(float)
    if len(group) < 2 or len(nongroup) < 2:
        return np.nan
    pooled_variance = (
        (len(group) - 1) * group.var(ddof=1)
        + (len(nongroup) - 1) * nongroup.var(ddof=1)
    ) / (len(group) + len(nongroup) - 2)
    if pooled_variance <= 0:
        return np.nan
    return float((group.mean() - nongroup.mean()) / np.sqrt(pooled_variance))


def summarize_group(
    returns: pd.Series,
    membership: pd.Series,
    maxlags: int,
) -> dict:
    frame = pd.concat(
        [returns.rename("return"), membership.rename("group")], axis=1
    ).dropna()
    if frame.empty or frame["group"].nunique() < 2:
        return {}
    frame["group"] = frame["group"].astype(bool)
    group = frame.loc[frame["group"], "return"]
    nongroup = frame.loc[~frame["group"], "return"]
    if len(group) < 3 or len(nongroup) < 3:
        return {}

    mean_zero = hac_mean(group, maxlags=maxlags)
    design = sm.add_constant(frame["group"].astype(float), has_constant="add")
    group_model = _fit_hac(frame["return"], design, maxlags)
    difference_nongroup = _estimate_from_result(group_model, 1)

    # E(group)-E(all) = (1-p_group) * [E(group)-E(non-group)].
    # This is an HAC contrast and does not treat the unconditional mean as known.
    group_share = float(frame["group"].mean())
    difference_unconditional = _estimate_from_result(
        group_model, 1, scale=1.0 - group_share
    )

    positive = frame["return"].gt(0).astype(float)
    positive_group = positive[frame["group"]]
    positive_vs_half = hac_mean(positive_group, maxlags=maxlags, null_value=0.5)
    positive_model = _fit_hac(positive, design, maxlags)
    positive_vs_unconditional = _estimate_from_result(
        positive_model, 1, scale=1.0 - group_share
    )

    quantiles = group.quantile([0.05, 0.25, 0.75, 0.95])
    return {
        "observation_count": int(len(group)),
        "nongroup_count": int(len(nongroup)),
        "unconditional_count": int(len(frame)),
        "mean_return": float(group.mean()),
        "median_return": float(group.median()),
        "standard_deviation": float(group.std(ddof=1)),
        "standard_error_hac": mean_zero.standard_error,
        "positive_count": int(group.gt(0).sum()),
        "negative_count": int(group.lt(0).sum()),
        "zero_count": int(group.eq(0).sum()),
        "positive_rate": float(group.gt(0).mean()),
        "negative_rate": float(group.lt(0).mean()),
        "unconditional_mean": float(frame["return"].mean()),
        "nongroup_mean": float(nongroup.mean()),
        "difference_vs_zero": mean_zero.estimate,
        "zero_ci_low": mean_zero.ci_low,
        "zero_ci_high": mean_zero.ci_high,
        "zero_p_value": mean_zero.p_value,
        "difference_vs_unconditional": difference_unconditional.estimate,
        "unconditional_ci_low": difference_unconditional.ci_low,
        "unconditional_ci_high": difference_unconditional.ci_high,
        "unconditional_p_value": difference_unconditional.p_value,
        "difference_vs_nongroup": difference_nongroup.estimate,
        "nongroup_ci_low": difference_nongroup.ci_low,
        "nongroup_ci_high": difference_nongroup.ci_high,
        "nongroup_p_value": difference_nongroup.p_value,
        "positive_vs_50_difference": positive_vs_half.estimate,
        "positive_vs_50_ci_low": positive_vs_half.ci_low,
        "positive_vs_50_ci_high": positive_vs_half.ci_high,
        "positive_vs_50_p_value": positive_vs_half.p_value,
        "positive_vs_unconditional_difference": positive_vs_unconditional.estimate,
        "positive_vs_unconditional_ci_low": positive_vs_unconditional.ci_low,
        "positive_vs_unconditional_ci_high": positive_vs_unconditional.ci_high,
        "positive_vs_unconditional_p_value": positive_vs_unconditional.p_value,
        "cohens_d": pooled_cohens_d(group, nongroup),
        "minimum": float(group.min()),
        "maximum": float(group.max()),
        "p05": float(quantiles.loc[0.05]),
        "p25": float(quantiles.loc[0.25]),
        "p75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
    }


def continuous_hac_regression(
    predictor: pd.Series,
    returns: pd.Series,
    maxlags: int,
    controls: pd.DataFrame | None = None,
) -> dict:
    parts = [predictor.rename("predictor"), returns.rename("return")]
    if controls is not None:
        parts.append(controls)
    frame = pd.concat(parts, axis=1).dropna()
    if len(frame) < 10 or frame["predictor"].nunique() < 2:
        return {}
    x_columns = ["predictor"] + (
        list(controls.columns) if controls is not None else []
    )
    design = sm.add_constant(frame[x_columns], has_constant="add")
    result = _fit_hac(frame["return"], design, maxlags)
    estimate = _estimate_from_result(result, 1)
    return {
        "observation_count": int(len(frame)),
        "beta": estimate.estimate,
        "hac_standard_error": estimate.standard_error,
        "ci_low": estimate.ci_low,
        "ci_high": estimate.ci_high,
        "raw_p_value": estimate.p_value,
        "r_squared": float(result.rsquared),
    }


def add_fdr_columns(
    frame: pd.DataFrame,
    p_columns: tuple[str, ...] = (
        "zero_p_value",
        "unconditional_p_value",
        "nongroup_p_value",
        "positive_vs_50_p_value",
        "positive_vs_unconditional_p_value",
    ),
) -> pd.DataFrame:
    result = frame.copy()
    family_columns = [
        column
        for column in [
            "market_scope",
            "institution",
            "flow_type",
            "accumulation_window",
            "normalization_type",
            "return_type",
        ]
        if column in result.columns
    ]
    for p_column in p_columns:
        if p_column not in result:
            continue
        valid = result[p_column].notna()
        adjusted = pd.Series(np.nan, index=result.index, dtype=float)
        if valid.any():
            adjusted.loc[valid] = multipletests(
                result.loc[valid, p_column], method="fdr_bh"
            )[1]
        result[f"{p_column}_fdr_global"] = adjusted
        family_adjusted = pd.Series(np.nan, index=result.index, dtype=float)
        if family_columns:
            grouped = result.loc[valid].groupby(
                family_columns, dropna=False, sort=False
            ).groups
            for indices in grouped.values():
                family_adjusted.loc[indices] = multipletests(
                    result.loc[indices, p_column], method="fdr_bh"
                )[1]
        result[f"{p_column}_fdr_family"] = family_adjusted
        result[f"{p_column}_significant_raw_05"] = result[p_column] < 0.05
        result[f"{p_column}_significant_fdr_05"] = adjusted < 0.05
        result[f"{p_column}_significant_fdr_family_05"] = (
            family_adjusted < 0.05
        )
    return result
