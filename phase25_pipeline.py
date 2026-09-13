from __future__ import annotations

import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from config import StudyConfig
from data_loader import RawData
from features import add_normalizations, build_phase2_activity_predictors, build_turnover
from grouping import assign_pr_group, assign_z_group
from phase2_pipeline import CONFIRMATORY_SPECS
from phase2_statistics import hac_signal_prior_interaction, hac_signal_regression
from reporting import (
    build_manifest,
    timestamped_commit_output_directory,
    write_json,
)
from returns import build_forward_returns, build_prior_returns
from study_statistics import summarize_group


# One pre-specified representative per Phase 2 family.  Thresholds are copied
# from Phase 2 CONFIRMATORY_SPECS; Phase 2.5 does not search new definitions.
PHASE25_REPRESENTATIVE_SPECS = (
    ("otc__total_institutional__sell__5d", "pr", 504, "PR_05_20"),
    ("listed__dealer__net__10d", "pr", 756, "PR_95_100"),
    ("otc__dealer__sell__10d", "z", 504, "Z_M2_5_M1_5"),
    ("listed__foreign__net__5d", "pr", 756, "PR_95_100"),
    ("otc__foreign__buy__5d", "z", 756, "Z_P0_5_P1_5"),
    ("combined__foreign__sell__10d", "z", 756, "Z_GE_P2_5"),
)


def _validate_representative_specs() -> None:
    for predictor, method, window, group in PHASE25_REPRESENTATIVE_SPECS:
        matches = [
            spec for spec in CONFIRMATORY_SPECS
            if spec[0] == predictor
            and spec[1] == method
            and window in spec[2]
            and spec[3] == group
        ]
        if not matches:
            raise ValueError(
                "Phase 2.5 representative must come from Phase 2 CONFIRMATORY_SPECS: "
                f"{predictor}, {method}, {window}, {group}"
            )


def _hac_lag(horizon: int) -> int:
    return max(int(horizon) - 1, 0)


def _add_fdr(
    frame: pd.DataFrame,
    p_column: str,
    family_columns: tuple[str, ...] = ("model", "return_type"),
    global_name: str = "q_global",
    family_name: str = "q_family",
) -> pd.DataFrame:
    result = frame.copy()
    result[global_name] = np.nan
    result[family_name] = np.nan
    if result.empty or p_column not in result:
        return result
    valid = result[p_column].notna()
    if "eligible_for_inference" in result:
        valid &= result["eligible_for_inference"].fillna(False).astype(bool)
    if valid.any():
        result.loc[valid, global_name] = multipletests(
            result.loc[valid, p_column], method="fdr_bh"
        )[1]
    present = [column for column in family_columns if column in result]
    if present:
        for indices in result.loc[valid].groupby(present, dropna=False).groups.values():
            result.loc[indices, family_name] = multipletests(
                result.loc[indices, p_column], method="fdr_bh"
            )[1]
    return result


def _candidate_signals(
    normalized: pd.DataFrame,
    horizons: tuple[int, ...],
) -> tuple[pd.DataFrame, list[dict]]:
    records: list[dict] = []
    definitions: list[dict] = []
    for predictor, method, window, group in PHASE25_REPRESENTATIVE_SPECS:
        column = f"{predictor}__rolling_{window}d_{method}"
        if column not in normalized:
            raise KeyError(f"Phase 2.5 candidate feature missing: {column}")
        groups = (
            assign_pr_group(normalized[column])
            if method == "pr"
            else assign_z_group(normalized[column])
        )
        membership = groups.eq(group).astype("boolean").where(groups.notna(), pd.NA)
        candidate_id = f"{predictor}__rolling_{window}d_{method}__{group}"
        definition = {
            "candidate_id": candidate_id,
            "predictor": predictor,
            "normalization_type": f"rolling_{method}",
            "normalization_window": window,
            "group": group,
            "accumulation_window": int(predictor.rsplit("__", 1)[1].removesuffix("d")),
            "market_scope": predictor.split("__", 1)[0],
            "selected_from": "Phase 2 confirmatory representative family",
            "membership": membership,
        }
        definitions.append(definition)
        records.append({
            **{k: v for k, v in definition.items() if k != "membership"},
            "forward_horizons": ",".join(map(str, horizons)),
            "primary_mechanism_horizons": "5,10,20",
        })
    table = pd.DataFrame(records)
    return table, definitions


def _horizon_results(
    candidates: list[dict], returns: pd.DataFrame, horizons: tuple[int, ...], min_group_n: int
) -> pd.DataFrame:
    rows: list[dict] = []
    for item in candidates:
        signal = item["membership"].astype("Float64")
        for horizon in horizons:
            outcome = returns[f"return_o1_c{horizon}"]
            group = summarize_group(outcome, item["membership"], _hac_lag(horizon))
            regression = hac_signal_regression(signal, outcome, _hac_lag(horizon))
            if not group or not regression:
                continue
            rows.append({
                "candidate_id": item["candidate_id"],
                "predictor": item["predictor"],
                "normalization_type": item["normalization_type"],
                "normalization_window": item["normalization_window"],
                "group": item["group"],
                "horizon": horizon,
                "return_type": f"O1_to_C{horizon}",
                "hac_lag": _hac_lag(horizon),
                "observation_count": group["observation_count"],
                "minimum_group_n": min_group_n,
                "eligible_for_inference": group["observation_count"] >= min_group_n,
                "mean_signal_return": group["mean_return"],
                "median_signal_return": group["median_return"],
                "non_group_mean": group["nongroup_mean"],
                "difference_vs_non_group": group["difference_vs_nongroup"],
                "positive_rate": group["positive_rate"],
                "regression_beta": regression["beta"],
                "hac_standard_error": regression["hac_standard_error"],
                "raw_p_value": regression["raw_p_value"],
            })
    return _add_fdr(pd.DataFrame(rows), "raw_p_value", ("return_type",))


def _classify_horizon_pattern(group: pd.DataFrame) -> str:
    values = group.sort_values("horizon")["regression_beta"].dropna()
    if len(values) < 2:
        return "cannot_determine"
    nonzero = values.loc[values.ne(0)]
    if nonzero.empty:
        return "flat"
    signs = np.sign(nonzero)
    if signs.gt(0).any() and signs.lt(0).any():
        return "effect_reversing"
    absolute = values.abs()
    peak_position = int(np.argmax(absolute.to_numpy()))
    if peak_position == len(absolute) - 1:
        return "effect_increasing_to_c20"
    if absolute.iloc[-1] <= 0.5 * absolute.iloc[peak_position]:
        return "effect_decaying_by_c20"
    return "effect_peaking_before_c20"


def _horizon_profile(results: pd.DataFrame) -> pd.DataFrame:
    profile = results.copy()
    if profile.empty:
        profile["horizon_pattern"] = pd.Series(dtype=str)
        return profile
    patterns = {
        candidate_id: _classify_horizon_pattern(group)
        for candidate_id, group in profile.groupby("candidate_id", sort=False)
    }
    profile["horizon_pattern"] = profile["candidate_id"].map(patterns)
    profile["pattern_is_descriptive_only"] = True
    return profile


def _controlled_results(
    candidates: list[dict],
    returns: pd.DataFrame,
    prior: pd.DataFrame,
    horizons: tuple[int, ...],
    prior_window: int | None,
    min_group_n: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    model = "uncontrolled" if prior_window is None else f"prior{prior_window}_controlled"
    for item in candidates:
        signal = item["membership"].astype("Float64")
        for horizon in horizons:
            controls = None
            if prior_window is not None:
                controls = prior[[f"prior_ret_{prior_window}d"]]
            result = hac_signal_regression(
                signal,
                returns[f"return_o1_c{horizon}"],
                _hac_lag(horizon),
                controls,
            )
            if result:
                parts = [signal.rename("signal"), returns[f"return_o1_c{horizon}"].rename("return")]
                if controls is not None:
                    parts.append(controls)
                complete = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
                signal_count = int(complete["signal"].eq(1).sum())
                rows.append({
                    "candidate_id": item["candidate_id"],
                    "predictor": item["predictor"],
                    "normalization_window": item["normalization_window"],
                    "group": item["group"],
                    "model": model,
                    "prior_return_window": prior_window,
                    "horizon": horizon,
                    "return_type": f"O1_to_C{horizon}",
                    "hac_lag": _hac_lag(horizon),
                    "signal_count": signal_count,
                    "minimum_group_n": min_group_n,
                    "eligible_for_inference": signal_count >= min_group_n,
                    **result,
                })
    return _add_fdr(pd.DataFrame(rows), "raw_p_value")


def _interaction_results(
    candidates: list[dict],
    returns: pd.DataFrame,
    prior: pd.DataFrame,
    horizons: tuple[int, ...],
    prior_window: int,
    min_group_n: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    prior_column = f"prior_ret_{prior_window}d"
    for item in candidates:
        signal = item["membership"].astype("Float64")
        for horizon in horizons:
            result = hac_signal_prior_interaction(
                signal,
                returns[f"return_o1_c{horizon}"],
                prior[prior_column],
                _hac_lag(horizon),
            )
            if result:
                complete = pd.concat(
                    [
                        signal.rename("signal"),
                        returns[f"return_o1_c{horizon}"].rename("return"),
                        prior[prior_column],
                    ],
                    axis=1,
                ).replace([np.inf, -np.inf], np.nan).dropna()
                signal_count = int(complete["signal"].eq(1).sum())
                rows.append({
                    "candidate_id": item["candidate_id"],
                    "predictor": item["predictor"],
                    "normalization_window": item["normalization_window"],
                    "group": item["group"],
                    "model": f"prior{prior_window}_interaction",
                    "prior_return_window": prior_window,
                    "horizon": horizon,
                    "return_type": f"O1_to_C{horizon}",
                    "hac_lag": _hac_lag(horizon),
                    "signal_count": signal_count,
                    "minimum_group_n": min_group_n,
                    "eligible_for_inference": signal_count >= min_group_n,
                    **result,
                })
    frame = _add_fdr(
        pd.DataFrame(rows), "signal_p", ("return_type",), "signal_q_global", "signal_q_family"
    )
    return _add_fdr(
        frame, "interaction_p", ("return_type",), "interaction_q_global", "interaction_q_family"
    )


def _attenuation(
    uncontrolled: pd.DataFrame,
    prior5: pd.DataFrame,
    prior20: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["candidate_id", "horizon", "return_type"]
    base = uncontrolled[keys + ["beta", "signal_count", "eligible_for_inference"]].rename(
        columns={"beta": "beta_uncontrolled"}
    )
    for window, frame in ((5, prior5), (20, prior20)):
        base = base.merge(
            frame[keys + ["beta"]].rename(columns={"beta": f"beta_prior{window}_controlled"}),
            on=keys,
            how="left",
        )
    for window in (5, 20):
        controlled = f"beta_prior{window}_controlled"
        base[f"absolute_beta_change_prior{window}"] = base[controlled] - base["beta_uncontrolled"]
        denominator = base["beta_uncontrolled"].abs()
        stable = denominator.gt(1e-12)
        base[f"relative_beta_change_prior{window}"] = np.where(
            stable,
            (base[controlled] - base["beta_uncontrolled"]) / denominator,
            np.nan,
        )
        base[f"attenuation_fraction_prior{window}"] = np.where(
            stable, 1 - base[controlled].abs() / denominator, np.nan
        )
        base[f"sign_changed_prior{window}"] = (
            np.sign(base[controlled]) != np.sign(base["beta_uncontrolled"])
        ) & base[controlled].notna()
        base[f"sign_preserved_prior{window}"] = ~base[f"sign_changed_prior{window}"]
        base[f"unstable_denominator_prior{window}"] = ~stable
    return base


def _prior_descriptive(
    candidates: list[dict], returns: pd.DataFrame, prior: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict] = []
    for item in candidates:
        for prior_window in (5, 20):
            prior_column = f"prior_ret_{prior_window}d"
            for horizon in (5, 10, 20):
                frame = pd.concat(
                    [
                        item["membership"].rename("membership"),
                        prior[prior_column],
                        returns[f"return_o1_c{horizon}"].rename("return"),
                    ],
                    axis=1,
                ).dropna()
                frame = frame.loc[frame["membership"].astype(bool)]
                if len(frame) < 4:
                    continue
                frame["prior_quantile"] = pd.qcut(
                    frame[prior_column], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
                )
                for quantile, values in frame.groupby("prior_quantile", observed=True):
                    rows.append({
                        "candidate_id": item["candidate_id"],
                        "prior_return_window": prior_window,
                        "prior_quantile": str(quantile),
                        "horizon": horizon,
                        "return_type": f"O1_to_C{horizon}",
                        "observation_count": len(values),
                        "prior_return_mean": values[prior_column].mean(),
                        "mean_return": values["return"].mean(),
                        "median_return": values["return"].median(),
                        "positive_rate": values["return"].gt(0).mean(),
                        "descriptive_only": True,
                    })
    return pd.DataFrame(rows)


def _temporal_robustness(
    candidates: list[dict], returns: pd.DataFrame
) -> pd.DataFrame:
    """Describe subperiod and leave-one-year-out stability for primary horizons."""
    periods = {
        "2018_2019": (2018, 2019),
        "2020_2022": (2020, 2022),
        "2023_latest": (2023, 9999),
    }
    rows: list[dict] = []
    for item in candidates:
        for horizon in (5, 10, 20):
            frame = pd.concat(
                [
                    item["membership"].rename("membership"),
                    returns[f"return_o1_c{horizon}"].rename("return"),
                ],
                axis=1,
            ).dropna()
            values = frame.loc[frame["membership"].astype(bool), "return"]
            for label, (start, end) in periods.items():
                subset = values.loc[(values.index.year >= start) & (values.index.year <= end)]
                rows.append({
                    "candidate_id": item["candidate_id"],
                    "horizon": horizon,
                    "diagnostic_type": "subperiod",
                    "diagnostic_label": label,
                    "observation_count": len(subset),
                    "mean_return": subset.mean(),
                    "positive_rate": subset.gt(0).mean() if len(subset) else np.nan,
                })
            for year in sorted(values.index.year.unique()):
                subset = values.loc[values.index.year != year]
                rows.append({
                    "candidate_id": item["candidate_id"],
                    "horizon": horizon,
                    "diagnostic_type": "leave_one_year_out",
                    "diagnostic_label": str(year),
                    "observation_count": len(subset),
                    "mean_return": subset.mean(),
                    "positive_rate": subset.gt(0).mean() if len(subset) else np.nan,
                })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    for (candidate_id, horizon), indices in result.groupby(["candidate_id", "horizon"]).groups.items():
        sub = result.loc[indices]
        subperiod = sub.loc[sub["diagnostic_type"].eq("subperiod")].dropna(subset=["mean_return"])
        loo = sub.loc[sub["diagnostic_type"].eq("leave_one_year_out")].dropna(subset=["mean_return"])
        result.loc[indices, "subperiod_sign_flip"] = bool(
            subperiod["mean_return"].gt(0).any() and subperiod["mean_return"].lt(0).any()
        )
        result.loc[indices, "leave_one_year_out_sign_flip"] = bool(
            loo["mean_return"].gt(0).any() and loo["mean_return"].lt(0).any()
        )
    return result


def _mechanism_classification(attenuation: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_id, group in attenuation.groupby("candidate_id", sort=False):
        primary = group.loc[group["horizon"].isin((5, 10, 20))]
        eligible_primary = primary.loc[primary["eligible_for_inference"].fillna(False)]
        values = eligible_primary[["attenuation_fraction_prior5", "attenuation_fraction_prior20"]].to_numpy().ravel()
        values = values[np.isfinite(values)]
        interaction = interactions.loc[
            interactions["candidate_id"].eq(candidate_id)
            & interactions["horizon"].isin((5, 10, 20))
        ]
        if interaction.get("interaction_q_global", pd.Series(dtype=float)).lt(0.05).any():
            classification = "Prior-return interaction dependent"
        elif len(values) == 0:
            classification = "Unstable / cannot determine"
        elif np.nanmedian(values) >= 0.5:
            classification = "Strongly explained by prior return"
        elif np.nanmedian(values) >= 0.2:
            classification = "Partially explained by prior return"
        else:
            classification = "Independent / mostly independent"
        rows.append({
            "candidate_id": candidate_id,
            "mechanism_classification": classification,
            "classification_is_descriptive": True,
        })
    return pd.DataFrame(rows)


def _final_decisions(
    attenuation: pd.DataFrame,
    controlled5: pd.DataFrame,
    controlled20: pd.DataFrame,
    interactions: pd.DataFrame,
    temporal: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for candidate_id, group in attenuation.groupby("candidate_id", sort=False):
        primary = group.loc[group["horizon"].isin((5, 10, 20))]
        eligible = bool(primary["eligible_for_inference"].all()) and len(primary) == 3
        sign_changed = bool(
            primary[["sign_changed_prior5", "sign_changed_prior20"]].any().any()
        )
        c5 = controlled5.loc[
            controlled5["candidate_id"].eq(candidate_id)
            & controlled5["horizon"].isin((5, 10, 20))
        ]
        c20 = controlled20.loc[
            controlled20["candidate_id"].eq(candidate_id)
            & controlled20["horizon"].isin((5, 10, 20))
        ]
        controlled_evidence = bool(
            c5.get("q_global", pd.Series(dtype=float)).lt(0.05).any()
            or c20.get("q_global", pd.Series(dtype=float)).lt(0.05).any()
        )
        interaction = interactions.loc[
            interactions["candidate_id"].eq(candidate_id)
            & interactions["horizon"].isin((5, 10, 20))
        ]
        interaction_dependent = bool(
            interaction.get("interaction_q_global", pd.Series(dtype=float)).lt(0.05).any()
        )
        time = temporal.loc[
            temporal["candidate_id"].eq(candidate_id)
            & temporal["horizon"].isin((5, 10, 20))
        ]
        unstable_time = bool(
            time.get("leave_one_year_out_sign_flip", pd.Series(dtype=bool)).fillna(False).any()
        )
        if eligible and not sign_changed and not interaction_dependent and not unstable_time and controlled_evidence:
            decision = "保留"
            reason = "primary horizons樣本足夠，控制後仍有FDR證據，且未見sign reversal或逐年剔除翻向"
        elif sign_changed and unstable_time and not controlled_evidence:
            decision = "淘汰"
            reason = "控制後與時間穩健性均不支持原方向"
        else:
            decision = "修改後再測"
            reason = "樣本、控制後證據、interaction或時間穩健性至少一項仍有限制"
        rows.append({"candidate_id": candidate_id, "decision": decision, "decision_reason": reason})
    return pd.DataFrame(rows)


def _write_summary(
    path: Path,
    horizon: pd.DataFrame,
    attenuation: pd.DataFrame,
    interactions: pd.DataFrame,
    controlled5: pd.DataFrame,
    controlled20: pd.DataFrame,
    temporal: pd.DataFrame,
) -> None:
    patterns = horizon[["candidate_id", "horizon_pattern"]].drop_duplicates()
    mechanism = _mechanism_classification(attenuation, interactions)
    decisions = _final_decisions(
        attenuation, controlled5, controlled20, interactions, temporal
    )
    lines = [
        "# Phase 2.5 — Prior Return Mechanism + Longer Forward Horizon",
        "",
        "## A. C20 extension",
        "",
    ]
    for row in patterns.itertuples(index=False):
        lines.append(f"- `{row.candidate_id}`：{row.horizon_pattern}")
    lines += [
        "",
        "## B–D. Prior-return control and mechanism classification",
        "",
    ]
    for row in mechanism.itertuples(index=False):
        lines.append(f"- `{row.candidate_id}`：{row.mechanism_classification}")
    lines += ["", "### C5／C10／C20 control與interaction稽核", ""]
    for candidate_id in attenuation["candidate_id"].drop_duplicates():
        parts = []
        for horizon_value in (5, 10, 20):
            a = attenuation.loc[
                attenuation["candidate_id"].eq(candidate_id)
                & attenuation["horizon"].eq(horizon_value)
            ]
            if a.empty:
                continue
            row = a.iloc[0]
            parts.append(
                f"C{horizon_value}: beta={row['beta_uncontrolled']:.4g}, "
                f"prior5={row['beta_prior5_controlled']:.4g}, "
                f"prior20={row['beta_prior20_controlled']:.4g}"
            )
        lines.append(f"- `{candidate_id}` — " + "; ".join(parts))
    lines += [
        "",
        "## E. Final research decision",
        "",
    ]
    for row in decisions.itertuples(index=False):
        lines.append(f"- `{row.candidate_id}`：**{row.decision}** — {row.decision_reason}")
    lines += [
        "",
        "以上判定是預先定義的描述性研究分類，不等同策略已驗證。",
        "",
        "> Phase 2.5 仍使用 Phase 1 / Phase 2 已經看過的歷史樣本，",
        "> 因此不是 untouched out-of-sample validation。只能說 mechanism evidence strengthened / weakened。",
        "",
        "## Research safeguards",
        "",
        "- Signal d0 after close；最早可交易時間為下一交易日 adjusted open O1。",
        "- prior return只使用不晚於d0的adjusted close交易列。",
        "- C20使用Newey–West HAC maxlags=19。",
        "- 未新增法人20日累積、normalization window或rotation outcome。",
        "- Interaction只套用Phase 2代表性candidate families。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _write_run_info(path: Path, output: Path, config: StudyConfig, tests_passed: bool | None) -> None:
    manifest = json.loads((output / "phase25_run_metadata.json").read_text(encoding="utf-8"))
    lines = [
        "Repository: hh4832/-institutional-spot-flow-study",
        f"Branch: {manifest.get('git_branch')}",
        f"Git commit: {manifest.get('git_commit_hash')}",
        f"Run timestamp: {manifest.get('created_at_utc')}",
        "Timezone: Asia/Taipei",
        f"Python version: {platform.python_version()}",
        f"FinLab version: {_package_version('finlab')}",
        "Study mode: phase25",
        "Price source open: etl:adj_open",
        "Price source close: etl:adj_close",
        "Accumulation windows: 1,5,10",
        "Prior return windows: 5,20",
        "Forward horizons: 1,2,3,5,10,20",
        "Primary mechanism horizons: 5,10,20",
        f"Drive output folder: {output.name}",
        f"Tests passed: {str(tests_passed).lower() if tests_passed is not None else 'not_recorded'}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase25_study(
    raw: RawData,
    config: StudyConfig | None = None,
    tests_passed: bool | None = None,
) -> Path:
    config = config or StudyConfig(study_mode="phase25_prior_return_mechanism")
    _validate_representative_specs()
    if tuple(config.accumulation_windows) != (1, 5, 10):
        raise ValueError("Phase 2.5 法人 accumulation windows 必須維持 1/5/10")
    horizons = tuple(config.phase25_return_horizons)
    if horizons != (1, 2, 3, 5, 10, 20):
        raise ValueError("Phase 2.5 forward horizons 必須為 1/2/3/5/10/20")
    if raw.price_dataset_names != {"open": "etl:adj_open", "close": "etl:adj_close"}:
        raise RuntimeError("Phase 2.5 必須直接使用 etl:adj_open 與 etl:adj_close")

    turnover = build_turnover(raw.market_amount)
    levels = build_phase2_activity_predictors(
        raw.institutional_buy,
        raw.institutional_sell,
        raw.institutional_net,
        turnover,
        config.accumulation_windows,
    )
    normalized = add_normalizations(
        levels,
        windows=tuple(config.normalization_windows),
        include_global=False,
    )
    returns = build_forward_returns(
        raw.adjusted_open, raw.adjusted_close, levels.index, horizons=horizons
    )
    prior = build_prior_returns(
        raw.adjusted_close,
        levels.index,
        windows=tuple(config.phase25_prior_return_windows),
    )
    candidate_table, candidates = _candidate_signals(normalized, horizons)
    min_group_n = int(config.phase2_min_group_n)
    horizon_results = _horizon_results(candidates, returns, horizons, min_group_n)
    profile = _horizon_profile(horizon_results)
    uncontrolled = _controlled_results(candidates, returns, prior, horizons, None, min_group_n)
    prior5 = _controlled_results(candidates, returns, prior, horizons, 5, min_group_n)
    prior20 = _controlled_results(candidates, returns, prior, horizons, 20, min_group_n)
    interaction5 = _interaction_results(candidates, returns, prior, horizons, 5, min_group_n)
    interaction20 = _interaction_results(candidates, returns, prior, horizons, 20, min_group_n)
    attenuation = _attenuation(uncontrolled, prior5, prior20)
    descriptive = _prior_descriptive(candidates, returns, prior)
    temporal = _temporal_robustness(candidates, returns)
    significant = pd.concat(
        [
            prior5.loc[prior5.get("q_global", pd.Series(dtype=float)).lt(0.05)],
            prior20.loc[prior20.get("q_global", pd.Series(dtype=float)).lt(0.05)],
            interaction5.loc[interaction5.get("interaction_q_global", pd.Series(dtype=float)).lt(0.05)],
            interaction20.loc[interaction20.get("interaction_q_global", pd.Series(dtype=float)).lt(0.05)],
        ],
        ignore_index=True,
        sort=False,
    )

    output = timestamped_commit_output_directory(config.output_root)
    candidate_table.to_csv(output / "phase25_candidate_signals.csv", index=False)
    horizon_results.to_csv(output / "phase25_forward_horizon_results.csv", index=False)
    profile.to_csv(output / "phase25_horizon_profile.csv", index=False)
    prior5.to_csv(output / "phase25_prior5_controlled_regressions.csv", index=False)
    prior20.to_csv(output / "phase25_prior20_controlled_regressions.csv", index=False)
    interaction5.to_csv(output / "phase25_prior5_interactions.csv", index=False)
    interaction20.to_csv(output / "phase25_prior20_interactions.csv", index=False)
    attenuation.to_csv(output / "phase25_effect_attenuation.csv", index=False)
    descriptive.to_csv(output / "phase25_prior_return_descriptive.csv", index=False)
    temporal.to_csv(output / "phase25_temporal_robustness.csv", index=False)
    significant.to_csv(output / "phase25_significant_results.csv", index=False)
    pd.concat([levels, normalized, prior, returns], axis=1).to_parquet(
        output / "phase25_analysis_dataset.parquet"
    )
    _write_summary(
        output / "phase25_summary.md",
        profile,
        attenuation,
        pd.concat([interaction5, interaction20]),
        prior5,
        prior20,
        temporal,
    )
    metadata = {
        "repository": "hh4832/-institutional-spot-flow-study",
        "study_mode": "phase25_prior_return_mechanism",
        "timezone": "Asia/Taipei",
        "price_source_open": raw.price_dataset_names["open"],
        "price_source_close": raw.price_dataset_names["close"],
        "outcome_price_adjusted": True,
        "outcome_definition": "signal d0 after close; O1=adjusted_open[d0+1]; Ck=adjusted_close[d0+k] on trading rows",
        "accumulation_windows": list(config.accumulation_windows),
        "prior_return_windows": list(config.phase25_prior_return_windows),
        "return_horizons": list(horizons),
        "primary_mechanism_horizons": list(config.phase25_primary_mechanism_horizons),
        "rotation_study_included": False,
        "data_start": str(levels.index.min()),
        "data_end": str(levels.index.max()),
        "sample_size": int(len(levels)),
        "candidate_hypothesis_count": int(len(candidate_table)),
        "candidate_family_count": int(len(candidates)),
        "fdr_family_counts": {
            "controlled_prior5": int(len(prior5)),
            "controlled_prior20": int(len(prior20)),
            "interaction_prior5": int(len(interaction5)),
            "interaction_prior20": int(len(interaction20)),
        },
        "hac_lag_rule": "max(horizon - 1, 0)",
        "c20_hac_lag": 19,
        "phase1_phase2_history_reused": True,
        "untouched_out_of_sample_validation": False,
    }
    write_json(
        output / "phase25_run_metadata.json",
        build_manifest(config.to_dict(), metadata),
    )
    write_json(output / "phase25_config_snapshot.json", config.to_dict())
    _write_run_info(output / "run_info.txt", output, config, tests_passed)
    return output
