from __future__ import annotations

import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from config import (
    OTC_INDEX_DATASET,
    OTC_PRICE_INDEX_SERIES,
    OTC_TOTAL_RETURN_SERIES,
    StudyConfig,
)
from data_loader import RawData
from features import add_normalizations, build_phase2_activity_predictors, build_turnover
from grouping import assign_pr_group, assign_z_group
from phase2_statistics import hac_signal_regression
from reporting import build_manifest, timestamped_commit_output_directory, write_json
from study_statistics import summarize_group


PHASE3_CANDIDATE_SPECS = (
    ("otc__foreign__buy__5d", "z", 756, "Z_P0_5_P1_5", "OTC foreign Buy", "primary"),
    ("listed__foreign__net__5d", "pr", 756, "PR_95_100", "Listed foreign Net", "primary"),
    ("combined__foreign__sell__10d", "z", 756, "Z_GE_P2_5", "Extreme foreign Sell", "secondary"),
    ("listed__dealer__net__10d", "pr", 756, "PR_95_100", "Dealer secondary", "secondary"),
    ("otc__dealer__sell__10d", "z", 504, "Z_M2_5_M1_5", "Dealer secondary", "secondary"),
)

OUTCOME_LABELS = {
    "0050": "0050 adjusted return",
    "OTC_TR": "OTC total-return",
    "ROT_OTC_MINUS_0050": "OTC total-return minus 0050",
}


def _hac_lag(horizon: int) -> int:
    return max(int(horizon) - 1, 0)


def build_phase3_outcomes(
    adjusted_close: pd.Series,
    otc_total_return: pd.Series,
    otc_price_index: pd.Series,
    signal_index: pd.DatetimeIndex,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
) -> pd.DataFrame:
    """Build C0-to-Ck mechanism outcomes on common trading rows.

    C0 is the signal-date close.  These outcomes are deliberately marked as
    non-tradable because the d0 institutional signal is only complete after C0.
    """
    prices = pd.concat(
        [
            pd.to_numeric(adjusted_close, errors="coerce").rename("0050_close"),
            pd.to_numeric(otc_total_return, errors="coerce").rename("otc_total_return"),
            pd.to_numeric(otc_price_index, errors="coerce").rename("otc_price_index"),
        ],
        axis=1,
    ).sort_index()
    prices = prices.loc[~prices.index.duplicated(keep="last")].dropna()
    if prices.empty or (prices <= 0).any().any():
        raise ValueError("Phase 3 price/index series must be non-empty and positive")
    outcome = pd.DataFrame(index=pd.DatetimeIndex(signal_index).sort_values())
    outcome.index.name = "signal_date"
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError("Phase 3 horizons must be positive")
        exit_date = pd.Series(prices.index, index=prices.index).shift(-horizon)
        r0050 = prices["0050_close"].shift(-horizon).div(prices["0050_close"]) - 1
        rotc = prices["otc_total_return"].shift(-horizon).div(prices["otc_total_return"]) - 1
        rotc_price = prices["otc_price_index"].shift(-horizon).div(prices["otc_price_index"]) - 1
        outcome[f"exit_date_C{horizon}"] = exit_date.reindex(outcome.index)
        outcome[f"0050_C0_C{horizon}"] = r0050.reindex(outcome.index)
        outcome[f"OTC_TR_C0_C{horizon}"] = rotc.reindex(outcome.index)
        outcome[f"ROT_OTC_MINUS_0050_C0_C{horizon}"] = (rotc - r0050).reindex(outcome.index)
        outcome[f"OTC_PRICE_C0_C{horizon}"] = rotc_price.reindex(outcome.index)
        outcome[f"ROT_OTC_PRICE_MINUS_0050_C0_C{horizon}"] = (
            rotc_price - r0050
        ).reindex(outcome.index)
    return outcome


def _index_audit(raw: RawData) -> pd.DataFrame:
    records = []
    for name, series in (
        (OTC_TOTAL_RETURN_SERIES, raw.otc_total_return_index),
        (OTC_PRICE_INDEX_SERIES, raw.otc_price_index),
    ):
        if series is None:
            raise RuntimeError(f"Phase 3 missing required OTC series: {name}")
        index = pd.DatetimeIndex(series.index)
        records.append({
            "series_name": name,
            "first_date": index.min(),
            "last_date": index.max(),
            "n_obs": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "duplicate_dates": int(index.duplicated().sum()),
            "monotonic_increasing": bool(index.is_monotonic_increasing),
        })
    overlap = pd.concat(
        [raw.adjusted_close, raw.otc_total_return_index, raw.otc_price_index], axis=1
    ).dropna()
    for record in records:
        record["overlap_first_date"] = overlap.index.min()
        record["overlap_last_date"] = overlap.index.max()
        record["overlap_n_obs_with_0050"] = len(overlap)
    return pd.DataFrame(records)


def _candidate_signals(normalized: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    rows, candidates = [], []
    for predictor, method, window, group, family, role in PHASE3_CANDIDATE_SPECS:
        column = f"{predictor}__rolling_{window}d_{method}"
        if column not in normalized:
            raise KeyError(f"Phase 3 candidate feature missing: {column}")
        assigned = assign_pr_group(normalized[column]) if method == "pr" else assign_z_group(normalized[column])
        membership = assigned.eq(group).astype("boolean").where(assigned.notna(), pd.NA)
        candidate_id = f"{predictor}__rolling_{window}d_{method}__{group}"
        row = {
            "candidate_id": candidate_id,
            "predictor": predictor,
            "normalization_type": f"rolling_{method}",
            "normalization_window": window,
            "group": group,
            "hypothesis_family": family,
            "analysis_role": role,
            "accumulation_window": int(predictor.rsplit("__", 1)[1].removesuffix("d")),
            "selected_from": "Phase 2/2.5 pre-specified candidate",
        }
        rows.append(row)
        candidates.append({**row, "membership": membership})
    return pd.DataFrame(rows), candidates


def _apply_fdr(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["fdr_global"] = np.nan
    result["fdr_family"] = np.nan
    if result.empty:
        return result
    valid = result["raw_p_value"].notna() & result["eligible_for_inference"]
    if valid.any():
        result.loc[valid, "fdr_global"] = multipletests(
            result.loc[valid, "raw_p_value"], method="fdr_bh"
        )[1]
        for indices in result.loc[valid].groupby("hypothesis_family").groups.values():
            result.loc[indices, "fdr_family"] = multipletests(
                result.loc[indices, "raw_p_value"], method="fdr_bh"
            )[1]
    return result


def _regression_results(
    candidates: list[dict], outcomes: pd.DataFrame, horizons: tuple[int, ...],
    outcome_prefixes: tuple[str, ...], min_group_n: int,
) -> pd.DataFrame:
    rows = []
    for item in candidates:
        signal = item["membership"].astype("Float64")
        for horizon in horizons:
            for prefix in outcome_prefixes:
                column = f"{prefix}_C0_C{horizon}"
                group = summarize_group(outcomes[column], item["membership"], _hac_lag(horizon))
                regression = hac_signal_regression(signal, outcomes[column], _hac_lag(horizon))
                if not group or not regression:
                    continue
                rows.append({
                    "candidate_id": item["candidate_id"],
                    "hypothesis_family": item["hypothesis_family"],
                    "analysis_role": item["analysis_role"],
                    "outcome": column,
                    "outcome_family": prefix,
                    "horizon": horizon,
                    "hac_lag": _hac_lag(horizon),
                    "signal_group_n": group["observation_count"],
                    "mean": group["mean_return"],
                    "median": group["median_return"],
                    "positive_rate": group["positive_rate"],
                    "non_group_mean": group["nongroup_mean"],
                    "difference": group["difference_vs_nongroup"],
                    "regression_beta": regression["beta"],
                    "hac_se": regression["hac_standard_error"],
                    "raw_p_value": regression["raw_p_value"],
                    "minimum_group_n": min_group_n,
                    "eligible_for_inference": (
                        group["observation_count"] >= min_group_n and horizon in (5, 10, 20)
                    ),
                    "tradable_outcome": False,
                })
    return _apply_fdr(pd.DataFrame(rows))


def _temporal_robustness(candidates: list[dict], outcomes: pd.DataFrame) -> pd.DataFrame:
    periods = {"2018_2019": (2018, 2019), "2020_2022": (2020, 2022), "2023_latest": (2023, 9999)}
    rows = []
    for item in candidates:
        for horizon in (5, 10, 20):
            column = f"ROT_OTC_MINUS_0050_C0_C{horizon}"
            frame = pd.concat([item["membership"].rename("signal"), outcomes[column]], axis=1).dropna()
            values = frame.loc[frame["signal"].astype(bool), column]
            chunks = []
            for label, (start, end) in periods.items():
                mask = (values.index.year >= start) & (values.index.year <= end)
                chunks.append(("subperiod", label, values.loc[mask]))
            for year in sorted(values.index.year.unique()):
                chunks.append(("leave_one_year_out", str(year), values.loc[values.index.year != year]))
            means = [chunk.mean() for _, _, chunk in chunks if len(chunk)]
            sign_flip = bool(any(value > 0 for value in means) and any(value < 0 for value in means))
            for kind, label, chunk in chunks:
                rows.append({
                    "candidate_id": item["candidate_id"], "hypothesis_family": item["hypothesis_family"],
                    "horizon": horizon, "diagnostic_type": kind, "diagnostic_label": label,
                    "sample_size": len(chunk), "mean_relative_return": chunk.mean(),
                    "positive_rate": chunk.gt(0).mean() if len(chunk) else np.nan,
                    "sign_flip": sign_flip,
                })
    return pd.DataFrame(rows)


def _classifications(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate_id, horizon), group in primary.groupby(["candidate_id", "horizon"]):
        by_outcome = group.set_index("outcome_family")
        relative = by_outcome.loc["ROT_OTC_MINUS_0050"]
        otc = by_outcome.loc["OTC_TR"]
        listed = by_outcome.loc["0050"]
        significant = bool(relative["fdr_global"] < 0.05)
        if not significant:
            label = "No Evidence"
        elif relative["regression_beta"] < 0:
            label = "Opposite Rotation"
        elif otc["regression_beta"] > 0 and otc["fdr_global"] < 0.05:
            label = "Rotation Supported"
        elif otc["fdr_global"] >= 0.05 or pd.isna(otc["fdr_global"]):
            label = "Relative Rotation Only"
        elif np.sign(otc["regression_beta"]) == np.sign(listed["regression_beta"]):
            label = "Market-wide Effect"
        else:
            label = "No Evidence"
        rows.append({"candidate_id": candidate_id, "horizon": horizon, "rotation_classification": label})
    return pd.DataFrame(rows)


def _make_figures(output: Path, results: pd.DataFrame, temporal: pd.DataFrame) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    for family, filename in (("OTC foreign Buy", "figure1_otc_foreign_buy.png"), ("Listed foreign Net", "figure2_listed_foreign_net.png")):
        frame = results.loc[(results["hypothesis_family"] == family) & results["horizon"].isin((5, 10, 20))]
        if frame.empty:
            continue
        pivot = frame.pivot(index="horizon", columns="outcome_family", values="regression_beta")
        pivot.rename(columns=OUTCOME_LABELS).plot(kind="bar", figsize=(8, 4))
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylabel("Signal regression beta")
        plt.title(family)
        plt.tight_layout()
        plt.savefig(figure_dir / filename, dpi=160)
        plt.close()
    frame = temporal.loc[temporal["diagnostic_type"].eq("subperiod")]
    if not frame.empty:
        pivot = frame.pivot_table(index=["hypothesis_family", "diagnostic_label"], columns="horizon", values="mean_relative_return")
        pivot.plot(kind="bar", figsize=(10, 5))
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylabel("Mean OTC − 0050 return")
        plt.title("Relative rotation by subperiod")
        plt.tight_layout()
        plt.savefig(figure_dir / "figure3_rotation_by_subperiod.png", dpi=160)
        plt.close()


def _decision(group: pd.DataFrame, temporal: pd.DataFrame) -> str:
    primary = group.loc[group["horizon"].isin((5, 10, 20))]
    has_evidence = primary["fdr_global"].lt(0.05).any()
    time = temporal.loc[temporal["candidate_id"].isin(primary["candidate_id"])]
    stable = not time.empty and not time["sign_flip"].fillna(True).any()
    if has_evidence and stable:
        return "保留"
    if has_evidence:
        return "修改後再測"
    return "淘汰"


def _relative_sensitivity_consistency(
    primary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    hypothesis_family: str,
) -> bool:
    """Compare only like-for-like relative-rotation outcomes.

    The key is candidate plus horizon.  Absolute OTC price-index rows are not
    part of this comparison, and duplicate relative keys are a data error.
    """
    keys = ["candidate_id", "horizon"]
    primary_relative = primary.loc[
        primary["hypothesis_family"].eq(hypothesis_family)
        & primary["outcome_family"].eq("ROT_OTC_MINUS_0050"),
        keys + ["regression_beta"],
    ].copy()
    sensitivity_relative = sensitivity.loc[
        sensitivity["hypothesis_family"].eq(hypothesis_family)
        & sensitivity["outcome_family"].eq("ROT_OTC_PRICE_MINUS_0050"),
        keys + ["regression_beta"],
    ].copy()

    for label, frame in (
        ("primary total-return relative", primary_relative),
        ("price-index relative sensitivity", sensitivity_relative),
    ):
        duplicates = frame.loc[frame.duplicated(keys, keep=False), keys]
        if not duplicates.empty:
            raise ValueError(
                f"Phase 3 summary {label} comparison is not unique:\n"
                f"{duplicates.to_string(index=False)}"
            )

    p = primary_relative.set_index(keys)["regression_beta"]
    s = sensitivity_relative.set_index(keys)["regression_beta"]
    shared = p.index.intersection(s.index)
    if shared.empty:
        return False
    return bool((np.sign(p.loc[shared]) == np.sign(s.loc[shared])).all())


def _write_summary(path: Path, results: pd.DataFrame, sensitivity: pd.DataFrame, temporal: pd.DataFrame) -> None:
    relative = results.loc[results["outcome_family"].eq("ROT_OTC_MINUS_0050")]
    classifications = _classifications(results.loc[results["horizon"].isin((5, 10, 20))])
    lines = [
        "# Phase 3 — Market Rotation Mechanism Study", "",
        "> C0→Ck anchors at the signal-day close. The signal is only complete after C0, so these are mechanism outcomes, not tradable returns.", "",
        "## A–D. Rotation questions", "",
    ]
    for family in ("OTC foreign Buy", "Listed foreign Net", "Extreme foreign Sell", "Dealer secondary"):
        rows = classifications.loc[classifications["candidate_id"].isin(results.loc[results["hypothesis_family"].eq(family), "candidate_id"])]
        if rows.empty:
            continue
        detail = ", ".join(f"C{r.horizon}={r.rotation_classification}" for r in rows.itertuples())
        lines.append(f"- **{family}**：{detail}")
    lines += ["", "## E. Total-return vs price-index sensitivity", ""]
    for family in ("OTC foreign Buy", "Listed foreign Net"):
        consistent = _relative_sensitivity_consistency(results, sensitivity, family)
        lines.append(f"- {family}：方向{'一致' if consistent else '不完全一致'}。Primary inference 仍以 total-return index 為準。")
    lines += ["", "## F. Signal-family decisions", ""]
    decisions = []
    for family, group in relative.groupby("hypothesis_family"):
        decision = _decision(group, temporal)
        decisions.append(decision)
        lines.append(f"- {family}：**{decision}**")
    overall = "保留" if "保留" in decisions else "修改後再測" if "修改後再測" in decisions else "淘汰"
    lines += ["", f"Phase 3 overall：**{overall}**", "", "## Bias audit and opposing views", "",
        "- 已稽核 look-ahead、signal timing、C0 non-tradability、共同交易日 alignment、假日與缺值、重疊 outcome HAC、樣本量與期間集中。",
        "- Relative return 可能只是 sector composition、不同 beta、科技／半導體 regime 或 small-cap risk-on factor。",
        "- OTC flow 與 OTC index constituents 並非外生；本研究不能推論因果。",
        "- C20 高度重疊；HAC 改善標準誤，但不增加真正獨立事件數。",
        "- Candidates 來自 Phase 1/2 已看過的資料，仍有 selection bias 與 data snooping 風險。",
        "- OTC total-return index 與 0050 adjusted-price methodology 未必完全一致；price index 僅作 sensitivity。",
        "- Survivorship、少數年份主導與 index constituent changes 仍需在解讀時保留。",
        "", "## Possible next step (not executed)", "",
        "若 rotation evidence 成立，Phase 3.5 可預先定義 market-beta、size、sector 與 semiconductor concentration adjustment。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _write_run_info(path: Path, output: Path, tests_passed: bool | None) -> None:
    metadata = json.loads((output / "phase3_run_metadata.json").read_text(encoding="utf-8"))
    lines = [
        "Repository: hh4832/-institutional-spot-flow-study", f"Branch: {metadata.get('git_branch')}",
        f"Git commit: {metadata.get('git_commit_hash')}", f"Timestamp: {metadata.get('created_at_utc')}",
        "Timezone: Asia/Taipei", f"Python version: {platform.python_version()}", f"FinLab version: {_package_version('finlab')}",
        "Study mode: phase3_rotation", "0050 price source: etl:adj_close",
        f"OTC primary index: {OTC_TOTAL_RETURN_SERIES}", f"OTC sensitivity index: {OTC_PRICE_INDEX_SERIES}",
        "Outcome horizons: 1,5,10,20", "Primary horizons: 5,10,20",
        "Primary outcome: OTC - 0050 relative return", "Tradable outcome: false", "Rotation study: true",
        f"Drive output folder: {output.name}", f"Tests passed: {str(tests_passed).lower() if tests_passed is not None else 'not_recorded'}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase3_study(raw: RawData, config: StudyConfig | None = None, tests_passed: bool | None = None) -> Path:
    config = config or StudyConfig(study_mode="phase3_rotation")
    if tuple(config.accumulation_windows) != (1, 5, 10):
        raise ValueError("Phase 3 must preserve accumulation windows 1/5/10")
    if tuple(config.normalization_windows) != (252, 504, 756):
        raise ValueError("Phase 3 must preserve normalization windows 252/504/756")
    horizons = tuple(config.phase3_outcome_horizons)
    if horizons != (1, 5, 10, 20) or tuple(config.phase3_primary_horizons) != (5, 10, 20):
        raise ValueError("Phase 3 horizons must be 1/5/10/20 with 5/10/20 primary")
    if raw.price_dataset_names.get("close") != "etl:adj_close":
        raise RuntimeError("Phase 3 requires etl:adj_close for 0050")
    if raw.otc_index_dataset_name != OTC_INDEX_DATASET:
        raise RuntimeError(f"Phase 3 requires {OTC_INDEX_DATASET}")

    audit = _index_audit(raw)
    if audit["duplicate_dates"].ne(0).any() or ~audit["monotonic_increasing"].all():
        raise RuntimeError("OTC index dates must be monotonic with no duplicates")
    turnover = build_turnover(raw.market_amount)
    levels = build_phase2_activity_predictors(raw.institutional_buy, raw.institutional_sell, raw.institutional_net, turnover, config.accumulation_windows)
    normalized = add_normalizations(levels, windows=config.normalization_windows, include_global=False)
    candidate_table, candidates = _candidate_signals(normalized)
    outcomes = build_phase3_outcomes(raw.adjusted_close, raw.otc_total_return_index, raw.otc_price_index, levels.index, horizons)
    primary = _regression_results(candidates, outcomes, horizons, ("0050", "OTC_TR", "ROT_OTC_MINUS_0050"), config.phase2_min_group_n)
    sensitivity = _regression_results(candidates, outcomes, horizons, ("OTC_PRICE", "ROT_OTC_PRICE_MINUS_0050"), config.phase2_min_group_n)
    relative = primary.loc[primary["outcome_family"].eq("ROT_OTC_MINUS_0050")].copy()
    absolute = primary.loc[primary["outcome_family"].isin(("0050", "OTC_TR"))].copy()
    temporal = _temporal_robustness(candidates, outcomes)
    comparison = primary.merge(_classifications(primary.loc[primary["horizon"].isin((5, 10, 20))]), on=["candidate_id", "horizon"], how="left")
    significant = primary.loc[primary["fdr_global"].lt(0.05)].copy()

    output = timestamped_commit_output_directory(config.output_root)
    audit.to_csv(output / "phase3_otc_index_audit.csv", index=False)
    candidate_table.to_csv(output / "phase3_candidate_signals.csv", index=False)
    pd.concat([levels, normalized, outcomes], axis=1).to_parquet(output / "phase3_outcome_dataset.parquet")
    absolute.to_csv(output / "phase3_absolute_returns.csv", index=False)
    relative.to_csv(output / "phase3_relative_rotation_results.csv", index=False)
    primary.to_csv(output / "phase3_primary_regressions.csv", index=False)
    significant.to_csv(output / "phase3_significant_results.csv", index=False)
    primary.to_csv(output / "phase3_total_return_results.csv", index=False)
    sensitivity.to_csv(output / "phase3_price_index_sensitivity.csv", index=False)
    temporal.to_csv(output / "phase3_temporal_robustness.csv", index=False)
    comparison.to_csv(output / "phase3_signal_comparison.csv", index=False)
    _write_summary(output / "phase3_summary.md", primary, sensitivity, temporal)
    _make_figures(output, primary, temporal)
    metadata = {
        "repository": "hh4832/-institutional-spot-flow-study", "study_mode": "phase3_rotation",
        "rotation_study_included": True, "otc_index_dataset": OTC_INDEX_DATASET,
        "otc_primary_series": OTC_TOTAL_RETURN_SERIES, "otc_sensitivity_series": OTC_PRICE_INDEX_SERIES,
        "0050_price_source": "etl:adj_close", "outcome_entry_definition": "C0 mechanism anchor",
        "outcome_horizons": list(horizons), "primary_horizons": list(config.phase3_primary_horizons),
        "primary_outcome": "OTC - 0050 relative return", "tradable_outcome": False,
        "hac_lag_rule": "horizon - 1", "candidate_count": len(candidate_table),
        "predictor_definitions_changed": False, "prior_return_model_included": False,
        "etf_proxy_used": False, "synthetic_otc_open_used": False,
    }
    write_json(output / "phase3_run_metadata.json", build_manifest(config.to_dict(), metadata))
    write_json(output / "phase3_config_snapshot.json", config.to_dict())
    _write_run_info(output / "run_info.txt", output, tests_passed)
    return output
