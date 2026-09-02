from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import PR_LABELS, Z_LABELS, StudyConfig
from data_loader import RawData
from features import (
    add_normalizations,
    build_flow_predictors,
    build_turnover,
)
from grouping import assign_pr_group, assign_z_group, normalization_metadata
from reporting import (
    build_manifest,
    timestamped_output_directory,
    write_json,
    write_research_summary,
)
from returns import build_forward_returns
from study_statistics import (
    add_fdr_columns,
    continuous_hac_regression,
    summarize_group,
)
from validation import quality_report_markdown, validate_raw_data


def _parse_predictor_name(column: str) -> dict:
    base = re.sub(
        r"__(?:rolling_\d+d_(?:pr|z)|global_(?:pr|z)|raw)$", "", column
    )
    scope, institution, flow_type, accumulation = base.split("__")
    return {
        "predictor": base,
        "market_scope": scope,
        "institution": institution,
        "flow_type": flow_type,
        "accumulation_window": int(accumulation.removesuffix("d")),
    }


def _return_metadata(column: str) -> dict:
    if column == "return_c0_o1":
        return {
            "horizon": 1,
            "return_type": "C0_to_O1",
            "entry_time": "d0_close",
            "exit_time": "d1_open",
            "tradability": "statistical_only_not_tradable_from_d0_signal",
            "hac_lag": 0,
        }
    horizon = int(column.removeprefix("return_o1_c"))
    return {
        "horizon": horizon,
        "return_type": f"O1_to_C{horizon}",
        "entry_time": "d1_open",
        "exit_time": f"d{horizon}_close",
        "tradability": "tradable_after_d0_signal",
        "hac_lag": max(horizon - 1, 0),
    }


def _analysis_level(meta: dict, normalization: str, return_meta: dict) -> str:
    if (
        meta["market_scope"] == "combined"
        and meta["institution"] in {"foreign", "investment_trust", "dealer"}
        and meta["flow_type"] == "net"
        and normalization in {"rolling_pr", "rolling_z"}
        and return_meta["return_type"] in {"O1_to_C1", "O1_to_C5"}
    ):
        return "primary"
    if (
        meta["market_scope"] == "combined"
        and not normalization.startswith("global")
    ):
        return "secondary"
    return "exploratory"


def _group_results(
    normalized: pd.DataFrame,
    returns: pd.DataFrame,
    sample_type: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    predictor_columns = [
        column
        for column in normalized.columns
        if not column.endswith("__raw")
    ]
    return_columns = ["return_c0_o1"] + [
        column for column in returns.columns if column.startswith("return_o1_c")
    ]
    total_steps = len(predictor_columns) * len(return_columns)
    with tqdm(total=total_steps, desc="[7/8] 執行分組統計", leave=False) as progress:
        for predictor_column in predictor_columns:
            normalization, group_method, lookahead, normalization_window = normalization_metadata(
                predictor_column
            )
            values = normalized[predictor_column]
            if group_method == "pr":
                groups = assign_pr_group(values)
                labels = PR_LABELS
            elif group_method == "z":
                groups = assign_z_group(values)
                labels = Z_LABELS
            else:
                progress.update(len(return_columns))
                continue
            predictor_meta = _parse_predictor_name(predictor_column)
            for return_column in return_columns:
                return_meta = _return_metadata(return_column)
                for label in labels:
                    summary = summarize_group(
                        returns[return_column],
                        groups.eq(label),
                        maxlags=return_meta["hac_lag"],
                    )
                    if not summary:
                        continue
                    rows.append(
                        {
                            "analysis_type": "group",
                            **predictor_meta,
                            "normalization_type": normalization,
                            "normalization_window": normalization_window,
                            "sample_type": sample_type,
                            "group_method": group_method,
                            "group": label,
                            "lookahead_descriptive_only": lookahead,
                            "return_column": return_column,
                            **return_meta,
                            "analysis_level": _analysis_level(
                                predictor_meta, normalization, return_meta
                            ),
                            **summary,
                        }
                    )
                progress.update(1)
    result = pd.DataFrame(rows)
    return add_fdr_columns(result) if not result.empty else result


def _continuous_results(
    normalized: pd.DataFrame,
    returns: pd.DataFrame,
    sample_type: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    predictor_columns = list(normalized.columns)
    return_columns = ["return_c0_o1"] + [
        column for column in returns.columns if column.startswith("return_o1_c")
    ]
    for predictor_column in tqdm(
        predictor_columns, desc="[7/8] 執行連續變數迴歸", leave=False
    ):
        normalization, _, lookahead, normalization_window = normalization_metadata(
            predictor_column
        )
        predictor_meta = _parse_predictor_name(predictor_column)
        for return_column in return_columns:
            return_meta = _return_metadata(return_column)
            summary = continuous_hac_regression(
                normalized[predictor_column],
                returns[return_column],
                maxlags=return_meta["hac_lag"],
            )
            if summary:
                rows.append(
                    {
                        **predictor_meta,
                        "normalization_type": normalization,
                        "normalization_window": normalization_window,
                        "sample_type": sample_type,
                        "lookahead_descriptive_only": lookahead,
                        "return_column": return_column,
                        **return_meta,
                        **summary,
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        valid = result["raw_p_value"].notna()
        result["fdr_p_value_global"] = np.nan
        if valid.any():
            from statsmodels.stats.multitest import multipletests

            result.loc[valid, "fdr_p_value_global"] = multipletests(
                result.loc[valid, "raw_p_value"], method="fdr_bh"
            )[1]
    return result


def _annual_breakdown(
    normalized: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    primary = [
        column
        for column in normalized.columns
        if "__combined__" not in column
    ]
    # Correct filter using the actual prefix layout.
    primary = [
        column
        for column in normalized.columns
        if column.startswith("combined__")
        and "__net__" in column
        and re.search(r"__rolling_\d+d_(?:pr|z)$", column)
    ]
    rows: list[dict] = []
    for predictor_column in primary:
        normalization, method, _, normalization_window = normalization_metadata(
            predictor_column
        )
        groups = (
            assign_pr_group(normalized[predictor_column])
            if method == "pr"
            else assign_z_group(normalized[predictor_column])
        )
        for return_column in ("return_o1_c1", "return_o1_c5"):
            if return_column not in returns.columns:
                continue
            frame = pd.concat(
                [groups.rename("group"), returns[return_column].rename("return")],
                axis=1,
            ).dropna()
            frame["year"] = frame.index.year
            summary = frame.groupby(["year", "group"], observed=True)["return"].agg(
                observation_count="size",
                mean_return="mean",
                positive_rate=lambda x: x.gt(0).mean(),
            )
            if summary.empty:
                continue
            summary = summary.reset_index()
            meta = _parse_predictor_name(predictor_column)
            summary["predictor"] = meta["predictor"]
            summary["normalization_type"] = normalization
            summary["normalization_window"] = normalization_window
            summary["sample_type"] = "full_available"
            summary["return_column"] = return_column
            rows.extend(summary.to_dict("records"))
    return pd.DataFrame(rows)


def _subperiod_breakdown(
    normalized: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    periods = {
        "2009_2014": (2009, 2014),
        "2015_2019": (2015, 2019),
        "2020_latest": (2020, 9999),
    }
    rows: list[dict] = []
    columns = [
        column
        for column in normalized.columns
        if column.startswith("combined__")
        and "__net__" in column
        and re.search(r"__rolling_\d+d_(?:pr|z)$", column)
    ]
    for predictor_column in columns:
        normalization, method, _, normalization_window = normalization_metadata(
            predictor_column
        )
        groups = (
            assign_pr_group(normalized[predictor_column])
            if method == "pr"
            else assign_z_group(normalized[predictor_column])
        )
        for return_column in ("return_o1_c1", "return_o1_c5"):
            if return_column not in returns:
                continue
            base = pd.concat(
                [groups.rename("group"), returns[return_column].rename("return")],
                axis=1,
            ).dropna()
            for period, (start, end) in periods.items():
                frame = base.loc[(base.index.year >= start) & (base.index.year <= end)]
                summary = frame.groupby("group", observed=True)["return"].agg(
                    observation_count="size",
                    mean_return="mean",
                    positive_rate=lambda x: x.gt(0).mean(),
                )
                if summary.empty:
                    continue
                summary = summary.reset_index()
                meta = _parse_predictor_name(predictor_column)
                summary["period"] = period
                summary["predictor"] = meta["predictor"]
                summary["normalization_type"] = normalization
                summary["normalization_window"] = normalization_window
                summary["sample_type"] = "full_available"
                summary["return_column"] = return_column
                rows.extend(summary.to_dict("records"))
    return pd.DataFrame(rows)


def _generate_primary_figures(
    normalized: pd.DataFrame,
    group_results: pd.DataFrame,
    figure_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    primary = group_results.loc[group_results["analysis_level"].eq("primary")]
    for institution in ("foreign", "investment_trust", "dealer"):
        institution_subset = primary.loc[
            primary["institution"].eq(institution)
            & primary["accumulation_window"].eq(1)
            & primary["normalization_type"].eq("rolling_pr")
            & primary["return_type"].eq("O1_to_C1")
            & primary["sample_type"].eq("common_window_start")
        ].copy()
        for window, subset in institution_subset.groupby(
            "normalization_window", dropna=True
        ):
            subset = subset.sort_values("group")
            error_low = subset["mean_return"] - subset["zero_ci_low"]
            error_high = subset["zero_ci_high"] - subset["mean_return"]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.errorbar(
                subset["group"].astype(str),
                subset["mean_return"] * 100,
                yerr=np.vstack([error_low, error_high]) * 100,
                fmt="o-",
                capsize=3,
            )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(
                f"{institution} Net 1d rolling {int(window)}d PR: O1 to C1"
            )
            ax.set_ylabel("Mean return (%) with HAC 95% CI")
            ax.tick_params(axis="x", rotation=35)
            fig.tight_layout()
            fig.savefig(
                figure_dir
                / f"combined_{institution}_net_1d_rolling_{int(window)}d_pr_O1_C1.png",
                dpi=150,
            )
            plt.close(fig)

    correlation_columns = [
        column
        for column in normalized.columns
        if column.startswith("combined__")
        and "__net__" in column
        and column.endswith("__raw")
    ]
    if correlation_columns:
        correlation = normalized[correlation_columns].corr(method="spearman")
        fig, ax = plt.subplots(figsize=(12, 10))
        image = ax.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
        labels = [column.removesuffix("__raw") for column in correlation.columns]
        ax.set_xticks(range(len(labels)), labels=labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(labels)), labels=labels, fontsize=6)
        ax.set_title("Combined-market Net predictor Spearman correlation")
        fig.colorbar(image, ax=ax, fraction=0.03)
        fig.tight_layout()
        fig.savefig(figure_dir / "combined_net_predictor_spearman_correlation.png", dpi=150)
        plt.close(fig)


def _normalization_columns_for_window(
    normalized: pd.DataFrame, window: int
) -> list[str]:
    pattern = re.compile(rf"__rolling_{window}d_(?:pr|z)$")
    return [column for column in normalized.columns if pattern.search(column)]


def _sample_availability(
    normalized: pd.DataFrame, windows: tuple[int, ...]
) -> tuple[pd.DataFrame, pd.Timestamp]:
    rows: list[dict] = []
    all_ready_starts: list[pd.Timestamp] = []
    for window in windows:
        columns = _normalization_columns_for_window(normalized, window)
        first_valid_dates = [normalized[column].first_valid_index() for column in columns]
        if not columns or any(date is None for date in first_valid_dates):
            start = pd.NaT
            end = pd.NaT
            count = 0
        else:
            start = min(first_valid_dates)
            all_predictors_ready_start = max(first_valid_dates)
            end = normalized.index.max()
            count = int((normalized.index >= start).sum())
            all_ready_starts.append(all_predictors_ready_start)
        rows.append(
            {
                "sample_type": "full_available",
                "normalization_window": window,
                "sample_start": start,
                "all_predictors_ready_start": (
                    all_predictors_ready_start if count else pd.NaT
                ),
                "sample_end": end,
                "available_date_count": count,
            }
        )
    if len(all_ready_starts) != len(windows):
        raise ValueError("至少一個標準化視窗沒有任何可用樣本")
    # The common comparison begins only after every 1/5/10-day predictor has
    # completed the longest window's warm-up.
    common_start = max(all_ready_starts)
    common_count = int((normalized.index >= common_start).sum())
    rows.append(
        {
            "sample_type": "common_window_start",
            "normalization_window": max(windows),
            "sample_start": common_start,
            "all_predictors_ready_start": common_start,
            "sample_end": normalized.index.max(),
            "available_date_count": common_count,
        }
    )
    return pd.DataFrame(rows), common_start


def run_study(raw: RawData, config: StudyConfig | None = None) -> Path:
    config = config or StudyConfig()
    print("[2/8] 驗證原始資料")
    quality = validate_raw_data(
        raw.institutional_buy,
        raw.institutional_sell,
        raw.institutional_net,
        raw.market_amount,
    )

    print("[3/8] 重建法人買賣金額")
    turnover = build_turnover(raw.market_amount)
    print("[4/8] 建立 Predictors")
    predictors = build_flow_predictors(
        raw.institutional_buy,
        raw.institutional_sell,
        raw.institutional_net,
        turnover,
        windows=config.accumulation_windows,
        include_gross_activity=config.include_gross_activity,
    )

    print("[5/8] 建立 0050 Forward Returns")
    returns = build_forward_returns(
        raw.adjusted_open,
        raw.adjusted_close,
        predictors.index,
        horizons=config.return_horizons,
    )

    print("[6/8] 計算 PR 與 Z-score")
    normalized = add_normalizations(
        predictors,
        windows=config.normalization_windows,
        include_global=config.include_global_normalization,
    )
    analysis_dataset = pd.concat([predictors, normalized, returns], axis=1)
    sample_availability, common_start = _sample_availability(
        normalized, config.normalization_windows
    )
    common_normalized = normalized.loc[normalized.index >= common_start]
    common_returns = returns.loc[returns.index >= common_start]

    group_results = pd.concat(
        [
            _group_results(normalized, returns, "full_available"),
            _group_results(
                common_normalized, common_returns, "common_window_start"
            ),
        ],
        ignore_index=True,
    )
    continuous_results = pd.concat(
        [
            _continuous_results(normalized, returns, "full_available"),
            _continuous_results(
                common_normalized, common_returns, "common_window_start"
            ),
        ],
        ignore_index=True,
    )
    annual = _annual_breakdown(normalized, returns)
    subperiod = _subperiod_breakdown(normalized, returns)

    print("[8/8] 輸出檔案與摘要")
    output = timestamped_output_directory(config.output_root)
    analysis_dataset.to_parquet(output / "analysis_dataset.parquet")
    predictors.to_parquet(output / "predictor_dataset.parquet")
    for window in config.normalization_windows:
        window_columns = _normalization_columns_for_window(normalized, window)
        window_data = pd.concat(
            [predictors, normalized[window_columns], returns], axis=1
        )
        window_start = sample_availability.loc[
            sample_availability["sample_type"].eq("full_available")
            & sample_availability["normalization_window"].eq(window),
            "sample_start",
        ].iloc[0]
        window_data.loc[window_data.index >= window_start].to_parquet(
            output / f"analysis_dataset_rolling_{window}d_full.parquet"
        )
    analysis_dataset.loc[analysis_dataset.index >= common_start].to_parquet(
        output / "analysis_dataset_common_window_start.parquet"
    )
    sample_availability.to_csv(
        output / "sample_availability.csv", index=False
    )
    group_results.to_csv(output / "group_statistics.csv", index=False)
    continuous_results.to_csv(output / "continuous_regression.csv", index=False)
    annual.to_csv(output / "annual_breakdown.csv", index=False)
    subperiod.to_csv(output / "subperiod_breakdown.csv", index=False)
    quality.to_csv(output / "data_quality_report.csv", index=False)
    (output / "data_quality_report.md").write_text(
        quality_report_markdown(quality), encoding="utf-8"
    )

    with pd.ExcelWriter(output / "group_statistics.xlsx", engine="openpyxl") as writer:
        pd.DataFrame(
            {
                "說明": [
                    "rolling 252／504／756 日並列分析；global PR/Z 僅供描述性對照。",
                    "full_available 使用各視窗全部可用日期；common_window_start 使用共同起點。",
                    "C0→O1 不可由 d0 盤後訊號交易；O1→Ch 才是可交易報酬。",
                ]
            }
        ).to_excel(writer, sheet_name="README", index=False)
        for level, sheet in {
            "primary": "Main_Results",
            "secondary": "Secondary_Results",
            "exploratory": "Exploratory_Results",
        }.items():
            group_results.loc[group_results.get("analysis_level") == level].to_excel(
                writer, sheet_name=sheet, index=False
            )
        group_results.to_excel(writer, sheet_name="Group_Statistics", index=False)
        continuous_results.to_excel(
            writer, sheet_name="Continuous_Regression", index=False
        )
        annual.to_excel(writer, sheet_name="Annual_Breakdown", index=False)
        subperiod.to_excel(writer, sheet_name="Subperiod_Breakdown", index=False)
        quality.to_excel(writer, sheet_name="Data_Quality", index=False)
        sample_availability.to_excel(
            writer, sheet_name="Sample_Availability", index=False
        )
        fdr_columns = [
            column
            for column in group_results.columns
            if "p_value" in column or "significant" in column
        ]
        group_results[fdr_columns].to_excel(
            writer, sheet_name="FDR_Summary", index=False
        )

    continuous_results.to_excel(
        output / "continuous_regression.xlsx", index=False
    )
    failures = int(
        quality.get("difference_over_tolerance_count", pd.Series(dtype=float))
        .fillna(0)
        .sum()
    )
    write_research_summary(
        output / "research_summary.md", group_results, failures
    )
    _generate_primary_figures(normalized, group_results, output / "figures")
    metadata = {
        "data_start": str(analysis_dataset.index.min()),
        "data_end": str(analysis_dataset.index.max()),
        "price_dataset_names": raw.price_dataset_names,
        "hac_lag_rule": "max(horizon - 1, 0)",
        "global_normalization_warning": "lookahead_descriptive_only",
        "common_sample_start": str(common_start),
        "sample_types": ["full_available", "common_window_start"],
    }
    write_json(output / "run_manifest.json", build_manifest(config.to_dict(), metadata))
    write_json(output / "config_snapshot.json", config.to_dict())
    return output
