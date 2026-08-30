from __future__ import annotations

import numpy as np
import pandas as pd

from features import reconstruct_institution_flows


def _frame_profile(name: str, frame: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        rows.append(
            {
                "check": "column_profile",
                "dataset": name,
                "column": str(column),
                "start_date": series.first_valid_index(),
                "end_date": series.last_valid_index(),
                "observation_count": int(series.notna().sum()),
                "missing_count": int(series.isna().sum()),
                "zero_count": int(series.eq(0).sum()),
                "infinite_count": int(np.isinf(series).sum()),
                "duplicate_date_count": int(frame.index.duplicated().sum()),
            }
        )
    return rows


def validate_raw_data(
    buy: pd.DataFrame,
    sell: pd.DataFrame,
    official_net: pd.DataFrame,
    market_amount: pd.DataFrame,
    tolerance: float = 1.0,
) -> pd.DataFrame:
    common_index = buy.index.intersection(sell.index).intersection(official_net.index)
    common_columns = buy.columns.intersection(sell.columns).intersection(
        official_net.columns
    )
    difference = (
        buy.loc[common_index, common_columns]
        - sell.loc[common_index, common_columns]
        - official_net.loc[common_index, common_columns]
    )

    rows: list[dict] = []
    for column in common_columns:
        series = difference[column]
        rows.append(
            {
                "check": "buy_minus_sell_equals_net",
                "dataset": "institutional",
                "column": column,
                "max_absolute_difference": float(series.abs().max(skipna=True)),
                "difference_over_tolerance_count": int(
                    series.abs().gt(tolerance).sum()
                ),
                "missing_count": int(series.isna().sum()),
            }
        )

    for name, frame in {
        "institutional_buy": buy,
        "institutional_sell": sell,
        "institutional_net": official_net,
        "market_amount": market_amount,
    }.items():
        rows.extend(_frame_profile(name, frame))

    for flow_name, frame in {
        "buy": buy,
        "sell": sell,
        "net": official_net,
    }.items():
        rebuilt = reconstruct_institution_flows(frame)
        listed_diff = rebuilt[("listed", "total_institutional")] - frame["上市合計"]
        otc_diff = (
            rebuilt[("otc", "total_institutional")]
            - frame["上櫃三大法人合計*"]
        )
        for scope, series in {"listed": listed_diff, "otc": otc_diff}.items():
            rows.append(
                {
                    "check": "rebuilt_total_equals_official",
                    "dataset": flow_name,
                    "column": scope,
                    "max_absolute_difference": float(series.abs().max(skipna=True)),
                    "difference_over_tolerance_count": int(
                        series.abs().gt(tolerance).sum()
                    ),
                    "missing_count": int(series.isna().sum()),
                }
            )
    return pd.DataFrame(rows)


def quality_report_markdown(report: pd.DataFrame) -> str:
    formula_checks = report.loc[
        report["check"].isin(
            ["buy_minus_sell_equals_net", "rebuilt_total_equals_official"]
        )
    ]
    failures = int(
        formula_checks.get("difference_over_tolerance_count", pd.Series(dtype=float))
        .fillna(0)
        .sum()
    )
    lines = [
        "# 資料品質報告",
        "",
        f"- 公式驗證失敗筆數：{failures}",
        f"- 品質檢查列數：{len(report)}",
        "- NaN 不會自動填成 0。",
        "- 全域 PR/Z 僅標記為描述性分析。",
        "",
        "## 公式驗證",
        "",
        formula_checks.to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)
