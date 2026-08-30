from __future__ import annotations

import pandas as pd

from config import PR_BINS, PR_LABELS, Z_BINS, Z_LABELS


def assign_pr_group(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=PR_BINS,
        labels=PR_LABELS,
        right=True,
        include_lowest=True,
    )


def assign_z_group(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=Z_BINS,
        labels=Z_LABELS,
        right=False,
        include_lowest=True,
    )


def normalization_metadata(column_name: str) -> tuple[str, str, bool]:
    if column_name.endswith("__rolling_pr"):
        return "rolling_pr", "pr", False
    if column_name.endswith("__rolling_z"):
        return "rolling_z", "z", False
    if column_name.endswith("__global_pr"):
        return "global_pr", "pr", True
    if column_name.endswith("__global_z"):
        return "global_z", "z", True
    return "raw", "none", False
