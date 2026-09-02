from __future__ import annotations

import re

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


def normalization_metadata(column_name: str) -> tuple[str, str, bool, int | None]:
    rolling = re.search(r"__rolling_(\d+)d_(pr|z)$", column_name)
    if rolling:
        return f"rolling_{rolling.group(2)}", rolling.group(2), False, int(
            rolling.group(1)
        )
    if column_name.endswith("__global_pr"):
        return "global_pr", "pr", True, None
    if column_name.endswith("__global_z"):
        return "global_z", "z", True, None
    return "raw", "none", False, None
