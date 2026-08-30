import numpy as np
import pandas as pd

from config import PR_LABELS, Z_LABELS
from grouping import assign_pr_group, assign_z_group


def test_pr_boundaries_have_no_overlap_or_gap():
    values = pd.Series([0, 5, 5.0001, 20, 20.0001, 40, 60, 80, 95, 95.0001, 100])
    groups = assign_pr_group(values)
    assert groups.notna().all()
    assert set(groups.astype(str)).issubset(set(PR_LABELS))
    assert groups.iloc[1] == "PR_00_05"
    assert groups.iloc[2] == "PR_05_20"
    assert groups.iloc[-2] == "PR_95_100"


def test_z_boundaries_have_no_overlap_or_gap():
    values = pd.Series([-10, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 10])
    groups = assign_z_group(values)
    assert groups.notna().all()
    assert set(groups.astype(str)).issubset(set(Z_LABELS))
    assert groups.iloc[1] == "Z_M2_5_M1_5"
    assert groups.iloc[-2] == "Z_GE_P2_5"
