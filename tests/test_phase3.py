import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import OTC_INDEX_DATASET, StudyConfig
from phase3_pipeline import (
    PHASE3_CANDIDATE_SPECS,
    _hac_lag,
    _relative_sensitivity_consistency,
    build_phase3_outcomes,
    run_phase3_study,
)
from synthetic_data import make_synthetic_raw_data


def test_phase3_c0_ck_alignment_and_relative_identity():
    index = pd.bdate_range("2026-01-01", periods=25)
    close = pd.Series(np.arange(100.0, 125.0), index=index)
    otc_total = pd.Series(np.arange(200.0, 225.0), index=index)
    otc_price = pd.Series(np.arange(300.0, 325.0), index=index)
    result = build_phase3_outcomes(close, otc_total, otc_price, index, (1, 5, 10, 20))
    expected_0050 = close.iloc[5] / close.iloc[0] - 1
    expected_otc = otc_total.iloc[5] / otc_total.iloc[0] - 1
    assert result.loc[index[0], "exit_date_C5"] == index[5]
    assert np.isclose(result.loc[index[0], "0050_C0_C5"], expected_0050)
    assert np.isclose(result.loc[index[0], "OTC_TR_C0_C5"], expected_otc)
    assert np.isclose(
        result.loc[index[0], "ROT_OTC_MINUS_0050_C0_C5"],
        expected_otc - expected_0050,
    )


def test_phase3_hac_lags():
    assert _hac_lag(1) == 0
    assert _hac_lag(5) == 4
    assert _hac_lag(10) == 9
    assert _hac_lag(20) == 19


def test_phase3_scope_is_fixed_and_has_no_forbidden_grid():
    config = StudyConfig()
    assert config.accumulation_windows == (1, 5, 10)
    assert config.normalization_windows == (252, 504, 756)
    assert config.phase3_outcome_horizons == (1, 5, 10, 20)
    assert config.phase3_primary_horizons == (5, 10, 20)
    assert all(spec[0].endswith(("__5d", "__10d")) for spec in PHASE3_CANDIDATE_SPECS)
    assert all("prior" not in spec[0].lower() for spec in PHASE3_CANDIDATE_SPECS)
    assert all("etf" not in spec[0].lower() for spec in PHASE3_CANDIDATE_SPECS)


def _summary_result_rows(outcome_family: str, betas: tuple[float, ...]) -> pd.DataFrame:
    horizons = (1, 5, 10, 20)
    return pd.DataFrame({
        "candidate_id": ["candidate_a"] * len(horizons),
        "hypothesis_family": ["OTC foreign Buy"] * len(horizons),
        "horizon": horizons,
        "outcome_family": [outcome_family] * len(horizons),
        "regression_beta": betas,
    })


def test_sensitivity_relative_outcome_is_unique_and_absolute_rows_are_excluded():
    primary = _summary_result_rows("ROT_OTC_MINUS_0050", (0.01, 0.02, 0.03, -0.01))
    sensitivity = pd.concat([
        _summary_result_rows("OTC_PRICE", (-0.5, -0.5, -0.5, 0.5)),
        _summary_result_rows("ROT_OTC_PRICE_MINUS_0050", (0.02, 0.01, 0.04, -0.02)),
    ], ignore_index=True)
    relative = sensitivity.loc[
        sensitivity["outcome_family"].eq("ROT_OTC_PRICE_MINUS_0050")
    ]
    assert not relative.duplicated(["candidate_id", "horizon"]).any()
    assert _relative_sensitivity_consistency(
        primary, sensitivity, "OTC foreign Buy"
    ) is True


def test_duplicate_relative_sensitivity_fails_loudly():
    primary = _summary_result_rows("ROT_OTC_MINUS_0050", (0.01, 0.02, 0.03, -0.01))
    relative = _summary_result_rows(
        "ROT_OTC_PRICE_MINUS_0050", (0.02, 0.01, 0.04, -0.02)
    )
    sensitivity = pd.concat([relative, relative.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="price-index relative sensitivity.*not unique"):
        _relative_sensitivity_consistency(primary, sensitivity, "OTC foreign Buy")


def test_phase3_integration_outputs_and_metadata(tmp_path: Path):
    raw = make_synthetic_raw_data(periods=1000)
    raw.price_dataset_names = {"open": "etl:adj_open", "close": "etl:adj_close"}
    raw.otc_index_dataset_name = OTC_INDEX_DATASET
    output = run_phase3_study(raw, StudyConfig(output_root=tmp_path))
    required = {
        "phase3_run_metadata.json", "phase3_config_snapshot.json", "phase3_otc_index_audit.csv",
        "phase3_candidate_signals.csv", "phase3_outcome_dataset.parquet", "phase3_absolute_returns.csv",
        "phase3_relative_rotation_results.csv", "phase3_primary_regressions.csv",
        "phase3_significant_results.csv", "phase3_total_return_results.csv",
        "phase3_price_index_sensitivity.csv", "phase3_temporal_robustness.csv",
        "phase3_signal_comparison.csv", "phase3_summary.md", "run_info.txt",
    }
    assert required.issubset({item.name for item in output.iterdir()})
    metadata = json.loads((output / "phase3_run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["tradable_outcome"] is False
    assert metadata["rotation_study_included"] is True
    assert metadata["prior_return_model_included"] is False
    assert metadata["etf_proxy_used"] is False
    assert metadata["synthetic_otc_open_used"] is False
    audit = pd.read_csv(output / "phase3_otc_index_audit.csv")
    assert audit["duplicate_dates"].eq(0).all()
