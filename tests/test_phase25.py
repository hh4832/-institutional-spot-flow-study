import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import StudyConfig
from phase25_pipeline import (
    PHASE25_REPRESENTATIVE_SPECS,
    _hac_lag,
    _validate_representative_specs,
    run_phase25_study,
)
from returns import build_forward_returns, build_prior_returns
from synthetic_data import make_synthetic_raw_data


def test_prior_returns_use_trading_rows_and_no_future_values():
    index = pd.bdate_range("2026-01-01", periods=30)
    close = pd.Series(np.arange(100.0, 130.0), index=index)
    result = build_prior_returns(close, index, windows=(5, 20))
    assert result["prior_ret_5d"].iloc[:5].isna().all()
    assert result["prior_ret_20d"].iloc[:20].isna().all()
    assert np.isclose(result.loc[index[5], "prior_ret_5d"], close.iloc[5] / close.iloc[0] - 1)
    assert np.isclose(result.loc[index[20], "prior_ret_20d"], close.iloc[20] / close.iloc[0] - 1)
    changed_future = close.copy()
    changed_future.iloc[21:] = 99999
    changed = build_prior_returns(changed_future, index, windows=(5, 20))
    pd.testing.assert_series_equal(result.loc[: index[20], "prior_ret_20d"], changed.loc[: index[20], "prior_ret_20d"])


def test_prior_returns_shift_on_price_rows_not_sparse_signal_rows():
    price_index = pd.bdate_range("2026-01-01", periods=30)
    signal_index = price_index.delete(10)
    close = pd.Series(np.arange(100.0, 130.0), index=price_index)
    result = build_prior_returns(close, signal_index, windows=(5,))
    signal_date = price_index[12]
    assert np.isclose(
        result.loc[signal_date, "prior_ret_5d"],
        close.loc[signal_date] / close.iloc[7] - 1,
    )


def test_c20_alignment_and_hac_lag():
    index = pd.bdate_range("2026-01-01", periods=25)
    opened = pd.Series(np.arange(100.0, 125.0), index=index)
    closed = opened + 1
    result = build_forward_returns(opened, closed, index, horizons=(20,))
    assert result.loc[index[0], "entry_date"] == index[1]
    assert result.loc[index[0], "exit_date_c20"] == index[20]
    assert np.isclose(result.loc[index[0], "return_o1_c20"], closed.iloc[20] / opened.iloc[1] - 1)
    assert _hac_lag(20) == 19


def test_phase25_scope_and_strict_adjusted_price(tmp_path: Path):
    _validate_representative_specs()
    assert {spec[2] for spec in PHASE25_REPRESENTATIVE_SPECS}.issubset({504, 756})
    config = StudyConfig()
    assert config.accumulation_windows == (1, 5, 10)
    assert config.phase25_return_horizons == (1, 2, 3, 5, 10, 20)
    raw = make_synthetic_raw_data(periods=800)
    with pytest.raises(RuntimeError, match="etl:adj_open"):
        run_phase25_study(raw, StudyConfig(output_root=tmp_path))


def test_phase25_integration_outputs_metadata_and_never_rotation(tmp_path: Path):
    raw = make_synthetic_raw_data(periods=1000)
    raw.price_dataset_names = {"open": "etl:adj_open", "close": "etl:adj_close"}
    output = run_phase25_study(raw, StudyConfig(output_root=tmp_path))
    required = {
        "phase25_summary.md",
        "phase25_run_metadata.json",
        "phase25_candidate_signals.csv",
        "phase25_forward_horizon_results.csv",
        "phase25_horizon_profile.csv",
        "phase25_prior5_controlled_regressions.csv",
        "phase25_prior20_controlled_regressions.csv",
        "phase25_prior5_interactions.csv",
        "phase25_prior20_interactions.csv",
        "phase25_effect_attenuation.csv",
        "phase25_prior_return_descriptive.csv",
        "phase25_temporal_robustness.csv",
        "phase25_significant_results.csv",
        "phase25_analysis_dataset.parquet",
        "phase25_config_snapshot.json",
        "run_info.txt",
    }
    assert required.issubset({path.name for path in output.iterdir()})
    metadata = json.loads((output / "phase25_run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["rotation_study_included"] is False
    assert metadata["price_source_open"] == "etl:adj_open"
    assert metadata["price_source_close"] == "etl:adj_close"
    assert metadata["outcome_price_adjusted"] is True
    assert metadata["c20_hac_lag"] == 19
    assert metadata["accumulation_windows"] == [1, 5, 10]
    assert metadata["prior_return_windows"] == [5, 20]
    assert output.name.endswith(metadata["git_commit_hash"][:8])
