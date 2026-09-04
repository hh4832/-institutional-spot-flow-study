from pathlib import Path
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from config import StudyConfig
from phase2_pipeline import run_phase2_study
from reporting import timestamped_output_directory
from synthetic_data import make_synthetic_raw_data


def test_phase2_pipeline_writes_scoped_outputs_without_legacy_grid(tmp_path: Path):
    raw = make_synthetic_raw_data(periods=180)
    config = StudyConfig(
        study_mode="phase2_flow_mechanism",
        accumulation_windows=(1, 5, 10),
        phase2_normalization_windows=(20, 40),
        phase2_return_horizons=(1,),
        phase2_min_group_n=5,
        phase2_include_252_sensitivity=False,
        output_root=tmp_path,
    )
    output = run_phase2_study(raw, config)
    required = {
        "phase2_feature_dictionary.csv",
        "phase2_confirmatory_results.csv",
        "phase2_sample_audit.csv",
        "phase2_nonlinear_results.csv",
        "phase2_gross_activity_results.csv",
        "phase2_flow_change_results.csv",
        "phase2_turning_point_results.csv",
        "phase2_data_regime_audit.csv",
        "phase2_run_metadata.json",
        "phase2_summary.md",
    }
    assert required.issubset({p.name for p in output.iterdir()})
    metadata = pd.read_json(output / "phase2_run_metadata.json", typ="series")
    assert metadata["legacy_full_grid_executed"] is False
    assert not (output / "group_statistics.csv").exists()


def test_phase2_group_membership_excludes_unavailable_rolling_values(tmp_path: Path):
    raw = make_synthetic_raw_data(periods=180)
    config = StudyConfig(
        study_mode="phase2_flow_mechanism",
        accumulation_windows=(1, 5, 10),
        phase2_normalization_windows=(20, 40),
        phase2_return_horizons=(10,),
        phase2_min_group_n=5,
        phase2_include_252_sensitivity=False,
        output_root=tmp_path,
    )
    output = run_phase2_study(raw, config)
    audit = pd.read_csv(output / "phase2_sample_audit.csv")
    confirmatory = pd.read_csv(output / "phase2_confirmatory_results.csv")

    assert audit["uncontrolled_count_identity_ok"].all()
    assert audit["controlled_count_identity_ok"].all()
    grouped = confirmatory.loc[confirmatory["model"].eq("group_dummy_uncontrolled")]
    expected = audit.set_index("candidate_id")["uncontrolled_valid_count"]
    actual = grouped.set_index("candidate_id")["observation_count"]
    pd.testing.assert_series_equal(
        actual.sort_index(), expected.loc[actual.index].sort_index(),
        check_names=False, check_dtype=False,
    )


def test_timestamped_output_directory_never_overwrites_existing_folder(tmp_path: Path):
    fixed_now = datetime(2026, 9, 4, 12, 34, 56)
    stamp = fixed_now.strftime("%Y%m%d_%H%M%S")
    existing = tmp_path / stamp
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("existing result", encoding="utf-8")

    with patch("reporting.datetime") as clock:
        clock.now.return_value = fixed_now
        created = timestamped_output_directory(tmp_path)

    assert created != existing
    assert created.name.startswith(f"{stamp}_")
    assert marker.read_text(encoding="utf-8") == "existing result"
