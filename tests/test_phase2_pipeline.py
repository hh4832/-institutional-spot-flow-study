from pathlib import Path

import pandas as pd

from config import StudyConfig
from phase2_pipeline import run_phase2_study
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
