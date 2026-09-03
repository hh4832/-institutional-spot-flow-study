from __future__ import annotations

from pathlib import Path

from config import StudyConfig
from phase2_pipeline import run_phase2_study
from synthetic_data import make_synthetic_raw_data


if __name__ == "__main__":
    raw = make_synthetic_raw_data(periods=520)
    config = StudyConfig(
        study_mode="phase2_flow_mechanism",
        accumulation_windows=(1, 5, 10),
        phase2_normalization_windows=(60, 120),
        phase2_return_horizons=(1, 5, 10),
        phase2_min_group_n=10,
        phase2_include_252_sensitivity=False,
        output_root=Path("outputs_synthetic_phase2"),
    )
    output = run_phase2_study(raw, config)
    print(f"Phase 2 合成資料研究完成：{output.resolve()}")
