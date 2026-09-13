from __future__ import annotations

from pathlib import Path

from config import StudyConfig
from phase25_pipeline import run_phase25_study
from synthetic_data import make_synthetic_raw_data


if __name__ == "__main__":
    raw = make_synthetic_raw_data(periods=1000)
    # Synthetic open/close are already adjusted series.  Use the production
    # source labels so the same strict price-source guard is exercised.
    raw.price_dataset_names = {"open": "etl:adj_open", "close": "etl:adj_close"}
    config = StudyConfig(
        study_mode="phase25_prior_return_mechanism",
        accumulation_windows=(1, 5, 10),
        phase25_return_horizons=(1, 2, 3, 5, 10, 20),
        output_root=Path("outputs_synthetic_phase25"),
    )
    output = run_phase25_study(raw, config)
    print(f"Phase 2.5 合成資料研究完成：{output.resolve()}")
