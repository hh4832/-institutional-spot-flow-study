from pathlib import Path

from config import StudyConfig
from phase3_pipeline import run_phase3_study
from synthetic_data import make_synthetic_raw_data


if __name__ == "__main__":
    raw = make_synthetic_raw_data(periods=1000)
    raw.price_dataset_names = {"open": "etl:adj_open", "close": "etl:adj_close"}
    output = run_phase3_study(
        raw,
        StudyConfig(study_mode="phase3_rotation", output_root=Path("outputs_synthetic_phase3")),
    )
    print(f"Phase 3 synthetic study complete: {output.resolve()}")
