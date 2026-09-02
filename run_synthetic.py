from __future__ import annotations

from pathlib import Path

from config import StudyConfig
from pipeline import run_study
from synthetic_data import make_synthetic_raw_data


if __name__ == "__main__":
    raw = make_synthetic_raw_data()
    config = StudyConfig(
        accumulation_windows=(1, 5, 10),
        return_horizons=(1, 2, 3, 5, 10),
        normalization_windows=(60, 120, 180),
        include_global_normalization=True,
        output_root=Path("outputs_synthetic"),
    )
    output = run_study(raw, config)
    print(f"合成資料研究完成：{output.resolve()}")
