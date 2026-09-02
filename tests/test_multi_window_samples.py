import pandas as pd

from config import StudyConfig
from pipeline import _parse_predictor_name, _sample_availability


def test_default_study_config_only_adds_normalization_windows():
    config = StudyConfig()
    assert config.normalization_windows == (252, 504, 756)
    assert config.accumulation_windows == (1, 5, 10)
    assert config.return_horizons == (1, 2, 3, 5, 10)


def test_predictor_parser_handles_windowed_normalization_suffix():
    result = _parse_predictor_name(
        "combined__foreign__net__5d__rolling_504d_z"
    )
    assert result["predictor"] == "combined__foreign__net__5d"
    assert result["accumulation_window"] == 5


def test_common_sample_starts_when_longest_window_is_available():
    index = pd.bdate_range("2020-01-01", periods=12)
    normalized = pd.DataFrame(
        {
            "x__rolling_3d_z": [None, None] + list(range(10)),
            "x__rolling_6d_z": [None] * 5 + list(range(7)),
            "x__rolling_9d_z": [None] * 8 + list(range(4)),
            # A longer accumulation predictor completes its warm-up later,
            # even though it uses the same normalization window.
            "y__rolling_9d_z": [None] * 10 + list(range(2)),
        },
        index=index,
    )
    availability, common_start = _sample_availability(
        normalized, (3, 6, 9)
    )
    assert common_start == index[10]
    longest_full = availability.loc[
        availability["normalization_window"].eq(9)
        & availability["sample_type"].eq("full_available")
    ].iloc[0]
    assert longest_full["sample_start"] == index[8]
    assert longest_full["all_predictors_ready_start"] == index[10]
    common = availability.loc[
        availability["sample_type"].eq("common_window_start")
    ].iloc[0]
    assert common["available_date_count"] == 2
