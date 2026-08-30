from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class StudyConfig:
    ticker: str = "0050"
    accumulation_windows: tuple[int, ...] = (1, 5, 10)
    return_horizons: tuple[int, ...] = (1, 2, 3, 5, 10)
    rolling_window: int = 252
    rolling_min_periods: int = 252
    primary_horizons: tuple[int, ...] = (1, 5)
    primary_flow_type: str = "net"
    primary_institutions: tuple[str, ...] = (
        "foreign",
        "investment_trust",
        "dealer",
    )
    include_global_normalization: bool = True
    include_gross_activity: bool = False
    generate_all_figures: bool = False
    winsor_fraction: float = 0.01
    output_root: Path = Path("outputs")
    random_seed: int = 42

    def to_dict(self) -> dict:
        result = asdict(self)
        result["output_root"] = str(self.output_root)
        return result


PR_BINS = [-float("inf"), 5, 20, 40, 60, 80, 95, float("inf")]
PR_LABELS = [
    "PR_00_05",
    "PR_05_20",
    "PR_20_40",
    "PR_40_60",
    "PR_60_80",
    "PR_80_95",
    "PR_95_100",
]

Z_BINS = [-float("inf"), -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, float("inf")]
Z_LABELS = [
    "Z_LT_M2_5",
    "Z_M2_5_M1_5",
    "Z_M1_5_M0_5",
    "Z_M0_5_P0_5",
    "Z_P0_5_P1_5",
    "Z_P1_5_P2_5",
    "Z_GE_P2_5",
]


INSTITUTION_COLUMNS = {
    "foreign": {
        "listed": ["上市外資及陸資(不含外資自營商)"],
        "otc": ["上櫃外資及陸資(不含自營商)"],
    },
    "investment_trust": {
        "listed": ["上市投信"],
        "otc": ["上櫃投信"],
    },
    "dealer": {
        "listed": ["上市自營商(自行買賣)", "上市自營商(避險)"],
        "otc": ["上櫃自營商(自行買賣)", "上櫃自營商(避險)"],
    },
}


PRICE_DATASET_CANDIDATES = {
    "open": ("etl:adj_open",),
    "close": ("etl:adj_close",),
}
