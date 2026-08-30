import numpy as np

from features import build_turnover, reconstruct_institution_flows
from synthetic_data import make_synthetic_raw_data


def test_buy_minus_sell_equals_net():
    raw = make_synthetic_raw_data(periods=30)
    difference = raw.institutional_buy - raw.institutional_sell - raw.institutional_net
    assert np.nanmax(np.abs(difference.to_numpy())) == 0


def test_reconstructed_totals_equal_official():
    raw = make_synthetic_raw_data(periods=30)
    rebuilt = reconstruct_institution_flows(raw.institutional_net)
    assert np.allclose(
        rebuilt[("listed", "total_institutional")],
        raw.institutional_net["上市合計"],
    )
    assert np.allclose(
        rebuilt[("otc", "total_institutional")],
        raw.institutional_net["上櫃三大法人合計*"],
    )
    assert np.allclose(
        rebuilt[("combined", "total_institutional")],
        rebuilt[("listed", "total_institutional")]
        + rebuilt[("otc", "total_institutional")],
    )


def test_turnover_scope_mapping():
    raw = make_synthetic_raw_data(periods=5)
    turnover = build_turnover(raw.market_amount)
    assert np.allclose(turnover["listed"], raw.market_amount["TAIEX"])
    assert np.allclose(turnover["otc"], raw.market_amount["OTC"])
    assert np.allclose(
        turnover["combined"], raw.market_amount["TAIEX"] + raw.market_amount["OTC"]
    )
