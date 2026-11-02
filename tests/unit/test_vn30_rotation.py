"""VN30-1: the rotation backtest is point-in-time, charges fees/tax on traded value, sells members that
left the basket, and computes its statistics correctly."""
import json
import math
import sys
from pathlib import Path

import pytest

AI_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(AI_ROOT))

import vn30_rotation as vr  # noqa: E402
from equity_base_rates import Costs  # noqa: E402

FREE = Costs(0.0, 0.0, 0.0)
BASE = Costs(0.0015, 0.0015, 0.001)


def _calendar(n: int) -> list[str]:
    # n consecutive "sessions" over several months (dates only need to sort and carry a month).
    out, y, m, d = [], 2020, 1, 1
    while len(out) < n:
        out.append(f"{y:04d}-{m:02d}-{d:02d}")
        d += 1
        if d > 20:
            d, m = 1, m + 1
            if m > 12:
                m, y = 1, y + 1
    return out


def _trend(calendar: list[str], start: float, step: float) -> dict[str, tuple[float, float]]:
    return {day: (start + step * i, start + step * i) for i, day in enumerate(calendar)}


def test_membership_needs_30_unique_tickers_in_chronological_periods(tmp_path):
    good = [f"T{i:02d}" for i in range(30)]
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"periods": [{"effectiveDate": "2018-02-05", "constituents": good},
                                             {"effectiveDate": "2018-08-06", "constituents": good}]}))
    assert len(vr.load_membership(path)) == 2
    for bad in ([good[:29]], [good[:29] + [good[0]]]):
        path.write_text(json.dumps({"periods": [{"effectiveDate": "2018-02-05", "constituents": bad[0]}]}))
        with pytest.raises(ValueError):
            vr.load_membership(path)
    path.write_text(json.dumps({"periods": [{"effectiveDate": "2018-08-06", "constituents": good},
                                             {"effectiveDate": "2018-02-05", "constituents": good}]}))
    with pytest.raises(ValueError):
        vr.load_membership(path)


def test_members_on_uses_the_latest_period_already_in_effect():
    periods = [("2018-02-05", frozenset({"A"})), ("2018-08-06", frozenset({"B"}))]
    assert vr.members_on(periods, "2018-02-04") is None
    assert vr.members_on(periods, "2018-02-05") == {"A"}
    assert vr.members_on(periods, "2018-08-05") == {"A"}
    assert vr.members_on(periods, "2018-08-06") == {"B"}


def test_selection_reads_no_session_after_the_decision_day():
    cal = _calendar(120)
    prices = {"A": _trend(cal, 100, 0.5), "B": _trend(cal, 100, 0.2), "C": _trend(cal, 100, -0.1)}
    i = 90
    before = vr.select(cal, prices, frozenset(prices), i, 3, 2)
    assert before == ["A", "B"]
    # Make C explode and A collapse from the very next session on: a look-ahead would change the pick.
    for day in cal[i + 1:]:
        prices["C"][day] = (1_000.0, 1_000.0)
        prices["A"][day] = (1.0, 1.0)
    assert vr.select(cal, prices, frozenset(prices), i, 3, 2) == before


def test_first_buy_pays_the_buy_fee_on_traded_value():
    cal = _calendar(3)
    prices = {"A": {d: (10.0, 12.0) for d in cal}}  # open != close: the trade must use the open
    units, cash, turnover = vr.rebalance({}, 1.0, ["A"], 1.0, prices, cal[1], {}, BASE)
    assert units["A"] * 10.0 + cash == pytest.approx(1.0 / (1.0 + 0.0015), abs=1e-12)
    assert cash == pytest.approx(0.0, abs=1e-12)
    assert turnover == pytest.approx(1.0 / 1.0015, abs=1e-12)


def test_full_switch_pays_sell_fee_and_tax_then_buy_fee():
    cal = _calendar(3)
    prices = {"A": {d: (10.0, 10.0) for d in cal}, "B": {d: (20.0, 20.0) for d in cal}}
    units, cash, _ = vr.rebalance({"A": 0.1}, 0.0, ["B"], 1.0, prices, cal[1], {"A": 10.0}, BASE)
    assert "A" not in units
    assert units["B"] * 20.0 + cash == pytest.approx((1.0 - 0.0015 - 0.001) / (1.0 + 0.0015), abs=1e-12)


def test_unchanged_holding_pays_nothing():
    cal = _calendar(3)
    prices = {"A": {d: (10.0, 10.0) for d in cal}}
    units, cash, turnover = vr.rebalance({"A": 0.1}, 0.0, ["A"], 1.0, prices, cal[1], {"A": 10.0}, BASE)
    assert units["A"] == pytest.approx(0.1) and turnover == pytest.approx(0.0, abs=1e-12)


def test_unused_weight_stays_in_cash_when_fewer_than_top_k_qualify():
    cal = _calendar(3)
    prices = {"A": {d: (10.0, 10.0) for d in cal}, "B": {d: (10.0, 10.0) for d in cal}}
    units, cash, _ = vr.rebalance({}, 1.0, ["A", "B"], 1.0 / 5, prices, cal[1], {}, FREE)
    assert cash == pytest.approx(0.6) and units["A"] * 10.0 == pytest.approx(0.2)


def test_a_member_that_left_the_basket_is_sold_at_the_next_rebalance():
    cal = _calendar(160)  # 8 "months" of 20 sessions
    prices = {"A": _trend(cal, 100, 1.0), "B": _trend(cal, 100, 0.3), "C": _trend(cal, 100, 0.1)}
    switch = cal[110]
    periods = [(cal[0], frozenset({"A", "B", "C"})), (switch, frozenset({"B", "C"}))]
    sim = vr.simulate(cal, prices, periods, 3, 1, FREE, cal[0], cal[-1])
    # Until 3 months of history exist nothing qualifies, so the first rebalances hold cash.
    held_before = [h["held"] for h in sim["holdings"] if h["date"] <= switch and h["held"]]
    held_after = [h["held"] for h in sim["holdings"] if h["date"] > switch]
    assert sim["holdings"][0]["held"] == []
    assert held_before and all(h == ["A"] for h in held_before)  # A is strongest while it is a member
    assert held_after and all(h == ["B"] for h in held_after)


def test_metrics_cagr_and_max_drawdown():
    m = vr.metrics(["2020-01-01", "2020-06-01", "2021-01-01", "2022-01-01"], [1.0, 1.2, 0.9, 1.21])
    years = 731 / 365.25
    assert m["cagr"] == pytest.approx(1.21 ** (1 / years) - 1)
    assert m["maxDrawdown"] == pytest.approx(0.9 / 1.2 - 1)


def test_deflated_sharpe_needs_more_than_luck_across_trials():
    assert vr.expected_max_sr([0.05]) == 0.0
    assert vr.expected_max_sr([0.01, 0.02, 0.03]) < vr.expected_max_sr([0.0, 0.05, 0.10])
    sr, n = 0.05, 1000
    assert vr.psr(sr, vr.expected_max_sr([sr]), n, 0.0, 3.0) == vr.psr(sr, 0.0, n, 0.0, 3.0)
    assert vr.psr(sr, vr.expected_max_sr([0.0, 0.05, 0.10, 0.02]), n, 0.0, 3.0) < vr.psr(sr, 0.0, n, 0.0, 3.0)
    assert vr.psr(0.0, 0.0, n, 0.0, 3.0) == pytest.approx(0.5)


SPEC = {  # the fields run() reads, copied from evidence/vn30-1-20261102/spec.json (sha256 d9434e99...3566)
    "signal": {"lookbackMonths": [3, 6, 12]},
    "portfolio": {"topK": [5, 10]},
    "costs": {"base": {"buyFeeFraction": 0.0015, "sellFeeFraction": 0.0015, "sellTaxFraction": 0.001},
              "sensitivity": {"buyFeeFraction": 0.0025, "sellFeeFraction": 0.0025, "sellTaxFraction": 0.001}},
    "periods": {"design": {"from": "2018-02-01", "to": "2022-12-31"}, "outOfSample": {"from": "2023-01-01"}},
}


def test_run_end_to_end_reports_every_declared_config_and_a_verdict():
    spec = SPEC
    # Synthetic sessions: ~22 per month from 2017 to 2024, 8 trending tickers + the index.
    days = []
    y, m, d = 2017, 1, 1
    while y < 2024 or (y == 2024 and m <= 6):
        days.append(f"{y:04d}-{m:02d}-{d:02d}")
        d += 1
        if d > 22:
            d, m = 1, m + 1
            if m > 12:
                m, y = 1, y + 1
    prices = {f"S{k}": {day: (100 + k * i * 0.01, 100 + k * i * 0.01) for i, day in enumerate(days)} for k in range(1, 9)}
    prices["VNINDEX"] = {day: (1000 + i * 0.1, 1000 + i * 0.1) for i, day in enumerate(days)}
    periods = [("2017-01-01", frozenset(f"S{k}" for k in range(1, 9)))]
    result = vr.run(spec, periods, prices)
    assert len(result["configs"]) == 6 and result["trials"] == 6
    assert result["verdict"] in ("worth-following", "not-enough-evidence")
    assert {"VNINDEX", "equalWeightVN30"} == set(result["benchmarks"])
    for c in result["configs"]:
        assert set(c) >= {"lookbackMonths", "topK", "base", "sensitivity"}
        assert c["base"]["full"]["cagr"] >= c["sensitivity"]["full"]["cagr"]  # higher fees never help
    # The verdict hangs on the DSR wiring: SR* from all six trials, and the decision rule on the selected config.
    sr_star = vr.expected_max_sr([c["base"]["full"]["dailySharpe"] for c in result["configs"]])
    assert result["expectedMaxDailySharpe"] == pytest.approx(sr_star) and sr_star > 0
    for c in result["configs"]:
        f = c["base"]["full"]
        assert f["deflatedSharpe"] == pytest.approx(vr.psr(f["dailySharpe"], sr_star, f["sessions"] - 1, f["skew"], f["kurtosis"]))
    best = max(result["configs"], key=lambda c: c["base"]["design"]["sharpe"])
    assert result["selected"] == {"lookbackMonths": best["lookbackMonths"], "topK": best["topK"]}
    oos = best["base"]["outOfSample"]["cagr"]
    worth = (oos > result["benchmarks"]["VNINDEX"]["outOfSample"]["cagr"]
             and oos > result["benchmarks"]["equalWeightVN30"]["outOfSample"]["cagr"]
             and best["base"]["full"]["deflatedSharpe"] >= 0.95)
    assert result["verdict"] == ("worth-following" if worth else "not-enough-evidence")
    json.dumps(result, allow_nan=False)


def test_zero_volume_bars_are_not_sessions(tmp_path):
    (tmp_path / "VHM.csv").write_text(
        "date,open,high,low,close,volume\n"
        "2018-01-02,9000,9000,9000,9000,0.0\n"          # pre-listing placeholder
        "2018-05-17,31000,31000,31000,31000,18000\n", encoding="utf-8")
    assert vr.load_prices(tmp_path) == {"VHM": {"2018-05-17": (31000.0, 31000.0)}}


def test_month_end_decisions_need_a_next_session():
    cal = ["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-28"]
    assert vr.month_end_indices(cal, "2020-01-01", "2020-12-31") == [1]
    assert math.isclose(len(vr.month_end_indices(cal, "2020-02-01", "2020-12-31")), 0)
