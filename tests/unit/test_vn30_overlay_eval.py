"""OVERLAY-1 (roadmap 3.2): exposure hook of the backtester, MA200 / vol-target overlays, composite score and
registry bookkeeping."""
import hashlib
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

AI_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(AI_ROOT))

import trial_registry  # noqa: E402
import vn30_2_run as v2  # noqa: E402
import vn30_overlay_eval as oe  # noqa: E402
import vn_portfolio_backtest as pb  # noqa: E402

SPEC_SHA = "e0a1036110eb880491a615c92cb1785aa6147b485c3eca59bc94360b3441cd02"


def _cal(n: int, start: date = date(2019, 1, 1)) -> list[str]:
    return [(start + timedelta(days=i)).isoformat() for i in range(n)]


def _flat(cal, price=20_000.0):
    return {d: (price, price, price, price, 1e6) for d in cal}


def _rules(**kw) -> pb.PortfolioRules:
    base = {"rebalance_every": 3, "top_k": 1, "keep_rank": 1, "initial_cash_vnd": 100_000_000.0,
            "min_history_sessions": 0}
    base.update(kw)
    return pb.PortfolioRules(**base)


def _run(cal, bars, exposure=None, **kw):
    return pb.run_backtest(cal, bars, [("2000-01-01", frozenset(bars))], lambda d, v: {"A": 1.0}, _rules(**kw),
                           first_decision=0, exposure=exposure)


# --------------------------------------------------------------------------------------------- backtester hook


def test_exposure_none_equals_exposure_one():
    cal = _cal(10)
    bars = {"A": _flat(cal)}
    a, b = _run(cal, bars), _run(cal, bars, lambda d, v, t: 1.0)
    assert a.trades == b.trades and [r["nav"] for r in a.daily] == [r["nav"] for r in b.daily]
    assert "exposure" not in a.rebalances[0] and b.rebalances[0]["exposure"] == 1.0


def test_exposure_scales_target_shares_and_zero_sells_everything():
    cal = _cal(8)
    bars = {"A": _flat(cal)}
    full = _run(cal, bars, rebalance_every=100)
    half = _run(cal, bars, lambda d, v, t: 0.5, rebalance_every=100)
    assert abs(half.trades[0]["shares"] - full.trades[0]["shares"] / 2) < 100
    # decisions at s0, s3, s6: invested at s1, the s3 decision says 0 -> everything sold at the s4 open
    seq = iter([1.0, 0.0, 1.0])
    res = _run(cal, bars, lambda d, v, t: next(seq))
    sides = [(t["date"], t["side"]) for t in res.trades]
    assert sides == [(cal[1], "buy"), (cal[4], "sell"), (cal[7], "buy")]
    assert res.daily[4]["positions"] == 0 and [r["exposure"] for r in res.rebalances] == [1.0, 0.0, 1.0]


def test_exposure_hook_sees_targets_and_rejects_out_of_range():
    cal = _cal(6)
    bars = {"A": _flat(cal)}
    seen = []
    _run(cal, bars, lambda d, v, t: seen.append((d, t, v.date)) or 1.0)
    assert seen[0] == (cal[0], ("A",), cal[0])
    for bad in (1.5, -0.1, math.nan):
        with pytest.raises(ValueError):
            _run(cal, bars, lambda d, v, t, bad=bad: bad)


# --------------------------------------------------------------------------------------------- overlays


def test_ma200_regime_uses_only_past_closes():
    cal = _cal(260)
    closes = {d: 100.0 + i for i, d in enumerate(cal)}  # rising: close above its MA
    for i in range(230, 260):
        closes[cal[i]] = 50.0  # crash below the MA
    m = oe.Ma200Cash(closes)
    assert m.state(cal[150])["exposure"] == 1.0 and m.state(cal[150])["ma"] is None  # < 200 closes
    assert m.state(cal[220])["exposure"] == 1.0
    s = m.state(cal[240])
    assert s["exposure"] == 0.0 and s["ma"] == pytest.approx(np.mean([closes[d] for d in cal[41:241]]))
    cut = oe.Ma200Cash({d: c for d, c in closes.items() if d <= cal[240]})
    assert cut.state(cal[240]) == s
    assert oe.Ma200Cash({d: 100.0 for d in cal}).state(cal[250])["exposure"] == 1.0  # close == MA -> invested
    assert m(cal[240], None, ("A",)) == 0.0 and m.log[-1]["date"] == cal[240]


def test_basket_returns_forward_fill_and_mean():
    cal = _cal(5)
    bars = {"A": {cal[i]: (0, 0, 0, c, 1) for i, c in enumerate([100, 110, 121, 121, 133.1])},
            "B": {cal[0]: (0, 0, 0, 50, 1), cal[2]: (0, 0, 0, 55, 1), cal[3]: (0, 0, 0, 44, 1)}}
    r = oe.basket_returns(cal, bars, ["A", "B"], cal[4], n=4)
    a = np.array([0.10, 0.10, 0.0, 0.10])
    b = np.array([0.0, 0.10, -0.20, 0.0])  # B missing on s1 and s4: forward-filled close -> 0 return
    assert r == pytest.approx((a + b) / 2)
    assert len(oe.basket_returns(cal, bars, ["A"], cal[3], n=4)) == 0  # not enough sessions


def test_vol_target_exposure():
    cal = _cal(70)
    closes, c = [], 10_000.0
    for i in range(70):
        c *= 1.02 if i % 2 else 0.98
        closes.append(c)
    bars = {"A": {d: (x, x, x, x, 1) for d, x in zip(cal, closes)}}
    vt = oe.VolTarget(cal, bars)
    r = oe.basket_returns(cal, bars, ["A"], cal[69])
    sigma = np.std(r, ddof=1) * math.sqrt(252)
    st = vt.state(cal[69], ["A"])
    assert st["sigma"] == pytest.approx(sigma) and st["exposure"] == pytest.approx(min(1.0, 0.15 / sigma))
    assert 0 < st["exposure"] < 1
    flat = oe.VolTarget(cal, {"A": _flat(cal)})
    assert flat.state(cal[69], ["A"])["exposure"] == 1.0  # sigma 0
    assert vt.state(cal[30], ["A"])["exposure"] == 1.0  # fewer than 60 returns


# --------------------------------------------------------------------------------------------- synthetic market


def _market(n_sym=22, n_days=330, seed=7):
    rng = np.random.default_rng(seed)
    cal = []
    d = date(2018, 1, 1)
    while len(cal) < n_days:
        if d.weekday() < 5:
            cal.append(d.isoformat())
        d += timedelta(days=1)
    bars = {}
    for k in range(n_sym):
        sym = f"S{k:02d}"
        px = 20_000.0
        rows = {}
        for day in cal:
            o = px
            px = max(1_000.0, px * math.exp(rng.normal(0.0003, 0.02)))
            rows[day] = (o, max(o, px), min(o, px), px, float(rng.integers(1e5, 1e6)))
        bars[sym] = rows
    index = {day: float(np.mean([bars[s][day][3] for s in bars])) for day in cal}
    return v2.Market(tuple(cal), bars, [("2000-01-01", frozenset(bars))], {"VN30": index})


def test_composite_score_equals_vn30_2_panel():
    market = _market()
    score = oe.CompositeScore(v2.symbol_series(market.bars))
    first = pb.first_decision_index(market.calendar, market.bars, market.periods)
    assert first is not None
    checked = oe.check_composite_against_panel(market, score, list(range(first, len(market.calendar) - 1, 20)))
    assert checked >= 22 * 3


def _registry_with_entry(tmp_path, sha=SPEC_SHA):
    reg = tmp_path / "registry.jsonl"
    trial_registry.append({"id": oe.REGISTRY_ID, "registeredAt": "2026-11-06", "market": "HOSE",
                           "family": "hose-risk-overlays", "status": "planned", "trials": 4, "trialsExact": True,
                           "description": "test", "source": "docs/research/protocols/overlay-1.json",
                           "specSha256": sha}, reg)
    return reg


def test_registered_entry_requires_matching_spec(tmp_path):
    with pytest.raises(ValueError, match="register the trial"):
        oe.registered_entry(tmp_path / "none.jsonl")
    (tmp_path / "other").mkdir()
    with pytest.raises(ValueError, match="another version"):
        oe.registered_entry(_registry_with_entry(tmp_path / "other", sha="0" * 64))
    assert oe.registered_entry(_registry_with_entry(tmp_path))["id"] == oe.REGISTRY_ID


def test_evaluate_end_to_end_on_synthetic_market_and_record(tmp_path):
    market = _market()
    reg = _registry_with_entry(tmp_path)
    result = oe.evaluate(market, "f" * 64, registry_path=reg)
    assert result["evaluatorSha256"] == oe.pe.evaluator_sha256()  # recorded next to the protocol sha256
    assert result["protocolSha256"] == oe.pe.protocol_sha256()
    assert set(result["runs"]) == {f"{p}/{v}" for p in oe.PORTFOLIOS for v in oe.VARIANTS}
    assert result["registryCount"]["trials"] == 4
    assert set(result["verdicts"]) == {f"{p}/{v}" for p in oe.PORTFOLIOS for v in oe.VARIANTS[1:]}
    assert len(result["dsrExcessVsEqualWeight"]["series"]) == 5
    assert result["runs"]["P-ew/none"]["meanExposure"] == 1.0
    for key in ("P-ew/vol-target-15", "P-composite/vol-target-15"):
        assert 0 < result["runs"][key]["meanExposure"] <= 1.0
        assert all(0 < row["exposure"] <= 1 for row in result["overlayLog"][key])
    assert result["compositeRowsVerifiedAgainstPanel"] > 0
    assert result["vsControl"]["P-ew/ma200-cash"]["netCagr"] == pytest.approx(
        result["runs"]["P-ew/ma200-cash"]["netCagr"] - result["runs"]["P-ew/none"]["netCagr"])
    path = tmp_path / "result.json"
    path.write_text(json.dumps(v2._clean(result)), encoding="utf-8")
    entry = oe.record_run(path, registry_path=reg, evidence="x/result.json")
    assert entry["status"] == "run" and entry["supersedes"] == oe.REGISTRY_ID and entry["trials"] == 4
    assert entry["resultsSha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        oe.record_run(path, registry_path=reg)


def test_evaluate_refuses_data_after_the_window(tmp_path):
    market = _market(n_days=40)
    late = v2.Market(market.calendar + ("2024-01-02",), market.bars, market.periods, market.benchmarks)
    with pytest.raises(RuntimeError, match="after the development window"):
        oe.evaluate(late, "f" * 64, registry_path=_registry_with_entry(tmp_path))


# --------------------------------------------------------------------------------------------- frozen spec


def test_spec_is_frozen_and_registered():
    assert hashlib.sha256(oe.SPEC_PATH.read_bytes()).hexdigest() == SPEC_SHA
    spec = oe.check_spec()
    assert spec["frozenBeforeResults"] is True
    entry = oe.registered_entry()
    assert entry["specSha256"] == SPEC_SHA and entry["trials"] == 4 and entry["market"] == "HOSE"
