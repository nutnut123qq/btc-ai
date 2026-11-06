"""STOPS-1 (roadmap 2.5): the -8% stop / -4% reduce replay, its backtester hook and its registry bookkeeping."""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

AI_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(AI_ROOT))

import trial_registry  # noqa: E402
import vn30_stops_eval as se  # noqa: E402
import vn_portfolio_backtest as pb  # noqa: E402


def _cal(n: int) -> list[str]:
    start = date(2019, 1, 1)
    return [(start + timedelta(days=i)).isoformat() for i in range(n)]


def _bars(cal: list[str], price: float, overrides: dict[int, tuple] | None = None) -> dict[str, tuple]:
    bars = {d: (price, price, price, price, 1e6) for d in cal}
    for i, (o, c) in (overrides or {}).items():
        bars[cal[i]] = (o, max(o, c), min(o, c), c, 1e6)
    return bars


def _rules(**kw) -> pb.PortfolioRules:
    base = {"rebalance_every": 100, "top_k": 1, "keep_rank": 1, "initial_cash_vnd": 100_000_000.0,
            "min_history_sessions": 0}
    base.update(kw)
    return pb.PortfolioRules(**base)


def _always_a(day, view):
    return {"A": 1.0}


def _run(cal, bars, rule, **kw):
    return pb.run_backtest(cal, bars, [("2000-01-01", frozenset(bars))], _always_a, _rules(**kw),
                           first_decision=0, risk_exit=rule)


def _rule(bars, reduce=False):
    return se.StopReduceRule({s: se.indicators_by_date(b) for s, b in bars.items()}, reduce=reduce)


# --------------------------------------------------------------------------------------------- indicators


def test_round_up_to_tick_matches_backend_grid():
    assert se.round_up_to_tick(20_050 * 0.92) == 18_450  # 18,446 -> 50 grid
    assert se.round_up_to_tick(9_200.4) == 9_210
    assert se.round_up_to_tick(46_000.0) == 46_000  # already on the grid
    assert se.round_up_to_tick(60_123 * 0.96) == 57_800  # 57,718.08 -> 100 grid


def test_sma_and_wilder_rsi():
    assert se.sma([1, 2, 3, 4], 2) == [None, 1.5, 2.5, 3.5]
    up = se.wilder_rsi(list(range(1, 20)), 14)
    assert up[:14] == [None] * 14 and up[14] == 100.0
    flat = se.wilder_rsi([5.0] * 20, 14)
    assert flat[14] == 50.0
    # hand check: seed over 14 deltas (7 gains of 2, 7 losses of 1) -> RS = 2, RSI = 66.67; then one loss of 3
    closes = [100.0]
    for k in range(14):
        closes.append(closes[-1] + (2.0 if k % 2 == 0 else -1.0))
    closes.append(closes[-1] - 3.0)
    rsi = se.wilder_rsi(closes, 14)
    assert rsi[14] == pytest.approx(100 - 100 / 3)
    ag, al = (1.0 * 13 + 0) / 14, (0.5 * 13 + 3) / 14
    assert rsi[15] == pytest.approx(100 - 100 / (1 + ag / al))


def test_bearish_needs_close_below_sma50_and_rsi_below_50():
    assert se.DayIndicators(9.0, 10.0, 40.0).bearish
    assert not se.DayIndicators(11.0, 10.0, 40.0).bearish
    assert not se.DayIndicators(9.0, 10.0, 55.0).bearish
    assert not se.DayIndicators(9.0, None, 40.0).bearish


def test_indicators_use_only_past_bars():
    cal = _cal(80)
    full = _bars(cal, 10_000, {i: (10_000 + 10 * i, 10_000 + 10 * i) for i in range(80)})
    cut = {d: b for d, b in full.items() if d <= cal[59]}
    a, b = se.indicators_by_date(full), se.indicators_by_date(cut)
    assert all(a[d] == b[d] for d in cut)


# --------------------------------------------------------------------------------------------- hook + rule


def test_stop_fires_on_close_at_level_and_sells_next_open():
    cal = _cal(8)
    # buy at s1: open 20,000 + 1 tick (50) = 20,050 -> stop level ceil50(18,446) = 18,450; close s3 = 18,450
    bars = {"A": _bars(cal, 20_000, {3: (19_000, 18_450), 4: (18_450, 18_400)})}
    rule = _rule(bars)
    res = _run(cal, bars, rule)
    buy, sell = res.trades
    assert (buy["date"], buy["price"], buy["origin"]) == (cal[1], 20_050, "rebalance")
    assert rule.events[0]["kind"] == "stop" and rule.events[0]["level"] == 18_450 and rule.events[0]["date"] == cal[3]
    assert (sell["date"], sell["side"], sell["shares"], sell["origin"]) == (cal[4], "sell", buy["shares"], "risk-exit")
    assert sell["price"] == 18_400  # open 18,450 - 1 tick
    assert res.daily[-1]["positions"] == 0
    # proceeds stay in cash until the next rebalance (none here): no re-buy
    assert len(res.trades) == 2
    control = _run(cal, bars, None)
    assert len(control.trades) == 1 and control.daily[-1]["positions"] == 1


def test_stop_not_hit_one_tick_above_level():
    cal = _cal(8)
    bars = {"A": _bars(cal, 20_000, {3: (19_000, 18_500)})}
    rule = _rule(bars)
    res = _run(cal, bars, rule)
    assert rule.events == [] and len(res.trades) == 1


def test_stop_waits_for_settled_shares():
    cal = _cal(8)
    bars = {"A": _bars(cal, 20_000, {1: (20_000, 18_000), 2: (18_000, 18_000), 3: (18_000, 18_000)})}
    res = _run(cal, bars, _rule(bars))
    sells = [t for t in res.trades if t["side"] == "sell"]
    assert sells[0]["date"] == cal[3]  # bought s1, sellable from the open of s3 (T+2)
    assert any(r["reason"] == "shares-not-settled" and r["date"] == cal[2] for r in res.rejections)


def test_exits_returned_at_a_decision_close_are_ignored():
    cal = _cal(8)
    # decisions at s0, s2, s4, s6: the stop condition holds at the close of s2 (a decision session) only
    bars = {"A": _bars(cal, 20_000, {2: (20_000, 18_000), 3: (19_000, 20_000)})}
    res = _run(cal, bars, _rule(bars), rebalance_every=2)
    assert not any(t.get("origin") == "risk-exit" for t in res.trades)


def test_risk_exit_replaces_an_open_buy_and_skips_an_open_sell():
    cal = _cal(6)
    bars = {"A": _bars(cal, 20_000)}
    calls = []

    def hook(ctx):
        calls.append(ctx)
        return {"A": 10**9} if ctx.index == 1 else {}

    # the s1 buy fills, then at the close of s1 the hook sells everything (more than held is capped)
    res = _run(cal, bars, hook)
    sells = [t for t in res.trades if t["side"] == "sell"]
    assert sells and sells[0]["shares"] == res.trades[0]["shares"] and sells[0]["origin"] == "risk-exit"
    assert calls[1].rebalance_entry and not calls[0].rebalance_entry and calls[0].decision_session
    assert [t["side"] for t in calls[1].fills] == ["buy"]
    # the s2 decision (price halved) tops A up at s3, but the s3 open is limit-locked up: the open buy is
    # replaced by the risk exit returned at the close of s3
    bars2 = {"A": _bars(cal, 20_000, {2: (20_000, 10_000), 3: (10_700, 10_700)})}
    res2 = _run(cal, bars2, lambda ctx: {"A": 100} if ctx.index == 3 else {}, rebalance_every=2)
    assert any(r["reason"] == "limit-locked" and r["side"] == "buy" and r["date"] == cal[3] for r in res2.rejections)
    assert any(r["reason"] == "superseded-by-risk-exit" and r["side"] == "buy" for r in res2.rejections)
    exits = [(t["date"], t["side"], t["shares"]) for t in res2.trades if t["origin"] == "risk-exit"]
    assert exits == [(cal[4], "sell", 100)]


def _ctx(day, shares, fills=(), open_orders=None, rebalance_entry=False, decision_session=False):
    return pb.RiskContext(day, 0, shares, tuple(fills), open_orders or {}, rebalance_entry, decision_session)


def _fill(side, shares, price, symbol="A"):
    return {"symbol": symbol, "side": side, "shares": shares, "price": price}


def test_rule_average_cost_reduce_once_and_stop_precedence():
    ind = {"A": {"d1": se.DayIndicators(10_000, 10_500, 60.0),
                 "d2": se.DayIndicators(9_600, 10_500, 40.0),   # bearish, at reduce level ceil(9,600)
                 "d3": se.DayIndicators(9_500, 10_500, 40.0),
                 "d4": se.DayIndicators(9_500, 10_500, 40.0),
                 "d5": se.DayIndicators(9_200, 10_500, 40.0)}}
    rule = se.StopReduceRule(ind, reduce=True)
    assert rule(_ctx("d1", {"A": 1000}, [_fill("buy", 1000, 10_000)])) == {}
    assert rule(_ctx("d2", {"A": 1000})) == {"A": 500}
    assert rule(_ctx("d2", {"A": 500}, [_fill("sell", 500, 9_590)])) == {}  # once between rebalances
    assert rule.avg_cost["A"] == 10_000  # a sell leaves the average cost unchanged
    # a rebalance entry re-arms the reduce; a top-up re-averages the cost
    assert rule(_ctx("d4", {"A": 1000}, [_fill("buy", 500, 9_500)], rebalance_entry=True)) == {}
    assert rule.avg_cost["A"] == pytest.approx(9_750)  # reduce level 9,750 x 0.96 = 9,360 < 9,500
    # stop level 9,750 x 0.92 = 8,970: 9,200 is not a stop; the reduce (9,360) fires
    assert rule(_ctx("d5", {"A": 1000})) == {"A": 500}
    assert [e["kind"] for e in rule.events] == ["reduce", "reduce"]
    stop = se.StopReduceRule({"A": {"d1": se.DayIndicators(9_000, 10_500, 40.0)}}, reduce=True)
    stop.avg_cost, stop.held = {"A": 10_000.0}, {"A": 300}
    assert stop(_ctx("d1", {"A": 300})) == {"A": 300} and stop.events[0]["kind"] == "stop"


def test_rule_reduce_needs_bearish_and_skips_open_sells_and_decisions():
    ind = {"A": {"d1": se.DayIndicators(9_000, 8_000, 40.0)}}  # below the reduce level but close > SMA50
    rule = se.StopReduceRule(ind, reduce=True)
    rule.avg_cost, rule.held = {"A": 9_500.0}, {"A": 100}
    assert rule(_ctx("d1", {"A": 100})) == {}
    ind2 = {"A": {"d1": se.DayIndicators(8_000, 9_000, 40.0)}}
    for kwargs in ({"open_orders": {"A": "sell"}}, {"decision_session": True}):
        r = se.StopReduceRule(ind2, reduce=True)
        r.avg_cost, r.held = {"A": 10_000.0}, {"A": 100}
        assert r(_ctx("d1", {"A": 100}, **kwargs)) == {}
    small = se.StopReduceRule({"A": {"d1": se.DayIndicators(9_600, 9_900, 40.0)}}, reduce=True)
    small.avg_cost, small.held = {"A": 10_000.0}, {"A": 100}
    assert small(_ctx("d1", {"A": 100})) == {"A": 100}  # half of one lot is 0 lots: sell the lot


def test_rule_detects_an_out_of_sync_book():
    rule = se.StopReduceRule({})
    with pytest.raises(RuntimeError):
        rule(_ctx("d1", {"A": 100}))


# --------------------------------------------------------------------------------------------- scores, metrics


def test_mom6_score_uses_126_bars_back():
    cal = _cal(130)
    a = {d: (100.0 + i, 0, 0, 100.0 + i, 1) for i, d in enumerate(cal)}
    b = {d: (50.0, 0, 0, 50.0, 1) for d in cal[:100]}
    view = pb.PointInTimeView(cal, 129, {"A": a, "B": b}, {"A": sorted(a), "B": sorted(b)}, frozenset({"A", "B"}))
    got = se.mom6_score(cal[129], view)
    assert set(got) == {"A"} and got["A"] == pytest.approx(229.0 / 103.0 - 1.0)
    assert se.ew_score(cal[129], view) == {"A": 0.0, "B": 0.0}


def test_nav_metrics_and_decision_rule():
    daily = [{"date": f"2020-01-0{k + 1}", "nav": v} for k, v in enumerate([100.0, 110.0, 99.0, 121.0])]
    m = se.nav_metrics(daily, {"years": 1.0, "annualTurnoverTwoSided": 2.0, "trades": 3, "costVnd": 1.0})
    assert m["netCagr"] == pytest.approx(0.21) and m["maxDrawdown"] == pytest.approx(-0.1)
    rets = np.array([0.1, -0.1, 121 / 99 - 1])
    assert m["sharpeAnnualized"] == pytest.approx(rets.mean() / rets.std(ddof=1) * np.sqrt(252))
    ctrl = {"netCagr": 0.1, "sharpeAnnualized": 0.5, "maxDrawdown": -0.4}
    better = {"netCagr": 0.12, "sharpeAnnualized": 0.6, "maxDrawdown": -0.3}
    assert se.decide(ctrl, better, 0.96)["verdict"] == "supported"
    assert se.decide(ctrl, better, 0.5)["verdict"] == "not supported"
    assert se.decide(ctrl, {**better, "maxDrawdown": -0.45}, 0.99)["verdict"] == "not supported"


def test_dsr_table_uses_given_trial_count():
    rng = np.random.default_rng(1)
    series = {k: {f"d{i:03d}": float(x) for i, x in enumerate(rng.normal(0.001, 0.01, 200))} for k in "abc"}
    out = se.dsr_table(series, 34)
    assert {v["nTrials"] for v in out["series"].values()} == {34}
    assert all(0.0 <= v["dsr"] <= 1.0 for v in out["series"].values())


# --------------------------------------------------------------------------------------------- registry


def _registry(tmp_path, spec_sha):
    reg = tmp_path / "registry.jsonl"
    trial_registry.append({"id": se.REGISTRY_ID, "registeredAt": "2026-11-06", "market": "HOSE",
                           "family": "hose-risk-rules", "status": "planned", "trials": 4, "trialsExact": True,
                           "description": "x", "source": "docs/research/protocols/stops-1.json",
                           "specSha256": spec_sha, "evaluatedAt": None, "result": None}, reg)
    return reg


def test_registered_entry_requires_the_exact_spec(tmp_path):
    spec = tmp_path / "stops-1.json"
    spec.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        se.registered_entry(tmp_path / "none.jsonl", spec)
    reg = _registry(tmp_path, se.sha256_file(spec))
    assert se.registered_entry(reg, spec)["id"] == se.REGISTRY_ID
    spec.write_text("{ }", encoding="utf-8")
    with pytest.raises(ValueError):
        se.registered_entry(reg, spec)


def test_repo_registry_has_the_frozen_spec_registered():
    entry = se.registered_entry()
    assert entry["trials"] == 4 and entry["family"] == "hose-risk-rules" and entry["market"] == "HOSE"
    spec = json.loads(se.SPEC_PATH.read_text(encoding="utf-8"))
    assert (spec["window"]["from"], spec["window"]["to"]) == (se.WINDOW_FROM, se.WINDOW_TO)
    assert spec["window"]["to"] < json.loads(se.PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "validation"]["holdout"]["from"]
    assert spec["data"]["snapshotSha256"] == se.SNAPSHOT_SHA256
    assert spec["sourceRules"]["stopLossFraction"] == se.STOP_LOSS_FRACTION
    assert spec["sourceRules"]["reduceLossFraction"] == se.REDUCE_LOSS_FRACTION


def test_record_run_supersedes_the_planned_entry_once(tmp_path, monkeypatch):
    spec_sha = se.sha256_file(se.SPEC_PATH)
    reg = _registry(tmp_path, spec_sha)
    run = {"runs": {"P-ew/none": {"netCagr": 0.1, "sharpeAnnualized": 0.5, "maxDrawdown": -0.3}},
           "dsrExcessVsControl": {"series": {"P-ew/stop-8 - none": {"dsr": 0.2}}}, "registryCount": {"trials": 34},
           "conclusion": "not supported", "specSha256": spec_sha}
    path = tmp_path / "result.json"
    path.write_text(json.dumps(run), encoding="utf-8")
    entry = se.record_run(path, registry_path=reg, evaluated_at="2026-11-06")
    assert entry["status"] == "run" and entry["supersedes"] == se.REGISTRY_ID and "not supported" in entry["result"]
    loaded = trial_registry.load(reg)
    assert [e["id"] for e in loaded.active()] == [f"{se.REGISTRY_ID}-run"]
    assert loaded.count("HOSE", ("run", "planned"))["trials"] == 4
    with pytest.raises(ValueError):
        se.record_run(path, registry_path=reg)
