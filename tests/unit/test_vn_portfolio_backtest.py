"""VN30-2 portfolio backtester (roadmap 2.1): hand-computed golden cases for fees/tax, slippage ticks, lot
rounding, T+2 blocking, limit-locked opens (proxy and raw), hysteresis and leavers, plus a look-ahead test."""
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

AI_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(AI_ROOT))

import vn_portfolio_backtest as pb  # noqa: E402


def _cal(n: int) -> list[str]:
    start = date(2024, 1, 1)
    return [(start + timedelta(days=i)).isoformat() for i in range(n)]


def _flat(cal: list[str], price: float, overrides: dict[int, tuple] | None = None) -> dict[str, tuple]:
    bars = {d: (price, price, price, price, 1e6) for d in cal}
    for i, (o, c) in (overrides or {}).items():
        bars[cal[i]] = (o, max(o, c), min(o, c), c, 1e6)
    return bars


def _periods(*members: str) -> list:
    return [("2000-01-01", frozenset(members))]


def _rules(**kw) -> pb.PortfolioRules:
    base = {"rebalance_every": 2, "top_k": 1, "keep_rank": 1, "initial_cash_vnd": 100_000_000.0,
            "min_history_sessions": 0}
    base.update(kw)
    return pb.PortfolioRules(**base)


def _by_date(scores: dict[str, dict[str, float]]):
    return lambda day, view: scores.get(day, {})


def test_hose_tick_and_lot_floor():
    assert [pb.hose_tick(p) for p in (9_990, 10_000, 49_950, 50_000, 120_000)] == [10, 50, 50, 100, 100]
    assert [pb.lot_floor(x) for x in (0, 99, 100, 9985.02, 12_345)] == [0, 0, 100, 9900, 12_300]
    with pytest.raises(ValueError):
        pb.hose_tick(0)


def test_golden_fee_tax_and_slippage_round_trip():
    cal = _cal(5)
    bars = {"A": _flat(cal, 9_990, {1: (9_990, 10_500), 2: (11_000, 11_000), 3: (11_000, 11_000),
                                     4: (11_000, 11_000)})}
    res = pb.run_backtest(cal, bars, _periods("A"), _by_date({cal[0]: {"A": 1.0}}), _rules(), first_decision=0)
    buy, sell = res.trades
    # buy at s1: open 9,990 + 1 tick (10) = 10,000; all-in 10,015 -> 1e8 / 10,015 = 9,985.02 -> 9,900 shares
    assert (buy["date"], buy["side"], buy["shares"], buy["price"]) == (cal[1], "buy", 9_900, 10_000)
    assert buy["grossVnd"] == pytest.approx(99_000_000)
    assert buy["feeVnd"] == pytest.approx(148_500)
    assert buy["taxVnd"] == 0 and buy["slippageVnd"] == pytest.approx(99_000)
    nav = res.nav_by_date()
    assert nav[cal[1]] == pytest.approx(851_500 + 9_900 * 10_500)
    # decision s2 scores nothing -> sell at s3 open 11,000 - 1 tick (50) = 10,950 (bought s1, sellable s3)
    assert (sell["date"], sell["side"], sell["shares"], sell["price"]) == (cal[3], "sell", 9_900, 10_950)
    assert sell["grossVnd"] == pytest.approx(108_405_000)
    assert sell["feeVnd"] == pytest.approx(162_607.5)
    assert sell["taxVnd"] == pytest.approx(108_405)
    assert nav[cal[3]] == pytest.approx(108_985_487.5) == nav[cal[4]]
    s = res.summary
    assert s["feeVnd"] == pytest.approx(311_107.5) and s["taxVnd"] == pytest.approx(108_405)
    assert s["slippageVnd"] == pytest.approx(594_000)
    assert s["costVnd"] == pytest.approx(311_107.5 + 108_405 + 594_000)
    assert s["tradedVnd"] == pytest.approx(99_000_000 + 108_405_000)
    assert res.daily[1]["turnover"] == pytest.approx(99_000_000 / 100_000_000)
    assert s["endNavVnd"] == pytest.approx(108_985_487.5)
    assert res.rebalances[0]["entryDate"] == cal[1] and res.rebalances[0]["targets"] == ["A"]


def test_golden_lot_rounding_two_names_equal_weight():
    cal = _cal(3)
    bars = {"A": _flat(cal, 20_000), "B": _flat(cal, 7_000)}
    rules = _rules(top_k=2, keep_rank=2, initial_cash_vnd=10_000_000.0)
    res = pb.run_backtest(cal, bars, _periods("A", "B"), _by_date({cal[0]: {"A": 2.0, "B": 1.0}}), rules,
                          first_decision=0)
    got = {(t["symbol"], t["shares"], t["price"]) for t in res.trades}
    # per name 5,000,000: A 20,050 x 1.0015 -> 249.0 -> 200; B 7,010 x 1.0015 -> 712.2 -> 700
    assert got == {("A", 200, 20_050), ("B", 700, 7_010)}
    assert all(t["shares"] % 100 == 0 for t in res.trades)
    assert res.daily[1]["cash"] == pytest.approx(10_000_000 - 4_016_015 - 4_914_360.5)


def test_golden_buy_limit_lock_retry_then_t2_blocks_the_sell():
    cal = _cal(6)
    bars = {"A": _flat(cal, 10_700, {0: (10_000, 10_000)})}  # s1 opens at 1.07 x prev close: locked
    scores = _by_date({cal[0]: {"A": 1.0}})
    res = pb.run_backtest(cal, bars, _periods("A"), scores, _rules(), first_decision=0)
    assert [(r["date"], r["side"], r["reason"], r["retried"]) for r in res.rejections] == [
        (cal[1], "buy", "limit-locked", True),
        (cal[3], "sell", "shares-not-settled", True),
    ]
    # 1e8 / (10,750 x 1.0015) = 9,288.4 -> 9,200; bought s2 -> sellable s4; sell sent s3, filled s4
    assert [(t["date"], t["side"], t["shares"], t["price"]) for t in res.trades] == [
        (cal[2], "buy", 9_200, 10_750), (cal[4], "sell", 9_200, 10_650)]
    assert res.nav_by_date()[cal[1]] == pytest.approx(100_000_000)


def test_golden_sell_limit_lock_proxy_retries_next_session():
    cal = _cal(6)
    bars = {"A": _flat(cal, 10_000, {3: (9_300, 9_300), 4: (9_300, 9_300), 5: (9_300, 9_300)})}
    res = pb.run_backtest(cal, bars, _periods("A"), _by_date({cal[0]: {"A": 1.0}}), _rules(), first_decision=0)
    assert [(r["date"], r["side"], r["reason"]) for r in res.rejections] == [(cal[3], "sell", "limit-locked")]
    assert [(t["date"], t["side"], t["price"]) for t in res.trades] == [
        (cal[1], "buy", 10_050), (cal[4], "sell", 9_290)]


def test_golden_rebalance_t2_with_one_session_period():
    cal = _cal(5)
    bars = {"A": _flat(cal, 10_000)}
    res = pb.run_backtest(cal, bars, _periods("A"), _by_date({cal[0]: {"A": 1.0}}), _rules(rebalance_every=1),
                          first_decision=0)
    assert [(t["date"], t["side"]) for t in res.trades] == [(cal[1], "buy"), (cal[3], "sell")]
    assert [(r["date"], r["reason"]) for r in res.rejections] == [
        (cal[2], "shares-not-settled"), (cal[3], "superseded-by-rebalance")]


def test_raw_limit_info_replaces_the_proxy_but_not_the_tick_unless_opted_in():
    cal = _cal(4)
    bars = {"A": _flat(cal, 10_700, {0: (10_000, 10_000)})}
    scores = _by_date({cal[0]: {"A": 1.0}})
    raw = {"A": {cal[1]: pb.LimitInfo(raw_open=21_400, floor=18_600, ceiling=21_450)}}
    res = pb.run_backtest(cal, bars, _periods("A"), scores, _rules(), first_decision=0, limits=raw)
    assert [r for r in res.rejections if r["date"] <= cal[2]] == []
    t = res.trades[0]
    # frozen tickRule: tick of the ADJUSTED open (10,700 -> 50) even when raw data exists
    assert t["date"] == cal[1] and t["price"] == pytest.approx(10_750)
    opt = pb.run_backtest(cal, bars, _periods("A"), scores, _rules(raw_tick_slippage=True), first_decision=0,
                          limits=raw)
    assert opt.trades[0]["price"] == pytest.approx(10_700 + 10_700 * 50 / 21_400)  # 10,725
    assert opt.to_dict()["rules"]["rawTickSlippage"] is True and res.summary["rawTickSlippage"] is False
    # raw says locked although the adjusted proxy would fill
    bars2 = {"A": _flat(cal, 10_000)}
    raw2 = {"A": {cal[1]: pb.LimitInfo(raw_open=21_450, floor=18_650, ceiling=21_450)}}
    res2 = pb.run_backtest(cal, bars2, _periods("A"), scores, _rules(), first_decision=0, limits=raw2)
    assert [(r["date"], r["reason"]) for r in res2.rejections if r["date"] <= cal[2]] == [(cal[1], "limit-locked")]
    assert res2.trades[0]["date"] == cal[2]


def test_no_bar_is_retried():
    cal = _cal(4)
    a = _flat(cal, 10_000)
    del a[cal[1]]  # suspended at the entry session
    res = pb.run_backtest(cal, {"A": a}, _periods("A"), _by_date({cal[0]: {"A": 1.0}}), _rules(),
                          first_decision=0)
    # (the s3 sell of the late buy is then T+2-blocked, as it should be)
    assert [(r["date"], r["reason"]) for r in res.rejections] == [(cal[1], "no-bar"), (cal[3], "shares-not-settled")]
    assert res.trades[0]["date"] == cal[2]


def _hysteresis_run(keep_rank: int, resize_kept_trades: bool = True):
    cal = _cal(7)
    bars = {s: _flat(cal, 10_000) for s in "ABC"}
    scores = _by_date({cal[0]: {"A": 3, "B": 2, "C": 1}, cal[2]: {"B": 3, "A": 2, "C": 1},
                       cal[4]: {"C": 3, "B": 2, "A": 1}})
    return cal, pb.run_backtest(cal, bars, _periods("A", "B", "C"), scores, _rules(keep_rank=keep_rank),
                                first_decision=0)


def test_golden_hysteresis_keeps_a_holding_while_rank_within_keep_rank():
    cal, res = _hysteresis_run(keep_rank=2)
    assert [r["targets"] for r in res.rebalances] == [["A"], ["A"], ["C"]]
    full = [(t["date"], t["symbol"], t["side"]) for t in res.trades if t["symbol"] != "A" or t["date"] != cal[3]]
    assert full == [(cal[1], "A", "buy"), (cal[5], "A", "sell"), (cal[5], "C", "buy")]
    # the kept A is only trimmed back to equal weight at s3 (costs shrank the NAV), never fully sold
    trim = [t for t in res.trades if t["date"] == cal[3]]
    assert [(t["symbol"], t["side"], t["shares"]) for t in trim] == [("A", "sell", 100)]
    # without hysteresis A (rank 2) is replaced by B at s3
    cal, res1 = _hysteresis_run(keep_rank=1)
    assert [r["targets"] for r in res1.rebalances] == [["A"], ["B"], ["C"]]
    assert [(t["date"], t["symbol"], t["side"]) for t in res1.trades][:3] == [
        (cal[1], "A", "buy"), (cal[3], "A", "sell"), (cal[3], "B", "buy")]


def test_select_targets_rules():
    ranked = ["A", "B", "C", "D", "E"]
    assert pb.select_targets(ranked, set(), 2, 4) == ["A", "B"]
    assert pb.select_targets(ranked, {"D"}, 2, 4) == ["A", "D"]
    assert pb.select_targets(ranked, {"E"}, 2, 4) == ["A", "B"]
    assert pb.select_targets(ranked, {"C", "D"}, 2, 4) == ["C", "D"]
    assert pb.select_targets(ranked, {"A"}, None, None) == ranked
    assert pb.rank_members({"A": 1.0, "B": 1.0, "C": float("nan"), "X": 9.0, "D": 2.0},
                           frozenset("ABCD")) == ["D", "A", "B"]


def test_leaver_is_sold_at_the_next_rebalance():
    cal = _cal(6)
    bars = {"A": _flat(cal, 10_000), "B": _flat(cal, 10_000)}
    periods = [(cal[0], frozenset({"A", "B"})), (cal[1], frozenset({"B"}))]
    always_a = lambda day, view: {"A": 2.0, "B": 1.0}  # noqa: E731
    res = pb.run_backtest(cal, bars, periods, always_a, _rules(), first_decision=0)
    assert [r["targets"] for r in res.rebalances] == [["A"], ["B"], ["B"]]
    assert [(t["date"], t["symbol"], t["side"]) for t in res.trades][:3] == [
        (cal[1], "A", "buy"), (cal[3], "A", "sell"), (cal[3], "B", "buy")]


def test_equal_weight_benchmark_mode():
    cal = _cal(3)
    bars = {"A": _flat(cal, 10_000), "B": _flat(cal, 10_000), "C": _flat(cal, 10_000)}
    rules = _rules(top_k=None, keep_rank=None, initial_cash_vnd=30_000_000.0)
    res = pb.run_backtest(cal, bars, _periods("A", "B"), lambda d, v: {s: 0.0 for s in v.members}, rules,
                          first_decision=0)
    # 15,000,000 / (10,050 x 1.0015) = 1,490.3 -> 1,400 each; C is not a member
    assert sorted((t["symbol"], t["shares"]) for t in res.trades) == [("A", 1_400), ("B", 1_400)]


def test_price_unit_thousands_matches_vnd():
    cal = _cal(5)
    vnd = {"A": _flat(cal, 9_990, {1: (9_990, 10_500), 2: (11_000, 11_000), 3: (11_000, 11_000)})}
    kvnd = {"A": {d: tuple(x / 1000 if k < 4 else x for k, x in enumerate(b)) for d, b in vnd["A"].items()}}
    scores = _by_date({cal[0]: {"A": 1.0}})
    a = pb.run_backtest(cal, vnd, _periods("A"), scores, _rules(), first_decision=0)
    b = pb.run_backtest(cal, kvnd, _periods("A"), scores, _rules(price_unit_vnd=1000.0), first_decision=0)
    assert [r["nav"] for r in b.daily] == pytest.approx([r["nav"] for r in a.daily])
    assert b.summary["costVnd"] == pytest.approx(a.summary["costVnd"])


def test_cash_t2_option_delays_buying_power():
    cal = _cal(7)
    bars = {"A": _flat(cal, 10_000), "B": _flat(cal, 10_000)}
    scores = _by_date({cal[0]: {"A": 2, "B": 1}, cal[3]: {"B": 2, "A": 1}})
    res = pb.run_backtest(cal, bars, _periods("A", "B"), scores,
                          _rules(rebalance_every=3, cash_settlement_sessions=2), first_decision=0)
    sides = [(t["date"], t["symbol"], t["side"], t["shares"]) for t in res.trades]
    # A sold at s4; its proceeds are spendable from s6, so B (retried while cash is pending) fills at s6.
    # B is sized net of A's exit costs: (99,355,757.5 - 741,262.5) / 10,065.075 = 9,797.7 -> 9,700
    assert sides == [(cal[1], "A", "buy", 9_900), (cal[4], "A", "sell", 9_900), (cal[6], "B", "buy", 9_700)]
    assert [(r["date"], r["reason"], r["retried"]) for r in res.rejections] == [
        (cal[4], "insufficient-cash", True), (cal[5], "insufficient-cash", True)]


def test_first_decision_index_rule():
    cal = _cal(10)
    bars = {"A": {d: (1, 1, 1, 1) for d in cal}, "B": {d: (1, 1, 1, 1) for d in cal[3:]}}
    periods = [(cal[0], frozenset({"A", "B"}))]
    assert pb.first_decision_index(cal, bars, periods, min_history=4, min_members=2) == 7
    assert pb.first_decision_index(cal, bars, periods, min_history=4, min_members=1) == 4
    assert pb.first_decision_index(cal, bars, periods, min_history=20, min_members=1) is None


def test_rules_from_frozen_protocol():
    rules = pb.PortfolioRules.from_protocol(AI_ROOT / "docs" / "research" / "protocols" / "vn30-2.json")
    assert (rules.rebalance_every, rules.top_k, rules.keep_rank, rules.slippage_ticks) == (20, 10, 15, 1)
    assert (rules.limit_up_proxy, rules.limit_down_proxy) == (1.065, 0.935)
    c = rules.costs
    assert (c.broker_fee_fraction, c.sell_tax_fraction, c.lot_size, c.settlement_trading_days) == (
        0.0015, 0.001, 100, 2)
    assert rules.min_history_sessions == 260 and rules.share_lag == 2 and not rules.raw_tick_slippage
    assert rules == pb.PortfolioRules()


def test_cash_short_buy_waits_for_a_limit_locked_sell():
    # A is dropped for B; A's entry open is 0.93 x the previous close, so its sell is limit-locked and B
    # cannot be paid. B must stay open (not be dropped) and fill the next session, when A's sale fills.
    cal = _cal(7)
    bars = {"A": _flat(cal, 10_000, {k: (9_300, 9_300) for k in range(3, 7)}), "B": _flat(cal, 10_000)}
    scores = _by_date({cal[0]: {"A": 1.0}, cal[2]: {"B": 2.0, "A": 1.0}})
    res = pb.run_backtest(cal, bars, _periods("A", "B"), scores, _rules(), first_decision=0)
    assert [(r["date"], r["symbol"], r["side"], r["reason"], r["retried"]) for r in res.rejections
            if r["date"] == cal[3]] == [(cal[3], "A", "sell", "limit-locked", True),
                                        (cal[3], "B", "buy", "insufficient-cash", True)]
    # A's tick below 10,000 is 10. B sized at s3:
    # (355,757.5 + 9,900 x 9,300 - 9,900 x (10 + 9,290 x 0.0025)) / 10,065.075 = 9,150.1 -> 9,100
    assert [(t["date"], t["symbol"], t["side"], t["shares"], t["price"]) for t in res.trades][:3] == [
        (cal[1], "A", "buy", 9_900, 10_050), (cal[4], "A", "sell", 9_900, 9_290),
        (cal[4], "B", "buy", 9_100, 10_050)]
    assert res.daily[4]["positions"] == 1 and res.daily[4]["cash"] < 0.01 * res.daily[4]["nav"]


def test_cash_capped_remainder_waits_for_a_limit_locked_sell():
    cal = _cal(6)
    bars = {"A": _flat(cal, 10_000, {k: (9_300, 9_300) for k in range(3, 6)}),
            "C": _flat(cal, 10_000), "D": _flat(cal, 10_000)}
    scores = _by_date({cal[0]: {"A": 1.0}, cal[2]: {"C": 3.0, "D": 2.0, "A": 1.0}})
    res = pb.run_backtest(cal, bars, _periods("A", "C", "D"), scores, _rules(top_k=2, keep_rank=2),
                          first_decision=0)
    assert [(r["date"], r["symbol"], r["reason"], r["shares"], r["retried"]) for r in res.rejections
            if r["date"] == cal[3]] == [(cal[3], "A", "limit-locked", 4_900, True),
                                        (cal[3], "D", "cash-capped", 4_400, True)]
    assert [(t["date"], t["symbol"], t["side"], t["shares"]) for t in res.trades][:5] == [
        (cal[1], "A", "buy", 4_900), (cal[3], "C", "buy", 4_700), (cal[3], "D", "buy", 300),
        (cal[4], "A", "sell", 4_900), (cal[4], "D", "buy", 4_400)]


def test_cash_short_buy_is_dropped_when_no_cash_can_come():
    cal = _cal(5)
    a = _flat(cal, 10_600)
    a[cal[0]] = (10_000, 10_000, 10_000, 10_000, 1e6)
    del a[cal[1]], a[cal[2]]  # sized at s2 from the last close 10,000, filled at s3 at 10,600 + 50
    res = pb.run_backtest(cal, {"A": a}, _periods("A"), _by_date({cal[1]: {"A": 1.0}}), _rules(),
                          first_decision=1)
    assert res.rebalances[0]["targetShares"] == {"A": 9_900}
    # 1e8 / (10,650 x 1.0015) = 9,375.6 -> 9,300; nothing to sell, no cash pending: the rest is dropped
    assert (res.trades[0]["date"], res.trades[0]["shares"]) == (cal[3], 9_300)
    assert [(r["date"], r["reason"], r["shares"], r["retried"]) for r in res.rejections if r["date"] <= cal[3]] == [
        (cal[2], "no-bar", 9_900, True), (cal[3], "cash-capped", 600, False)]


def test_last_close_is_seeded_from_history_before_the_first_decision():
    cal = _cal(5)
    a = _flat(cal, 10_700, {0: (10_000, 10_000)})
    del a[cal[1]]  # suspended on the first decision session
    res = pb.run_backtest(cal, {"A": a}, _periods("A"), _by_date({cal[1]: {"A": 1.0}}), _rules(),
                          first_decision=1)
    # entry s2 opens 10,700 >= 1.065 x 10,000 (the close before the start): locked, filled at s3
    assert [(r["date"], r["reason"]) for r in res.rejections][:1] == [(cal[2], "limit-locked")]
    assert res.trades[0]["date"] == cal[3]
    b = _flat(cal, 10_000)
    del b[cal[1]], b[cal[2]]  # no bar at the decision nor at the entry: sized from the seeded close
    res2 = pb.run_backtest(cal, {"A": b}, _periods("A"), _by_date({cal[1]: {"A": 1.0}}), _rules(),
                           first_decision=1)
    assert res2.rebalances[0]["targetShares"] == {"A": 9_900}
    assert [(r["date"], r["reason"]) for r in res2.rejections][:1] == [(cal[2], "no-bar")]
    assert (res2.trades[0]["date"], res2.trades[0]["shares"]) == (cal[3], 9_900)


def test_minimum_history_filter_is_applied_before_ranking_for_every_mode():
    cal = _cal(6)
    bars = {"A": _flat(cal, 10_000), "B": {d: b for d, b in _flat(cal, 10_000).items() if d >= cal[2]}}
    seen = []

    def score(day, view):
        seen.append(view.members)
        return {"A": 1.0, "B": 2.0}

    for top_k, keep in ((1, 1), (None, None)):
        res = pb.run_backtest(cal, bars, _periods("A", "B"), score,
                              _rules(top_k=top_k, keep_rank=keep, min_history_sessions=3), first_decision=3)
        first = res.rebalances[0]
        assert first["targets"] == ["A"] and first["excludedShortHistory"] == ["B"]
        assert res.summary["minHistorySessions"] == 3
    assert seen[0] == frozenset({"A"})


def test_shares_sellable_after_three_sessions_option():
    cal = _cal(6)
    bars = {"A": _flat(cal, 10_000)}
    res = pb.run_backtest(cal, bars, _periods("A"), _by_date({cal[0]: {"A": 1.0}}),
                          _rules(rebalance_every=1, shares_sellable_after_sessions=3), first_decision=0)
    assert [(t["date"], t["side"]) for t in res.trades] == [(cal[1], "buy"), (cal[4], "sell")]
    assert res.summary["sharesSellableAfterSessions"] == 3
    with pytest.raises(ValueError):
        _rules(shares_sellable_after_sessions=1)


def test_limit_info_must_be_vnd():
    cal = _cal(3)
    bars = {"A": _flat(cal, 21.4)}
    scores = _by_date({cal[0]: {"A": 1.0}})
    for bad in (pb.LimitInfo(21.4, 18.6, 21.45), pb.LimitInfo(21_400, 21_450, 18_600)):
        with pytest.raises(ValueError):
            pb.run_backtest(cal, bars, _periods("A"), scores, _rules(price_unit_vnd=1000.0), first_decision=0,
                            limits={"A": {cal[1]: bad}})


def test_stale_positions_are_reported():
    cal = _cal(10)
    a = {d: b for d, b in _flat(cal, 10_000).items() if d <= cal[2]}  # A stops trading after s2
    bars = {"A": a, "B": _flat(cal, 10_000)}
    scores = _by_date({cal[0]: {"A": 1.0}, cal[2]: {"B": 1.0}})
    res = pb.run_backtest(cal, bars, _periods("A", "B"), scores, _rules(), first_decision=0, stale_after=2)
    s = res.summary
    assert s["stalePositionsAtEnd"] == [{"symbol": "A", "sessionsWithoutBar": 7, "shares": 9_900,
                                         "markedAt": 10_000.0, "valueVnd": 99_000_000.0}]
    assert s["stalePositionSessions"] == 5 and s["maxStalePositions"] == 1
    assert s["stalePositionsAtEndValueVnd"] == pytest.approx(99_000_000)
    assert [r["stalePositions"] for r in res.daily] == [0] * 5 + [1] * 5


def _random_market(seed: int, symbols: int, sessions: int, gap_vol: float = 0.01) -> tuple[list[str], dict]:
    rng = random.Random(seed)
    cal = _cal(sessions)
    bars = {}
    for k in range(symbols):
        px, rows = rng.uniform(8_000, 80_000), {}
        for d in cal:
            o = px * (1 + rng.gauss(0, gap_vol))
            c = o * (1 + rng.gauss(0, 0.02))
            rows[d] = (o, max(o, c) * 1.005, min(o, c) * 0.995, c, rng.uniform(1e5, 1e6))
            px = c
        bars[f"S{k:02d}"] = rows
    return cal, bars


def _momentum(day, view):
    out = {}
    for s in view.members:
        hist = view.history(s)
        assert hist and hist[-1][0] <= view.date == day  # the view never shows a later bar
        closes = [b[3] for _, b in hist]
        if len(closes) > 10:
            out[s] = closes[-1] / closes[-11] - 1
    return out


def test_look_ahead_perturbing_future_bars_does_not_change_the_past():
    cal, bars = _random_market(7, 12, 120)
    periods = [(cal[0], frozenset(bars))]
    rules = _rules(rebalance_every=5, top_k=3, keep_rank=5, initial_cash_vnd=1_000_000_000.0)
    base = pb.run_backtest(cal, bars, periods, _momentum, rules, first_decision=15)
    cut = cal[60]
    rng = random.Random(99)
    future = {s: {d: (tuple(x * rng.uniform(0.5, 1.5) for x in b) if d > cut else b) for d, b in rows.items()}
              for s, rows in bars.items()}
    pert = pb.run_backtest(cal, future, periods, _momentum, rules, first_decision=15)
    assert base.trades != pert.trades  # the perturbation is real
    assert [r for r in base.daily if r["date"] <= cut] == [r for r in pert.daily if r["date"] <= cut]
    assert [t for t in base.trades if t["date"] <= cut] == [t for t in pert.trades if t["date"] <= cut]
    assert [r for r in base.rejections if r["date"] <= cut] == [r for r in pert.rejections if r["date"] <= cut]
    keys = ("decisionDate", "ranks", "targets", "heldBefore")
    past = lambda res: [{k: r[k] for k in keys} for r in res.rebalances if r["decisionDate"] <= cut]  # noqa: E731
    assert past(base) == past(pert) and len(past(base)) == 10
    assert base.summary["trades"] > 0 and all(t["shares"] % 100 == 0 for t in base.trades)


def test_invariants_on_a_random_market():
    cal, bars = _random_market(3, 30, 200)
    periods = [(cal[0], frozenset(list(bars)[:25])), (cal[100], frozenset(list(bars)[5:]))]
    res = pb.run_backtest(cal, bars, periods, _momentum, _rules(rebalance_every=20, top_k=10, keep_rank=15,
                                                                initial_cash_vnd=10_000_000_000.0),
                          first_decision=20)
    assert all(r["cash"] >= -1e-6 for r in res.daily)
    assert all(r["positions"] <= 15 for r in res.daily)
    assert all(t["shares"] > 0 and t["shares"] % 100 == 0 for t in res.trades)
    left = set(list(bars)[:5])
    after = [r for r in res.rebalances if r["decisionDate"] >= cal[100]]
    assert after and not any(set(r["targets"]) & left for r in after)
    d = res.to_dict()
    assert d["schema"] == pb.SCHEMA and d["priceBasis"] == "adjusted"
    assert d["rules"]["cashSettlementSessions"] == 0 == d["summary"]["cashSettlementSessions"]


def test_volatile_market_never_drops_a_buy_for_cash():
    # 3% open gaps hit the limit proxy often. A cash-short buy may not be dropped while a sell is still open,
    # and (trim costs being in the sizing) never on an entry session, whose open is the sizing price. It may
    # only be cut on a later retry whose open moved up, once every sell has filled.
    cal, bars = _random_market(11, 30, 600, gap_vol=0.03)
    res = pb.run_backtest(cal, bars, [(cal[0], frozenset(bars))], _momentum,
                          _rules(rebalance_every=20, top_k=10, keep_rank=15, initial_cash_vnd=10_000_000_000.0),
                          first_decision=20)
    assert sum(r["reason"] == "limit-locked" and r["side"] == "sell" for r in res.rejections) >= 2
    dropped = [r for r in res.rejections if r["reason"] in ("insufficient-cash", "cash-capped") and not r["retried"]]
    open_sell_days = {r["date"] for r in res.rejections if r["side"] == "sell" and r["retried"]}
    entries = {r["entryDate"] for r in res.rebalances}
    assert not [r for r in dropped if r["date"] in open_sell_days or r["date"] in entries]
    assert all(r["cash"] >= -1e-6 for r in res.daily)
    # no idle cash after limit-down days: more than 3% cash only while a buy itself is locked or has no bar
    waiting_buy_days = {r["date"] for r in res.rejections
                        if r["side"] == "buy" and r["reason"] in ("limit-locked", "no-bar")}
    assert not [r["date"] for r in res.daily[21:] if r["cash"] > 0.03 * r["nav"] and r["date"] not in waiting_buy_days]
