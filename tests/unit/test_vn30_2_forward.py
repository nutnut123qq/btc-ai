"""vn30_2_forward: live splice, schedule, protocol-equivalent scores, immutable ledger and forward evaluation."""

import json
import math
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import trial_registry as tr
import vn30_2_forward as fw
import vn30_2_run as run
import vn30_panel_eval as ev

L = "2020-06-30"  # last session of the synthetic "snapshot"
COSTS = {"buyFee": 0.0015, "sellFee": 0.0015, "sellTax": 0.001, "slippageTicks": 1}
NO_HOLIDAYS = frozenset()
COVER = "2030-12-31"


def synthetic_full(n_symbols=23, seed=11, end="2021-03-31"):
    rng = np.random.default_rng(seed)
    cal = [d.date().isoformat() for d in pd.bdate_range("2016-01-04", end)]
    syms = [f"S{j:02d}" for j in range(n_symbols)]
    bars = {}
    for j, s in enumerate(syms):
        close = 20_000.0 * (1 + j / 30)
        drift = 0.0006 * ((j % 5) - 2) / 2
        rows = {}
        for k, day in enumerate(cal):
            open_ = close * (1 + rng.normal(0, 0.004))
            close = max(1_000.0, open_ * (1 + rng.normal(drift, 0.017)))
            high = max(open_, close) * (1 + abs(rng.normal(0, 0.004)))
            low = min(open_, close) * (1 - abs(rng.normal(0, 0.004)))
            rows[day] = (open_, high, low, close, float(rng.integers(100_000, 1_000_000)))
        bars[s] = rows
    # membership starts after the protocol history start (the evaluator refuses decisions before 2018-01-22); the
    # date keeps the 20-session grid phase the tests were written for
    history_periods = [("2018-01-29", frozenset(syms[0:21])), ("2019-07-01", frozenset(syms[1:22]))]
    periods = history_periods + [("2020-07-01", frozenset(syms[0:21]))]
    return run.Market(tuple(cal), bars, periods, {}), history_periods


@pytest.fixture(scope="module")
def full():
    return synthetic_full()


@pytest.fixture(scope="module")
def history(full):
    market, history_periods = full
    cut = market.truncate(L)
    return run.Market(cut.calendar, cut.bars, history_periods, {})


def _ms(day):
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)


def fake_backend(market, asof, *, ratio=1.0, served=None, drop=(), calls=None, catalog=None, lacking=None):
    """A backend serving ``market`` bars dated <= asof (prices x ratio, as after a later re-adjustment).

    ``drop``: VNINDEX sessions not ingested; ``lacking``: {symbol: sessions without a bar}; ``catalog``: the symbols
    of GET /api/equities (default: the members in force at ``asof`` + VNINDEX)."""
    def fetch(path):
        if calls is not None:
            calls.append(path)
        if path == "/api/equities":
            syms = sorted(catalog if catalog is not None else (fw.members_on(market.periods, asof) or ()))
            return {"items": [{"symbol": "VNINDEX", "assetClass": "index"}]
                    + [{"symbol": x, "assetClass": "equity"} for x in syms]}
        sym = path.split("/")[3]
        limit = int(re.search(r"limit=(\d+)", path).group(1))
        if sym == "VNINDEX":
            rows = {d: (1000.0, 1001.0, 999.0, 1000.0, 1e6) for d in market.calendar if d <= asof and d not in drop}
        elif served is not None and sym not in served:
            raise fw.NotServed(path)
        else:
            gone = set((lacking or {}).get(sym, ()))
            rows = {d: tuple(x * ratio for x in b[:4]) + (b[4],) for d, b in market.bars[sym].items()
                    if d <= asof and d not in gone}
        return [{"openTimeMs": _ms(d), "open": rows[d][0], "high": rows[d][1], "low": rows[d][2],
                 "close": rows[d][3], "volume": rows[d][4]} for d in sorted(rows)[-limit:]]
    return fetch


def _evening(day):  # 17:00 ICT of ``day``
    return datetime.fromisoformat(day).replace(hour=10, tzinfo=timezone.utc)


def _grid_after(full_market, history, n):
    first = fw.grid_first(history)
    idx = [g for g in range(first, len(full_market.calendar), fw.REBALANCE) if full_market.calendar[g] > L]
    return first, idx[:n]


def _record(history, full_market, asof, ledger, **kw):
    return fw.record(history, fake_backend(full_market, asof, **{k: v for k, v in kw.items() if k != "now"}),
                     now_utc=kw.get("now", _evening(asof)), periods=full_market.periods, non_trading=NO_HOLIDAYS,
                     coverage_to=COVER, meta={"data": {"snapshotSha256": "s" * 64}}, ledger=ledger,
                     forward_start="2020-01-01")


# ------------------------------------------------------------------ schedule and scores


def test_fit_plan_equals_the_protocol_yearly_folds(full, history):
    market, _ = full
    first = fw.grid_first(history)
    decisions = list(range(first, len(market.calendar), fw.REBALANCE))
    dates = [market.calendar[s] for s in decisions]
    last = ev.last_session_on_or_before(market.calendar, "2020-12-31")
    folds = {f.year: f for f in ev.yearly_folds(decisions, dates, test_from="2018-01-01", test_to="2020-12-31",
                                                calendar=market.calendar,
                                                last_session_in_window=last)}
    for i in decisions:
        year = int(market.calendar[i][:4])
        if year not in folds or i not in folds[year].test_sessions:
            continue
        plan = fw.fit_plan(market.calendar, first, i)
        assert plan["fitAt"] == folds[year].fit_session
        assert tuple(plan["train"]) == folds[year].train_sessions
    with pytest.raises(ValueError, match="not a grid decision"):
        fw.fit_plan(market.calendar, first, first + 3)


def test_decision_scores_equal_the_walk_forward_and_ignore_the_future(full, history):
    market, _ = full
    first = fw.grid_first(history)
    decisions = list(range(first, len(market.calendar), fw.REBALANCE))
    i = next(g for g in decisions if market.calendar[g] >= "2019-09-01")
    dates = [market.calendar[s] for s in decisions]
    dev = market.truncate("2019-12-31")
    last = ev.last_session_on_or_before(dev.calendar, "2019-12-31")
    in_dev = [s for s in decisions if s <= last]
    panel = run.build_panel(dev, in_dev)
    folds = ev.yearly_folds(in_dev, dates[:len(in_dev)], test_from="2019-01-01", test_to="2019-12-31",
                            calendar=dev.calendar,
                            last_session_in_window=last)
    got = fw.decision_scores(market, first, i)
    rows = panel[panel["session"] == i]
    assert got["eligible"] == sorted(rows["symbol"])
    for m in ("B-composite", "B-mom_6", "M-ridge", "M-lgbm"):
        ref = ev.walk_forward_scores(panel, folds, run.FIT_PREDICT[m])
        for idx, sym in zip(rows.index, rows["symbol"]):
            assert got["scores"][m][sym] == pytest.approx(float(ref.loc[idx]), abs=1e-12)
    # bars after d never change the record
    poisoned = {s: {d: ((b[0] * 3, b[1] * 3, b[2] * 3, b[3] * 3, b[4]) if d > market.calendar[i] else b)
                    for d, b in rows_.items()} for s, rows_ in market.bars.items()}
    again = fw.decision_scores(run.Market(market.calendar, poisoned, market.periods), first, i)
    assert again["scores"] == got["scores"]
    assert got["fit"]["fitDecisionDate"].startswith("2019") and got["fit"]["trainDecisions"] > 0


def test_selection_keeps_holdings_ranked_15_or_better():
    syms = [f"X{k:02d}" for k in range(20)]
    scores = {s: float(20 - k) for k, s in enumerate(syms)}  # X00 best
    fresh = fw.select("M-ridge", scores, syms, set())
    assert fresh == syms[:10]
    kept = fw.select("M-ridge", scores, syms, {"X13", "X17"})
    assert "X13" in kept and "X17" not in kept and len(kept) == 10
    assert fw.select("B-ew", {s: 0.0 for s in syms}, syms, set()) == sorted(syms)


# ------------------------------------------------------------------ live data


def test_splice_scales_history_to_a_readjusted_backend(full, history):
    market, _ = full
    live_cal = [d for d in market.calendar if L < d <= "2020-07-31"]
    sym = "S03"
    live = {d: tuple(x * 0.97 for x in b[:4]) + (b[4],) for d, b in market.bars[sym].items() if d <= "2020-07-31"}
    bars, ratio = fw.splice(history.bars[sym], live, history.calendar, live_cal, sym)
    assert ratio == pytest.approx(0.97)
    d0, d1 = history.calendar[-200], history.calendar[-1]
    assert bars[d1][3] / bars[d0][3] == pytest.approx(history.bars[sym][d1][3] / history.bars[sym][d0][3])
    assert bars[live_cal[0]] == live[live_cal[0]]
    assert bars[d1][4] == history.bars[sym][d1][4]
    same, one = fw.splice(history.bars[sym], dict(market.bars[sym]), history.calendar, live_cal, sym)
    assert one == 1.0 and same[d1] == history.bars[sym][d1]
    broken = dict(live)
    k = history.calendar[-5]
    broken[k] = tuple(x * 1.05 for x in broken[k][:4]) + (broken[k][4],)
    with pytest.raises(fw.ForwardError, match="not constant"):
        fw.splice(history.bars[sym], broken, history.calendar, live_cal, sym)


def test_live_calendar_fails_closed_on_gaps(full, history):
    market, _ = full
    idx = {d: (1, 1, 1, 1, 1) for d in market.calendar if d <= "2020-07-31"}
    live = fw.live_calendar(history.calendar, idx, NO_HOLIDAYS, COVER)
    assert live[0] > L and live == [d for d in market.calendar if L < d <= "2020-07-31"]
    gap = {d: b for d, b in idx.items() if d != "2020-07-15"}
    with pytest.raises(fw.ForwardError, match="HOSE calendar"):
        fw.live_calendar(history.calendar, gap, NO_HOLIDAYS, COVER)
    with pytest.raises(fw.ForwardError, match="overlap"):
        fw.live_calendar(history.calendar, {d: b for d, b in idx.items() if d != history.calendar[-3]},
                         NO_HOLIDAYS, COVER)
    # a listed holiday is not expected
    holiday = frozenset({"2020-07-15"})
    assert "2020-07-15" not in fw.live_calendar(history.calendar, gap, holiday, COVER)
    with pytest.raises(fw.ForwardError, match="covers up to"):
        fw.live_calendar(history.calendar, idx, NO_HOLIDAYS, "2020-07-01")


def test_finalized_bar_rule():
    bars = {"2026-11-19": (1,), "2026-11-20": (1,), "2026-11-21": (1,)}
    before = datetime(2026, 11, 20, 8, 30, tzinfo=timezone.utc)  # 15:30 ICT
    after = datetime(2026, 11, 20, 9, 5, tzinfo=timezone.utc)  # 16:05 ICT
    assert sorted(fw.finalized(bars, before)) == ["2026-11-19"]
    assert sorted(fw.finalized(bars, after)) == ["2026-11-19", "2026-11-20"]


def test_trading_days_and_next_open():
    hol = frozenset({"2026-11-23"})
    assert fw.trading_days("2026-11-19", "2026-11-24", hol, COVER) == ["2026-11-20", "2026-11-24"]
    assert fw.next_trading_day("2026-11-20", hol) == "2026-11-24"


def test_symbols_in_force_between():
    periods = [("2020-01-01", frozenset({"A", "B"})), ("2020-07-01", frozenset({"B", "C"})),
               ("2021-01-01", frozenset({"C", "D"}))]
    assert fw.symbols_in_force_between(periods, "2020-06-15", "2020-08-01") == {"A", "B", "C"}
    assert fw.symbols_in_force_between(periods, "2021-02-01", "2021-03-01") == {"C", "D"}


# ------------------------------------------------------------------ record


def test_record_writes_only_forward_grid_dates_and_never_backfills(full, history, tmp_path):
    market, _ = full
    first, (d1, d2, d3) = _grid_after(market, history, 3)
    cal = market.calendar
    ledger = tmp_path / "vn30-2.jsonl"
    calls = []
    out = fw.record(history, fake_backend(market, cal[d1 - 1], calls=calls), now_utc=_evening(cal[d1 - 1]),
                    periods=market.periods, non_trading=NO_HOLIDAYS, coverage_to=COVER, meta={}, ledger=ledger,
                    forward_start="2020-01-01")
    assert out["status"] == "not-a-decision-date" and out["sessionsToNextDecision"] == 1
    assert len(calls) == 1 and "/VNINDEX/" in calls[0]  # no member fetch on a non-decision day
    assert not ledger.exists()

    out = _record(history, market, cal[d1], ledger)
    assert out["status"] == "recorded" and out["decisionDate"] == cal[d1]
    rows = fw.load_ledger(ledger)
    rec = rows[0]
    assert rec["decisionSession"] == d1 and rec["gridNumber"] == (d1 - first) // 20
    assert set(rec["models"]) == set(run.MODELS)
    assert all(len(rec["models"][m]["selection"]) == 10 for m in run.RANKED)
    assert rec["models"]["B-ew"]["selection"] == rec["eligible"]
    assert rec["inForce"] == sorted(market.periods[-1][1])
    assert rec["data"]["liveSessions"]["last"] == cal[d1] and rec["data"]["snapshotSha256"] == "s" * 64
    assert rec["fit"]["fitDecisionDate"].startswith(cal[d1][:4])
    # identical to scoring the full market cut at d (live data == history data here)
    direct = fw.decision_scores(market, first, d1)
    assert rec["models"]["M-lgbm"]["scores"] == pytest.approx(direct["scores"]["M-lgbm"])

    assert _record(history, market, cal[d1], ledger)["status"] == "unchanged"
    assert _record(history, market, cal[d1 + 1], ledger)["status"] == "not-a-decision-date"
    # d2 is skipped (machine off); d3 is recorded and lists d2 as missed, d2 can never be written afterwards
    out = _record(history, market, cal[d3], ledger)
    assert out["status"] == "recorded" and out["missedDecisions"] == [cal[d2]]
    rows = fw.load_ledger(ledger)
    assert rows[1]["previousRecordContiguous"] is False and rows[1]["prevSha256"] == rows[0]["sha256"]
    with pytest.raises(fw.LedgerError, match="backfill"):
        fw.append_record({**rows[0], "decisionDate": cal[d2]}, ledger)


def test_record_deadline_membership_and_unserved_members(full, history, tmp_path):
    market, _ = full
    _, (d1,) = _grid_after(market, history, 1)
    cal = market.calendar
    late = datetime.fromisoformat(cal[d1 + 1]).replace(hour=3, tzinfo=timezone.utc)  # 10:00 ICT next session
    out = _record(history, market, cal[d1], tmp_path / "a.jsonl", now=late)
    assert out["status"] == "missed-deadline" and not (tmp_path / "a.jsonl").exists()
    served = set(market.bars) - {"S05"}
    with pytest.raises(fw.ForwardError, match="not served"):
        _record(history, market, cal[d1], tmp_path / "b.jsonl", served=served)
    with pytest.raises(fw.ForwardError, match="does not start"):
        fw.record(history, fake_backend(market, cal[d1]), now_utc=_evening(cal[d1]), periods=market.periods[1:],
                  non_trading=NO_HOLIDAYS, coverage_to=COVER, meta={}, ledger=tmp_path / "c.jsonl",
                  forward_start="2020-01-01")
    stale = history.periods  # no period after 2019-07-01: > 190 days old in 2020-07+
    with pytest.raises(fw.ForwardError, match="older than"):
        fw.record(history, fake_backend(market, cal[d1]), now_utc=_evening(cal[d1]), periods=stale,
                  non_trading=NO_HOLIDAYS, coverage_to=COVER, meta={}, ledger=tmp_path / "d.jsonl",
                  forward_start="2020-01-01")
    out = fw.record(history, fake_backend(market, cal[d1]), now_utc=_evening(cal[d1]), periods=market.periods,
                    non_trading=NO_HOLIDAYS, coverage_to=COVER, meta={}, ledger=tmp_path / "e.jsonl",
                    forward_start="2030-01-01")
    assert out["status"] == "before-forward-start"


def test_readjusted_backend_gives_the_same_scores(full, history, tmp_path):
    market, _ = full
    _, (d1,) = _grid_after(market, history, 1)
    cal = market.calendar
    _record(history, market, cal[d1], tmp_path / "a.jsonl")
    _record(history, market, cal[d1], tmp_path / "b.jsonl", ratio=0.95)
    a, b = fw.load_ledger(tmp_path / "a.jsonl")[0], fw.load_ledger(tmp_path / "b.jsonl")[0]
    assert b["data"]["spliceRatios"]["S00"] == pytest.approx(0.95)
    for m in ("B-mom_6", "B-rev_1", "M-ridge"):
        assert a["models"][m]["selection"] == b["models"][m]["selection"]


def test_tampering_is_detected(full, history, tmp_path):
    market, _ = full
    _, (d1, d2) = _grid_after(market, history, 2)
    ledger = tmp_path / "l.jsonl"
    _record(history, market, market.calendar[d1], ledger)
    _record(history, market, market.calendar[d2], ledger)
    lines = ledger.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["models"]["M-ridge"]["selection"] = list(reversed(first["models"]["M-ridge"]["selection"]))
    ledger.write_text(json.dumps(first) + "\n" + lines[1] + "\n", encoding="utf-8")
    with pytest.raises(fw.LedgerError, match="sha256 mismatch"):
        fw.load_ledger(ledger)
    ledger.write_text(lines[1] + "\n", encoding="utf-8")
    with pytest.raises(fw.LedgerError, match="prevSha256"):
        fw.load_ledger(ledger)


def test_record_refuses_a_member_without_its_decision_bar(full, history, tmp_path):
    market, _ = full
    _, (d1,) = _grid_after(market, history, 1)
    cal = market.calendar
    ledger = tmp_path / "l.jsonl"
    out = _record(history, market, cal[d1], ledger, lacking={"S03": {cal[d1]}})  # S03 not ingested yet
    assert out["status"] == "data-not-ready" and out["membersWithoutDecisionBar"] == ["S03"]
    assert out["decisionDate"] == cal[d1] and not ledger.exists()
    # an earlier live gap (e.g. a suspension) is allowed and listed
    out = _record(history, market, cal[d1], ledger, lacking={"S03": {cal[d1 - 5]}})
    assert out["status"] == "recorded" and out["liveGaps"] == {"S03": [cal[d1 - 5]]}
    assert fw.load_ledger(ledger)[0]["data"]["liveGaps"] == {"S03": [cal[d1 - 5]]}


def test_record_late_index_ingest_is_data_not_ready_until_the_deadline(full, history, tmp_path):
    market, _ = full
    _, (d1,) = _grid_after(market, history, 1)
    cal = market.calendar
    ledger = tmp_path / "l.jsonl"
    # 17:00 ICT on the decision date, VNINDEX of d not ingested yet: not a silent "not-a-decision-date"
    out = _record(history, market, cal[d1], ledger, drop={cal[d1]})
    assert out["status"] == "data-not-ready" and out["decisionDate"] == cal[d1]
    assert out["latestSession"] == cal[d1 - 1] and out["backendLagSessions"] == [cal[d1]]
    assert not ledger.exists()
    # 08:30 ICT next session, ingest finished: recorded inside the window
    morning = datetime.fromisoformat(cal[d1 + 1]).replace(hour=1, minute=30, tzinfo=timezone.utc)
    out = _record(history, market, cal[d1], ledger, now=morning)
    assert out["status"] == "recorded" and out["decisionDate"] == cal[d1]
    # still missing after 09:00 of the next session: missed-deadline (exit 3), never written
    late = datetime.fromisoformat(cal[d1 + 1]).replace(hour=3, tzinfo=timezone.utc)
    other = tmp_path / "o.jsonl"
    out = _record(history, market, cal[d1], other, drop={cal[d1]}, now=late)
    assert out["status"] == "missed-deadline" and out["decisionDate"] == cal[d1] and not other.exists()
    # a lag on a non-decision date is reported but harmless
    out = _record(history, market, cal[d1 + 3], ledger, drop={cal[d1 + 3]})
    assert out["status"] == "not-a-decision-date" and out["backendLagSessions"] == [cal[d1 + 3]]


def test_record_cross_checks_the_backend_catalog(full, history, tmp_path):
    market, _ = full
    _, (d1,) = _grid_after(market, history, 1)
    cal = market.calendar
    in_force = set(fw.members_on(market.periods, cal[d1]))
    with pytest.raises(fw.ForwardError, match="no period"):  # a new constituent the membership file lacks
        _record(history, market, cal[d1], tmp_path / "a.jsonl", catalog=in_force | {"NEW"})
    with pytest.raises(fw.ForwardError, match="not in the backend catalog"):
        _record(history, market, cal[d1], tmp_path / "b.jsonl", catalog=in_force - {"S02"})
    out = _record(history, market, cal[d1], tmp_path / "c.jsonl", catalog=in_force | {"S21"})  # former member
    assert out["status"] == "recorded"
    assert fw.load_ledger(tmp_path / "c.jsonl")[0]["data"]["catalogNotInForce"] == ["S21"]


def test_expected_session_and_warnings():
    hol = frozenset({"2026-11-23"})
    utc = lambda *a: datetime(*a, tzinfo=timezone.utc)  # noqa: E731
    assert fw.expected_session(utc(2026, 11, 20, 9, 30), hol) == "2026-11-20"  # Fri 16:30 ICT
    assert fw.expected_session(utc(2026, 11, 20, 8, 30), hol) == "2026-11-19"  # Fri 15:30 ICT
    assert fw.expected_session(utc(2026, 11, 23, 10, 0), hol) == "2026-11-20"  # holiday Monday evening
    assert fw.expected_session(utc(2026, 11, 24, 1, 30), hol) == "2026-11-20"  # Tue 08:30 ICT
    assert fw.coverage_warnings(utc(2027, 10, 1, 0, 0), "2027-12-31") == []
    assert "56 days left" in fw.coverage_warnings(utc(2027, 11, 5, 0, 0), "2027-12-31")[0]
    periods = [("2026-08-03", frozenset({"A"}))]
    assert fw.membership_warnings(periods, "2027-01-15") == []
    assert "review is due" in fw.membership_warnings(periods, "2027-01-25")[0]


def test_record_code_provenance_covers_every_imported_module():
    prov = fw.code_provenance()
    for m in ("vn30_rotation.py", "vn30_research_data.py", "vn_execution_model.py", "trial_registry.py",
              "vn30_2_forward.py", "vn30_2_run.py", "vn30_panel_eval.py", "vn_portfolio_backtest.py"):
        assert len(prov["modules"][m]) == 64
    assert prov["gitHead"] is None or re.fullmatch(r"[0-9a-f]{40}", prov["gitHead"])
    assert prov["gitHead"] is None or isinstance(prov["modulesDifferFromHead"], list)
    broken = fw.code_provenance(fw.REPO / "no-such-dir")
    assert broken["gitHead"] is None and "gitError" in broken


# ------------------------------------------------------------------ evaluate


def _evaluate(history, market, asof, ledger, **kw):
    backend = {k: kw.pop(k) for k in ("served",) if k in kw}
    return fw.evaluate(history, fake_backend(market, asof, **backend), now_utc=_evening(asof), periods=market.periods,
                       non_trading=NO_HOLIDAYS, coverage_to=COVER, ledger=ledger, snapshot_sha256="s" * 64, **kw)


def test_evaluate_matured_records(full, history, tmp_path):
    market, _ = full
    cal = market.calendar
    _, (d1, d2, d3) = _grid_after(market, history, 3)
    ledger = tmp_path / "l.jsonl"
    for d in (d1, d2, d3):
        assert _record(history, market, cal[d], ledger)["status"] == "recorded"
    asof = cal[d2 + 21]  # d1 and d2 matured, d3 not
    out = _evaluate(history, market, asof, ledger)
    assert out["records"] == 3 and out["scored"] == 2 and out["immature"] == 1 and out["nonOverlapping"] == 2
    assert out["periodsWithMissingMembers"] == 0 and out["missingMemberPeriods"] == []
    rec = fw.load_ledger(ledger)
    p1 = out["periods"][0]
    # rank IC: manual check of the first period
    entry, exit_ = cal[d1 + 1], cal[d1 + 21]
    rets = {s: market.bars[s][exit_][0] / market.bars[s][entry][0] - 1 for s in rec[0]["inForce"]}
    bench = np.mean(list(rets.values()))
    sc = rec[0]["models"]["M-ridge"]["scores"]
    names = [s for s in rec[0]["eligible"] if s in rets]
    assert p1["models"]["M-ridge"]["rankIc"] == pytest.approx(
        ev.spearman([sc[s] for s in names], [rets[s] - bench for s in names]))
    # portfolio = the protocol backtester driven by the recorded scores, liquidated at the open of d2+21
    rules, ew_rules = run.portfolio_rules(run.PROTOCOL_PATH)
    for m, r in (("M-ridge", rules), ("B-ew", ew_rules)):
        lookup = {x["decisionDate"]: x["models"][m]["scores"] for x in rec[:2]}
        res = fw.run_backtest(market.calendar, market.bars, market.periods, lambda day, view: lookup.get(day, {}), r,
                              first_decision=d1, end=cal[d2 + 21])
        opens = {x["decisionDate"]: x["navAtOpen"] for x in res.rebalances}
        assert out["periods"][0]["models"][m]["net"] == pytest.approx(opens[cal[d2]] / opens[cal[d1]] - 1)
        assert out["periods"][1]["models"][m]["net"] == pytest.approx(res.daily[-1]["nav"] / opens[cal[d2]] - 1)
        assert res.daily[-1]["positions"] == 0  # sold at the open of d2+21
        targets = {x["decisionDate"]: x["targets"] for x in res.rebalances}
        assert targets[cal[d1]] == rec[0]["models"][m]["selection"]  # same top 10 / B-ew basket as recorded
    assert out["portfolioRules"]["model"]["topK"] == 10 and out["portfolioRules"]["ew"]["topK"] is None
    assert out["portfolioRules"]["model"]["limitUpProxy"] == 1.065
    ew = p1["models"]["B-ew"]
    assert p1["models"]["M-ridge"]["netExcess"] == pytest.approx(p1["models"]["M-ridge"]["net"] - ew["net"])
    s = out["models"]["M-ridge"]
    assert s["rankIc"]["n"] == 2 and s["periods"] == 2
    nets = [p["models"]["M-ridge"]["net"] for p in out["periods"][:2]]
    ews = [p["models"]["B-ew"]["net"] for p in out["periods"][:2]]
    cagr = lambda r: np.prod([1 + x for x in r]) ** (12.6 / len(r)) - 1  # noqa: E731
    assert s["netExcessAnnualized"] == pytest.approx(cagr(nets) - cagr(ews))
    assert out["portfolioRuns"][0]["records"] == 2 and out["portfolioRuns"][0]["end"] == cal[d2 + 21]
    with pytest.raises(fw.ForwardError, match="another snapshot"):
        fw.evaluate(history, fake_backend(market, asof), now_utc=_evening(asof), periods=market.periods,
                    non_trading=NO_HOLIDAYS, coverage_to=COVER, ledger=ledger, snapshot_sha256="t" * 64)


def test_evaluate_splits_the_portfolio_at_a_missed_decision(full, history, tmp_path):
    market, _ = full
    cal = market.calendar
    _, (d1, d2, d3) = _grid_after(market, history, 3)
    ledger = tmp_path / "l.jsonl"
    for d in (d1, d3):  # d2 missed
        assert _record(history, market, cal[d], ledger)["status"] == "recorded"
    out = _evaluate(history, market, cal[d3 + 21], ledger)
    assert [r["records"] for r in out["portfolioRuns"]] == [1, 1]
    assert out["portfolioRuns"][0]["end"] == cal[d1 + 21] and out["models"]["M-ridge"]["periods"] == 2


def test_evaluate_fails_closed_on_a_leaver_without_bars(full, history, tmp_path):
    market, _ = full
    cal = market.calendar
    _, (d1,) = _grid_after(market, history, 1)
    ledger = tmp_path / "l.jsonl"
    assert _record(history, market, cal[d1], ledger)["status"] == "recorded"
    asof = cal[d1 + 25]
    served = set(market.bars) - {"S04"}  # S04 removed from the catalog after d1: snapshot bars only
    with pytest.raises(fw.ForwardError, match="leaver-bars"):
        _evaluate(history, market, asof, ledger, served=served)
    out = _evaluate(history, market, asof, ledger, served=served, allow_missing_members=True)
    assert out["periodsWithMissingMembers"] == 1 and out["missingMemberPeriods"][0]["missing"] == ["S04"]
    # bars of the leaver from the public source (fetch_vn30_candles.py CSV) restore the period
    folder = tmp_path / "leavers"
    folder.mkdir()
    with (folder / "S04.csv").open("w", encoding="utf-8", newline="") as handle:
        handle.write("date,open,high,low,close,volume\n")
        for d, b in market.bars["S04"].items():
            if d <= asof:
                handle.write(f"{d},{b[0]!r},{b[1]!r},{b[2]!r},{b[3]!r},{b[4]!r}\n")
    leavers = fw.load_bars_dir(folder, ["S04", "S05"])
    assert set(leavers) == {"S04"}
    out = _evaluate(history, market, asof, ledger, served=served, leaver_bars=leavers)
    assert out["periodsWithMissingMembers"] == 0 and out["membersFromLeaverBars"] == ["S04"]
    full_out = _evaluate(history, market, asof, ledger)
    assert out["models"]["M-ridge"]["rankIc"] == full_out["models"]["M-ridge"]["rankIc"]


def test_dev_ic_reports_why_it_is_empty(tmp_path):
    assert fw.dev_ic_status(None) == ({}, "no results.json path")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"development": {"window": {}}}), encoding="utf-8")
    ic, why = fw.dev_ic_status(bad)
    assert ic == {} and "summary.mean not found" in why
    bad.write_text("{", encoding="utf-8")
    assert "not JSON" in fw.dev_ic_status(bad)[1]
    assert "cannot read" in fw.dev_ic_status(tmp_path / "missing.json")[1]


def test_summarize_empty_and_dev_gate_info(tmp_path):
    assert fw.summarize([])["nonOverlapping"] == 0
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"development": {"window": {"ic": {"M-ridge": {"summary": {"mean": 0.05}}}}}}),
                       encoding="utf-8")
    assert fw.dev_ic(results) == {"M-ridge": 0.05}
    assert fw.dev_ic(tmp_path / "missing.json") == {}


# ------------------------------------------------------------------ protocol / CLI


def test_check_protocol_against_the_registry(tmp_path):
    path = tmp_path / "registry.jsonl"
    tr.append({"id": run.REGISTRY_ENTRY_ID, "registeredAt": "2026-11-06", "market": "HOSE",
               "family": "hose-vn30-cross-section", "status": "planned", "trials": 8, "trialsExact": True,
               "description": "d", "source": "s", "specSha256": ev.protocol_sha256()}, path)
    info = fw.check_protocol(registry_path=path)
    assert info["protocolSha256"] == ev.protocol_sha256() and info["costs"]["sellTax"] == 0.001
    bad = tmp_path / "bad.jsonl"
    tr.append({"id": run.REGISTRY_ENTRY_ID, "registeredAt": "2026-11-06", "market": "HOSE", "family": "x",
               "status": "planned", "trials": 8, "trialsExact": True, "description": "d", "source": "s",
               "specSha256": "0" * 64}, bad)
    with pytest.raises(fw.ForwardError, match="not the registered"):
        fw.check_protocol(registry_path=bad)
    assert fw.check_protocol()["protocolSha256"] == ev.protocol_sha256()  # the real registry of this checkout


def test_cli_verify(tmp_path, capsys):
    assert fw.main(["verify", "--ledger", str(tmp_path / "none.jsonl")]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "records": 0, "lastDecision": None}
    (tmp_path / "bad.jsonl").write_text('{"schema": "x"}\n', encoding="utf-8")
    assert fw.main(["verify", "--ledger", str(tmp_path / "bad.jsonl")]) == 1


def test_http_fetch_retries_once_after_429(monkeypatch):
    import io
    import urllib.error

    calls, sleeps = [], []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "too many", {}, None)
        if "NOPE" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, None)
        assert req.get_header("User-agent") == fw.USER_AGENT
        return Resp(b"[]")

    monkeypatch.setattr(fw.urllib.request, "urlopen", urlopen)
    fetch = fw.http_fetch("http://127.0.0.1:5197", sleep=sleeps.append)
    assert fetch("/api/equities/MBB/candles") == []
    assert sleeps[0] == 60.0 and len(calls) == 2
    with pytest.raises(fw.NotServed):
        fetch("/api/equities/NOPE/candles")


def test_screen_task_runs_the_recorder_without_hiding_earlier_failures():
    from pathlib import Path

    text = (Path(fw.REPO) / "scripts" / "run_vn30_screen.ps1").read_bytes().decode("ascii")  # ASCII only
    assert '@("vn30_2_forward.py", "record")' in text
    assert text.index("vn30_screen_forward.py") < text.index("vn30_2_forward.py")
    assert "if ($v2Code -ne 0 -and $exitCode -eq 0) { $exitCode = 8 }" in text
    # data-not-ready (4) is left to the retry task only when that task exists
    assert '$v2Code -eq 4 -and $null -ne (Get-ScheduledTask -TaskName "BTCVn30TwoForward"' in text
    assert fw.EXIT_DATA_NOT_READY == 4 and fw.EXIT_MISSED_DEADLINE == 3
    assert text.rstrip().endswith("exit $exitCode")


def test_retry_task_scripts_run_only_the_recorder():
    from pathlib import Path

    runner = (Path(fw.REPO) / "scripts" / "run_vn30_2_forward.ps1").read_bytes().decode("ascii")
    assert '@("vn30_2_forward.py", "record")' in runner and "vn30_screen.py" not in runner
    assert "$exitCode = $proc.ExitCode" in runner and runner.rstrip().endswith("exit $exitCode")
    installer = (Path(fw.REPO) / "scripts" / "install_vn30_2_forward_task.ps1").read_bytes().decode("ascii")
    assert '$TaskName = "BTCVn30TwoForward"' in installer and '@("19:30", "08:30")' in installer
    assert "if ($LASTEXITCODE -ne 0)" in installer and "SupportsShouldProcess" in installer
