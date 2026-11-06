"""vn30_panel_eval: purged walk-forward, rank IC, DSR with the registry count, PBO via CSCV, holdout lock."""

import dataclasses
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import trial_registry as tr
import vn30_panel_eval as ev
import vn30_rotation as rot


# ------------------------------------------------------------------ protocol


def test_constants_match_the_frozen_protocol():
    p = ev.protocol_parameters()
    assert p["id"] == ev.PROTOCOL_ID
    assert (p["purgeSessions"], p["embargoSessions"], p["rebalanceEverySessions"]) == (
        ev.PURGE_SESSIONS, ev.EMBARGO_SESSIONS, ev.REBALANCE_SESSIONS)
    assert len(p["rankedModels"]) == 7 and "B-ew" not in p["rankedModels"]
    assert "16 blocks" in json.loads(ev.PROTOCOL_PATH.read_text(encoding="utf-8"))["metrics"]["multipleTesting"]
    assert ev.PBO_BLOCKS == 16


# ------------------------------------------------------------------ Spearman / IC


def test_spearman_known_answers():
    # 1 - 6 * sum(d^2) / (n (n^2 - 1)) with d = (-1, 1, -1, 1, 0): 1 - 24 / 120 = 0.8
    assert ev.spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) == pytest.approx(0.8, abs=1e-12)
    assert ev.spearman([1, 2, 3, 4], [10, 9, 8, 7]) == pytest.approx(-1.0)
    assert math.isnan(ev.spearman([1, 2, 3], [5, 5, 5]))
    assert math.isnan(ev.spearman([1, 2], [1, 2]))
    # ties use average ranks: x ranks (1, 2.5, 2.5, 4), y ranks (1, 2, 3, 4) -> 4.5 / sqrt(4.5 * 5)
    assert ev.spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(4.5 / math.sqrt(4.5 * 5.0), abs=1e-12)
    # NaN pairs are dropped
    assert ev.spearman([1, np.nan, 2, 3, 4], [1, 7, 2, 3, 4]) == pytest.approx(1.0)


def test_spearman_matches_scipy_with_ties():
    rng = np.random.default_rng(1)
    for _ in range(50):
        x = rng.integers(0, 6, size=30).astype(float)
        y = rng.normal(size=30) + 0.3 * x
        y[:5] = y[0]
        assert ev.spearman(x, y) == pytest.approx(stats.spearmanr(x, y).statistic, abs=1e-12)


def _panel(n_dates=40, n_names=30, spacing=20, ic_signal=0.0, seed=0, start_year=2018):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_dates):
        session = 300 + spacing * k
        day = pd.Timestamp(f"{start_year}-01-22") + pd.Timedelta(days=int(session * 365 / 250) - 420)
        score = rng.normal(size=n_names)
        target = ic_signal * score + rng.normal(size=n_names)
        for i in range(n_names):
            rows.append({"session": session, "date": day.date().isoformat(), "symbol": f"S{i:02d}",
                         "target": target[i], "score": score[i]})
    return pd.DataFrame(rows)


def test_rank_ic_by_date_and_min_names():
    panel = _panel(n_dates=3)
    panel.loc[panel["session"] == panel["session"].max(), "score"] = np.nan
    panel.loc[panel.index[-25:], "score"] = 1.0  # 25 of 30 present on the last date, all tied -> NaN
    table = ev.rank_ic_by_date(panel, panel["score"], min_names=10)
    assert list(table["n"]) == [30, 30, 25]
    g = panel[panel["session"] == table["session"][0]]
    assert table["ic"][0] == pytest.approx(stats.spearmanr(g["score"], g["target"]).statistic)
    assert math.isnan(table["ic"][2])
    few = ev.rank_ic_by_date(panel, panel["score"], min_names=31)
    assert few["ic"].isna().all()


def test_rank_ic_aligns_a_score_series_by_index():
    panel = _panel(n_dates=4, ic_signal=0.5, seed=8)
    ref = ev.rank_ic_by_date(panel, panel["score"])
    shuffled = panel["score"].sample(frac=1.0, random_state=1)
    assert np.allclose(ev.rank_ic_by_date(panel, shuffled)["ic"], ref["ic"])
    assert not np.allclose(ev.rank_ic_by_date(panel, shuffled.to_numpy())["ic"], ref["ic"])
    sub = panel[panel["symbol"] != "S00"].iloc[::-1]
    sub_ref = ev.rank_ic_by_date(sub, sub["score"].to_numpy())
    assert np.allclose(ev.rank_ic_by_date(sub, panel["score"])["ic"], sub_ref["ic"])
    with pytest.raises(ValueError, match="no score entry"):
        ev.rank_ic_by_date(panel, panel["score"].iloc[1:])
    with pytest.raises(ValueError, match="scores for"):
        ev.rank_ic_by_date(panel, panel["score"].to_numpy()[1:])


def test_non_overlapping_greedy_spacing():
    keep = ev.non_overlapping([0, 10, 20, 25, 45, 64, 65, 90], horizon=20)
    assert list(np.flatnonzero(keep)) == [0, 2, 4, 6, 7]
    with pytest.raises(ValueError):
        ev.non_overlapping([0, 0, 20])


def test_ic_summary_uses_only_non_overlapping_dates():
    panel = _panel(n_dates=60, spacing=10, ic_signal=0.5, seed=3)
    table = ev.rank_ic_by_date(panel, panel["score"])
    out = ev.ic_summary(table, samples=2000)
    assert (out["datesValid"], out["datesNonOverlapping"]) == (60, 30)
    chosen = table["ic"].to_numpy()[::2]
    assert out["mean"] == pytest.approx(chosen.mean())
    assert out["t"] == pytest.approx(chosen.mean() / (chosen.std(ddof=1) / math.sqrt(30)))
    assert out["mean"] > 0.2 and out["bootstrap"]["ci"][0] > 0 and out["bootstrap"]["pValueOneSided"] < 0.01


def test_block_bootstrap_is_seeded_and_brackets_the_mean():
    v = np.random.default_rng(5).normal(0.01, 0.05, size=80)
    a = ev.block_bootstrap_mean(v, block_size=4, samples=3000, seed=7)
    b = ev.block_bootstrap_mean(v, block_size=4, samples=3000, seed=7)
    assert a == b
    assert a["ci"][0] < v.mean() < a["ci"][1]
    # iid se ~ sd / sqrt(n)
    plain = ev.block_bootstrap_mean(v, block_size=1, samples=20000, seed=7)
    assert plain["se"] == pytest.approx(v.std(ddof=0) / math.sqrt(80), rel=0.05)
    const = ev.block_bootstrap_mean(np.full(10, 0.02), samples=100)
    assert const["ci"] == pytest.approx([0.02, 0.02])
    assert ev.block_bootstrap_mean([0.1])["ci"] is None


def test_paired_ic_difference_on_common_dates():
    a = pd.DataFrame({"session": [0, 20, 40, 60], "date": ["a", "b", "c", "d"], "n": 30, "ic": [0.1, 0.2, np.nan, 0.3]})
    b = pd.DataFrame({"session": [20, 40, 60, 80], "date": ["b", "c", "d", "e"], "n": 30, "ic": [0.0, 0.1, 0.1, 0.5]})
    out = ev.paired_ic_difference(a, b, samples=200)
    assert out["datesNonOverlapping"] == 2 and out["mean"] == pytest.approx(0.2)


# ------------------------------------------------------------------ purge / embargo / walk-forward


def test_purge_embargo_mask_index_logic():
    train = np.arange(0, 200, 10)
    keep = ev.purge_embargo_mask(train, 100, 120, purge=21, embargo=20)
    # before: s + 41 < 100 -> s <= 58; after: s > 120 + 41 -> s >= 162
    assert list(train[keep]) == [0, 10, 20, 30, 40, 50, 170, 180, 190]
    assert list(train[ev.purge_embargo_mask(train, 100, 120, purge=21, embargo=0)]) == [0, 10, 20, 30, 40, 50, 60,
                                                                                      70, 150, 160, 170, 180, 190]
    # boundary: label end + embargo exactly at the test start is dropped
    assert not ev.purge_embargo_mask([59], 100, 100, purge=21, embargo=20)[0]
    assert ev.purge_embargo_mask([58], 100, 100, purge=21, embargo=20)[0]
    with pytest.raises(ValueError):
        ev.purge_embargo_mask([0], 10, 5)


def _calendar(start="2018-01-22", n=3200):
    return [d.date().isoformat() for d in pd.bdate_range(start, periods=n)]


CAL = _calendar()  # HOSE-like session calendar from the protocol history start; session = position in it


def _decisions(start_session=0, n=100, spacing=20, calendar=CAL):
    sessions = [start_session + spacing * k for k in range(n)]
    return sessions, [calendar[s] for s in sessions]


def _last(calendar, day):
    return max(i for i, d in enumerate(calendar) if d <= day)


def _rows(sessions, dates, names=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s, d in zip(sessions, dates):
        target = rng.normal(size=names)
        for i in range(names):
            rows.append({"session": s, "date": d, "symbol": f"S{i:02d}", "target": target[i],
                         "f": target[i] + 0.1 * rng.normal(), "g": float(i)})
    return pd.DataFrame(rows)


def test_yearly_folds_expand_and_purge_the_last_two_decisions():
    sessions, dates = _decisions()
    folds = ev.yearly_folds(sessions, dates, test_from="2018-01-22", test_to="2023-12-31", calendar=CAL)
    assert [f.year for f in folds] == list(range(2019, 2024))  # 2018 has no earlier decision to train on
    for prev, fold in zip(folds, folds[1:]):
        assert set(prev.train_sessions) < set(fold.train_sessions)  # expanding
    for fold in folds:
        assert max(fold.train_sessions) + 21 + 20 < fold.fit_session
        assert fold.purged_sessions == (fold.fit_session - 40, fold.fit_session - 20)
        assert all(dates[sessions.index(s)][:4] == str(fold.year) for s in fold.test_sessions)
        assert fold.kind == "walk-forward" and fold.window == ("2018-01-22", "2023-12-31")
    tested = [s for f in folds for s in f.test_sessions]
    assert len(tested) == len(set(tested))
    assert dates[sessions.index(tested[0])] >= "2018-01-22" and dates[sessions.index(tested[-1])] <= "2023-12-31"


def test_yearly_folds_drop_test_labels_past_the_window_and_skip_thin_years():
    sessions, dates = _decisions()
    last = _last(CAL, "2023-12-31")
    folds = ev.yearly_folds(sessions, dates, test_from="2018-01-22", test_to="2023-12-31", calendar=CAL)
    overrun = tuple(s for s, d in zip(sessions, dates) if d <= "2023-12-31" and s + 21 > last)
    assert overrun and folds[-1].dropped_label_overrun == overrun
    assert folds[-1].last_session == last and max(folds[-1].test_sessions) + 21 <= last
    full = ev.yearly_folds(sessions, dates, test_from="2018-01-01", test_to="2020-12-31", calendar=CAL)
    k = len(full[0].train_sessions) + 1
    thin = ev.yearly_folds(sessions, dates, test_from="2018-01-01", test_to="2020-12-31", calendar=CAL,
                           min_train_dates=k)
    assert [f.year for f in thin] == [f.year for f in full if len(f.train_sessions) >= k] == [2020]
    with pytest.raises(ValueError):
        ev.yearly_folds([0, 20, 10], [CAL[0], CAL[20], CAL[10]], test_from="2018", test_to="2019", calendar=CAL)


def test_yearly_folds_take_the_calendar_and_check_the_last_session():
    sessions, dates = _decisions()
    with pytest.raises(TypeError):  # the session calendar is required
        ev.yearly_folds(sessions, dates, test_from="2018-01-22", test_to="2023-12-31")
    last = _last(CAL, "2023-12-31")
    assert CAL[last] <= "2023-12-31" < CAL[last + 1]
    folds = ev.yearly_folds(sessions, dates, test_from="2018-01-22", test_to="2023-12-31", calendar=CAL,
                            last_session_in_window=last)  # a caller's value must equal the computed one
    assert all(f.last_session == last for f in folds)
    for wrong in (last - 1, last + 1, sessions[-1]):
        with pytest.raises(ValueError, match="not the last session"):
            ev.yearly_folds(sessions, dates, test_from="2018-01-22", test_to="2023-12-31", calendar=CAL,
                            last_session_in_window=wrong)
    with pytest.raises(TypeError):
        ev.yearly_folds(sessions, dates, test_from="2018-01-22", test_to="2023-12-31", calendar=CAL,
                        last_session_in_window=float(last))
    # decision dates must be the calendar dates of their sessions (a shifted calendar is refused)
    with pytest.raises(ValueError, match="calendar dates"):
        ev.yearly_folds(sessions, dates, test_from="2018-01-22", test_to="2023-12-31", calendar=CAL[1:])
    with pytest.raises(ValueError, match="chronological"):
        ev.yearly_folds(sessions, dates, test_from="2018-01-22", test_to="2023-12-31", calendar=CAL[::-1])
    # boundary: a label ending exactly at the last session of the window is kept, one session later is dropped
    edge = [last - 301, last - 281, last - 61, last - 41, last - 21, last - 20]
    edge_dates = [CAL[s] for s in edge]
    (fold,) = [f for f in ev.yearly_folds(edge, edge_dates, test_from="2023-01-01", test_to="2023-12-31",
                                          calendar=CAL) if f.year == 2023]
    assert last - 21 in fold.test_sessions and fold.dropped_label_overrun == (last - 20,)
    # a calendar that ends before test_to: its last session bounds the labels
    short = CAL[:last - 10]
    cut = [s for s in edge if s < len(short)]
    folds = ev.yearly_folds(cut, [CAL[s] for s in cut], test_from="2023-01-01", test_to="2023-12-31",
                            calendar=short)
    assert all(s + 21 <= len(short) - 1 for f in folds for s in f.test_sessions)
    calendar = [(pd.Timestamp("2023-12-25") + pd.Timedelta(days=k)).date().isoformat() for k in range(14)]
    assert ev.last_session_on_or_before(calendar, "2023-12-31") == 6
    assert ev.last_session_on_or_before(calendar, "2024-01-01T") == 7
    with pytest.raises(ValueError):
        ev.last_session_on_or_before(calendar, "2023-01-01")


def test_yearly_folds_refuse_decisions_before_the_history_start():
    assert ev.history_start() == "2018-01-22"
    early = _calendar("2017-01-02")
    sessions, dates = _decisions(n=80, calendar=early)
    with pytest.raises(ValueError, match="history start"):
        ev.yearly_folds(sessions, dates, test_from="2019-01-01", test_to="2020-12-31", calendar=early)
    folds = ev.yearly_folds(sessions, dates, test_from="2019-01-01", test_to="2020-12-31", calendar=early,
                            allow_pre_history=True)
    assert folds and min(folds[0].train_dates) < "2018-01-22" and folds[0].allow_pre_history
    panel = _rows(sessions, dates)
    assert ev.walk_forward_scores(panel, folds, lambda tr_, te: te["f"].to_numpy()).notna().any()
    with pytest.raises(ValueError, match="history start"):
        ev.forward_fold(sessions[:30], early[:sessions[29] + 1])


def test_yearly_folds_refuse_the_holdout_without_an_acquired_lock(tmp_path):
    sessions, dates = _decisions(n=150)
    kwargs = dict(test_from="2024-01-01", test_to="2026-09-18", calendar=CAL)
    with pytest.raises(ev.HoldoutSealed, match="acquire HoldoutLock"):
        ev.yearly_folds(sessions, dates, **kwargs)
    with pytest.raises(ev.HoldoutSealed, match="acquire HoldoutLock"):  # a dev window overlapping by one day
        ev.yearly_folds(sessions, dates, test_from="2018-01-22", test_to="2024-01-01", calendar=CAL)
    with pytest.raises(ev.HoldoutSealed, match="forward records"):
        ev.yearly_folds(sessions, dates, test_from="2024-01-01", test_to="2026-12-31", calendar=CAL)
    root = tmp_path / "vn30-2"
    sha = ev.protocol_sha256()
    fake = {"schema": "holdout-lock/v1", "protocolId": "vn30-2", "protocolSha256": sha,
            "path": str(root / "r" / ev.LOCK_NAME)}
    with pytest.raises(ev.HoldoutSealed, match="missing"):
        ev.yearly_folds(sessions, dates, holdout_lock=fake, evidence_root=root, **kwargs)
    with pytest.raises(ev.HoldoutSealed, match="another protocol"):
        ev.yearly_folds(sessions, dates, holdout_lock={**fake, "protocolSha256": SHA}, evidence_root=root, **kwargs)
    record = ev.HoldoutLock("vn30-2", sha, root).acquire(root / "run-1", run_id="run-1", now=NOW)
    folds = ev.yearly_folds(sessions, dates, holdout_lock=record, evidence_root=root, **kwargs)
    assert [f.year for f in folds] == [2024, 2025, 2026]
    assert all(s + 21 <= _last(CAL, "2026-09-18") for f in folds for s in f.test_sessions)
    with pytest.raises(ev.HoldoutSealed, match="does not match"):
        ev.yearly_folds(sessions, dates, holdout_lock={**record, "runId": "run-9"}, evidence_root=root, **kwargs)
    # without an explicit evidence_root only the protocol evidence root of the main checkout is accepted
    with pytest.raises(ev.HoldoutSealed, match="evidence root"):
        ev.yearly_folds(sessions, dates, holdout_lock=record, **kwargs)


def test_verify_holdout_lock_refuses_a_lock_outside_the_evidence_root(tmp_path):
    sha = ev.protocol_sha256()
    foreign = tmp_path / "anywhere" / "vn30-2"
    record = ev.HoldoutLock("vn30-2", sha, foreign).acquire(foreign / "run-1", run_id="run-1", now=NOW)
    ev.verify_holdout_lock(record, evidence_root=foreign)  # accepted where it was taken, when named explicitly
    with pytest.raises(ev.HoldoutSealed, match="not a run record under the protocol evidence root"):
        ev.verify_holdout_lock(record)  # default root = docs/research/evidence/vn30-2 of the main checkout
    with pytest.raises(ev.HoldoutSealed, match="evidence root"):
        ev.verify_holdout_lock(record, evidence_root=tmp_path / "other")
    # a copy of a genuine record + sentinel planted in another directory is refused too
    planted = tmp_path / "planted"
    (planted / "run-1").mkdir(parents=True)
    (planted / "run-1" / ev.LOCK_NAME).write_bytes((foreign / "run-1" / ev.LOCK_NAME).read_bytes())
    (planted / ".holdout-vn30-2.lock").write_bytes((foreign / ".holdout-vn30-2.lock").read_bytes())
    moved = {**record, "path": str(planted / "run-1" / ev.LOCK_NAME)}
    with pytest.raises(ev.HoldoutSealed, match="evidence root"):
        ev.verify_holdout_lock(moved, evidence_root=foreign)
    assert ev.default_evidence_root() == ev.repo_main_checkout() / "docs" / "research" / "evidence" / "vn30-2"


def test_walk_forward_scores_never_trains_on_reachable_labels():
    sessions, dates = _decisions(n=60)
    rows = [{"session": s, "date": d, "symbol": f"S{i}", "target": float(i), "f": float(i), "g": 1.0}
            for s, d in zip(sessions, dates) for i in range(5)]
    panel = pd.DataFrame(rows)
    folds = ev.yearly_folds(sessions, dates, test_from="2018-01-01", test_to="2020-12-31", calendar=CAL)
    assert [f.year for f in folds] == [2019, 2020]
    seen = []

    def fit_predict(train, test):
        assert "target" in train.columns and "target" not in test.columns
        seen.append((int(train["session"].max()), int(test["session"].min())))
        return test["f"].to_numpy() * 2

    scores = ev.walk_forward_scores(panel, folds, fit_predict)
    assert all(tmax + 41 < tmin for tmax, tmin in seen) and len(seen) == len(folds)
    tested = panel["session"].isin([s for f in folds for s in f.test_sessions])
    assert scores[tested].notna().all() and scores[~tested].isna().all()
    assert (scores[tested] == panel.loc[tested, "f"] * 2).all()
    # hand-built or modified folds are refused, whatever they contain
    bad = ev.Fold(2019, folds[1].test_sessions, folds[1].train_sessions + (folds[1].fit_session - 20,), ())
    with pytest.raises(ev.UnguardedFold, match="not built by yearly_folds"):
        ev.walk_forward_scores(panel, [bad], fit_predict)
    copied = ev.Fold(**{k: getattr(folds[1], k) for k in folds[1].__dataclass_fields__ if k != "seal"})
    with pytest.raises(ev.UnguardedFold):
        ev.walk_forward_scores(panel, [copied], fit_predict)
    widened = dataclasses.replace(folds[1], train_sessions=folds[1].train_sessions + (folds[1].fit_session - 20,))
    with pytest.raises(ev.UnguardedFold):
        ev.walk_forward_scores(panel, [widened], fit_predict)
    with pytest.raises(ValueError, match="scored twice"):
        ev.walk_forward_scores(panel, [folds[1], folds[1]], fit_predict)
    # a panel on another calendar (dates shifted) is refused
    shifted = panel.assign(date=panel["session"].map(lambda s: CAL[s + 1]))
    with pytest.raises(ev.UnguardedFold, match="calendar dates"):
        ev.walk_forward_scores(shifted, folds, fit_predict)

    def cheat(train, test):
        return test["target"].to_numpy()

    with pytest.raises(KeyError):
        ev.walk_forward_scores(panel, folds, cheat)
    with pytest.raises(KeyError):  # every declared label column is withheld
        ev.walk_forward_scores(panel, folds, lambda tr_, te: te["f"].to_numpy(), label_columns=("target", "f"))


def test_walk_forward_scores_raises_on_a_missing_label_column_and_honours_a_feature_allowlist():
    sessions, dates = _decisions(n=60)
    panel = _rows(sessions, dates)
    folds = ev.yearly_folds(sessions, dates, test_from="2018-01-01", test_to="2020-12-31", calendar=CAL)
    with pytest.raises(ValueError, match="missing from the panel"):
        ev.walk_forward_scores(panel.drop(columns="target"), folds, lambda tr_, te: te["f"].to_numpy())
    with pytest.raises(ValueError, match="missing from the panel"):  # a misspelled label would leak the real one
        ev.walk_forward_scores(panel, folds, lambda tr_, te: te["f"].to_numpy(), label_columns=("Target",))
    handed = []

    def fp(train, test):
        handed.append(list(test.columns))
        return test["f"].to_numpy()

    ev.walk_forward_scores(panel, folds, fp, feature_columns=("f",))
    assert handed and all(cols == ["session", "date", "symbol", "f"] for cols in handed)
    with pytest.raises(ValueError, match="allowlist"):
        ev.walk_forward_scores(panel, folds, fp, feature_columns=("f", "target"))


def test_baselines_go_through_the_guarded_window_path():
    sessions, dates = _decisions()
    panel = _rows(sessions, dates, seed=4)
    last = _last(CAL, "2023-12-31")
    folds = ev.yearly_folds(sessions, dates, test_from="2018-01-22", test_to="2023-12-31", calendar=CAL)
    overrun = set(folds[-1].dropped_label_overrun)
    assert overrun and all(s + 21 > last for s in overrun)
    # a baseline (fixed rule, no fit) scored on the folds only: the dev decisions whose label ends after test_to
    # (and every holdout / later decision) stay NaN
    b = ev.baseline_scores(panel, folds, "f")
    tested = {s for f in folds for s in f.test_sessions}
    assert b[panel["session"].isin(tested)].notna().all() and b[~panel["session"].isin(tested)].isna().all()
    assert b[panel["session"].isin(overrun)].isna().all()
    with pytest.raises(ValueError, match="label column"):
        ev.baseline_scores(panel, folds, "target")
    # window_rank_ic ignores every row outside the fold test blocks, even for a full-panel feature column
    table = ev.window_rank_ic(panel, panel["f"], folds)
    assert set(table["session"]) == tested and not (set(table["session"]) & overrun)
    assert not (set(table["date"]) & {d for d in dates if d > "2023-12-31"})
    unguarded = ev.rank_ic_by_date(panel, panel["f"])
    assert overrun <= set(unguarded["session"])  # the raw table function has no window (not a protocol number)
    same = ev.window_rank_ic(panel, b, folds)
    assert np.allclose(same["ic"], table["ic"])
    assert np.allclose(ev.window_rank_ic(panel, panel["f"].to_numpy(), folds)["ic"], table["ic"])
    comp = ev.baseline_scores(panel, folds, lambda test: test["f"].to_numpy() + test["g"].to_numpy())
    assert comp[panel["session"].isin(overrun)].isna().all()
    bad = ev.Fold(2023, tuple(sorted(overrun)), (), ())
    with pytest.raises(ev.UnguardedFold):
        ev.window_rank_ic(panel, panel["f"], [bad])
    with pytest.raises(ev.UnguardedFold):
        ev.baseline_scores(panel, [bad], "f")
    with pytest.raises(ValueError, match="overlapping"):
        ev.window_rank_ic(panel, panel["f"], [folds[0], folds[0]])


def test_baseline_holdout_rows_need_a_verified_lock(tmp_path):
    sessions, dates = _decisions(n=150)
    panel = _rows(sessions, dates, seed=5)
    hold = [s for s, d in zip(sessions, dates) if "2024-01-01" <= d <= "2026-09-18"]
    # no lock: no holdout fold can be built, and a hand-built one is refused by every scoring entry point
    with pytest.raises(ev.HoldoutSealed):
        ev.yearly_folds(sessions, dates, test_from="2024-01-01", test_to="2026-09-18", calendar=CAL)
    fake = ev.Fold(2024, tuple(hold[:5]), tuple(sessions[:10]), ())
    with pytest.raises(ev.UnguardedFold):
        ev.window_rank_ic(panel, panel["f"], [fake])
    with pytest.raises(ev.UnguardedFold):
        ev.baseline_scores(panel, [fake], "f")
    # a dev fold applied to a panel whose sessions were renumbered into the holdout is refused
    dev = ev.yearly_folds(sessions, dates, test_from="2018-01-22", test_to="2023-12-31", calendar=CAL)
    moved = panel.assign(date=panel["session"].map(lambda s: CAL[min(s + 1500, len(CAL) - 1)]))
    with pytest.raises(ev.UnguardedFold, match="calendar dates"):
        ev.window_rank_ic(moved, moved["f"], dev)
    # with an acquired lock the holdout is scored; the lock is re-verified at every use
    root = tmp_path / "vn30-2"
    record = ev.HoldoutLock("vn30-2", ev.protocol_sha256(), root).acquire(root / "run-1", run_id="run-1", now=NOW)
    folds = ev.yearly_folds(sessions, dates, test_from="2024-01-01", test_to="2026-09-18", calendar=CAL,
                            holdout_lock=record, evidence_root=root)
    table = ev.window_rank_ic(panel, ev.baseline_scores(panel, folds, "f"), folds)
    assert len(table) and table["date"].between("2024-01-01", "2026-09-18").all()
    (root / ".holdout-vn30-2.lock").unlink()
    with pytest.raises(ev.HoldoutSealed, match="missing"):
        ev.window_rank_ic(panel, panel["f"], folds)
    with pytest.raises(ev.HoldoutSealed, match="missing"):
        ev.baseline_scores(panel, folds, "f")


def test_forward_fold_scores_only_a_decision_whose_label_lies_past_the_calendar():
    sessions, dates = _decisions(n=40)
    i = sessions[30]
    cut = CAL[:i + 1]
    fold = ev.forward_fold(sessions[:31], cut)
    year_first = next(s for s in sessions if CAL[s][:4] == CAL[i][:4])
    assert fold.kind == "forward" and fold.test_sessions == (i,) and fold.fit_session == year_first
    assert all(s + 41 < year_first for s in fold.train_sessions)
    ref = ev.yearly_folds(sessions[:31], dates[:31], test_from=CAL[i][:4] + "-01-01", test_to=CAL[i], calendar=cut)
    assert ref[-1].train_sessions == fold.train_sessions  # same yearly refit as the walk-forward
    panel = _rows(sessions[:31], dates[:31])
    with pytest.raises(ev.UnguardedFold, match="has a label"):  # the calendar was not cut at the decision
        ev.walk_forward_scores(panel, [fold], lambda tr_, te: te["f"].to_numpy())
    panel.loc[panel["session"] == i, "target"] = np.nan
    s = ev.walk_forward_scores(panel, [fold], lambda tr_, te: te["f"].to_numpy())
    assert s[panel["session"] == i].notna().all() and s[panel["session"] != i].isna().all()
    with pytest.raises(ev.UnguardedFold, match="walk-forward"):
        ev.window_rank_ic(panel, s, [fold])
    with pytest.raises(ValueError, match="last session of the calendar"):
        ev.forward_fold(sessions[:31], CAL[:i + 5])


def test_evaluator_provenance_records_both_hashes():
    prov = ev.evaluator_provenance()
    assert prov["protocolSha256"] == ev.protocol_sha256()
    assert prov["evaluatorSha256"] == hashlib.sha256(Path(ev.__file__).read_bytes()).hexdigest()


# ------------------------------------------------------------------ DSR


def test_dsr_reproduces_the_worked_example_of_bailey_lopez_de_prado_2014():
    # N = 100 trials, V[SR] = 1/2 annualised, best SR 2.5 annualised, T = 1250 daily returns, skew -3, kurtosis 10.
    # The paper reports SR0 ~ 0.1132 (daily) and DSR ~ 0.9004.
    sr0 = ev.expected_max_sharpe(100, 0.5 / 250)
    assert sr0 == pytest.approx(0.1132, abs=5e-5)
    assert ev.probabilistic_sharpe(2.5 / math.sqrt(250), sr0, 1250, -3.0, 10.0) == pytest.approx(0.9004, abs=5e-5)
    assert ev.expected_max_sharpe(1, 0.5) == 0.0 and ev.expected_max_sharpe(10, 0.0) == 0.0


def test_dsr_matches_the_independent_vn30_rotation_implementation():
    rng = np.random.default_rng(11)
    trials = [rng.standard_t(5, size=120) * 0.02 + 0.002 * k for k in range(7)]
    srs = [ev.sharpe_moments(t)["sharpe"] for t in trials]
    best = trials[int(np.argmax(srs))]
    out = ev.deflated_sharpe(best, n_trials=7, sr_variance=float(np.var(srs, ddof=1)))
    mean, pop_sd, skew, kurt = rot._moments(list(best))
    assert out["skew"] == pytest.approx(skew) and out["kurtosis"] == pytest.approx(kurt)
    assert out["expectedMaxSharpe"] == pytest.approx(rot.expected_max_sr(srs))
    assert out["dsr"] == pytest.approx(rot.psr(out["sharpe"], rot.expected_max_sr(srs), 120, skew, kurt))
    # more trials can only deflate further
    assert ev.deflated_sharpe(best, n_trials=500, sr_variance=out["srVariance"])["dsr"] < out["dsr"]


def _registry(tmp_path, *trials, protocol_trials=8):
    """Registry with the given (market, status, trials) entries, then the vn30-2 protocol entry (planned)."""
    path = tmp_path / "registry.jsonl"
    for i, (market, status, n) in enumerate(trials):
        tr.append({"id": f"t{i}", "registeredAt": "2026-11-06", "market": market, "family": "f", "status": status,
                   "trials": n, "trialsExact": True, "description": "d", "source": "s"}, path)
    if protocol_trials:
        tr.append({"id": "hose-vn30-2", "registeredAt": "2026-11-06", "market": "HOSE", "family": "f",
                   "status": "planned", "trials": protocol_trials, "trialsExact": True, "description": "d",
                   "source": "docs/research/protocols/vn30-2.json", "specSha256": ev.protocol_sha256()}, path)
    return path


def test_dsr_takes_n_from_the_registry_at_evaluation_time(tmp_path):
    path = _registry(tmp_path, ("HOSE", "run", 16), ("BTC", "run", 1000), ("HOSE", "abandoned", 50))
    count = ev.registry_trial_count(path)
    assert count["trials"] == 24 and count["registryHead"] == tr.load(path).head
    rng = np.random.default_rng(2)
    series = {f"m{k}": rng.normal(0.001 * k, 0.03, size=70) for k in range(7)}
    out = ev.deflated_sharpe_from_registry(series["m6"], series, registry_path=path)
    assert out["nTrials"] == 24 and out["registry"]["trials"] == 24
    assert out["registry"]["protocolEntry"] == "hose-vn30-2"
    var = ev.sr_variance(series)
    assert out["srVariance"] == var["value"] == max(var["crossTrial"], var["estimatorFloor"])
    assert out["dsr"] == pytest.approx(ev.deflated_sharpe(series["m6"], n_trials=24, sr_variance=var["value"])["dsr"])
    tr.append({"id": "later", "registeredAt": "2026-11-07", "market": "HOSE", "family": "f", "status": "planned",
               "trials": 76, "trialsExact": True, "description": "d", "source": "s"}, path)
    assert ev.deflated_sharpe_from_registry(series["m6"], series, registry_path=path)["nTrials"] == 100
    small = _registry(tmp_path / "x", ("HOSE", "run", 3), protocol_trials=5)
    with pytest.raises(ValueError, match="register every trial"):
        ev.deflated_sharpe_from_registry(series["m6"], series, registry_path=small)


def test_dsr_requires_the_protocol_entry_in_the_registry(tmp_path):
    rng = np.random.default_rng(3)
    series = {f"m{k}": rng.normal(0, 0.03, size=70) for k in range(7)}
    legacy = _registry(tmp_path / "a", ("HOSE", "run", 30), protocol_trials=0)
    with pytest.raises(ValueError, match="no planned/run HOSE registry entry for protocol vn30-2"):
        ev.deflated_sharpe_from_registry(series["m0"], series, registry_path=legacy)
    other = tmp_path / "b" / "registry.jsonl"
    tr.append({"id": "hose-vn30-2", "registeredAt": "2026-11-06", "market": "HOSE", "family": "f",
               "status": "planned", "trials": 8, "trialsExact": True, "description": "d",
               "source": "docs/research/protocols/vn30-2.json", "specSha256": "0" * 64}, other)
    with pytest.raises(ValueError, match="another version"):
        ev.deflated_sharpe_from_registry(series["m0"], series, registry_path=other)
    entry = ev.protocol_registry_entry(tr.load())  # the real registry has it (planned, or its run successor)
    assert entry["id"] == "hose-vn30-2" or entry.get("supersedes") == "hose-vn30-2"


def test_sr_variance_is_floored_for_correlated_trials():
    rng = np.random.default_rng(12)
    common = rng.normal(0, 0.03, size=40)
    correlated = {f"m{k}": common + rng.normal(0, 0.003, size=40) for k in range(7)}
    var = ev.sr_variance(correlated)
    assert var["floored"] and var["crossTrial"] < var["estimatorFloor"]
    assert var["estimatorFloor"] == pytest.approx(1 / 39, rel=0.2)
    assert var["value"] == var["estimatorFloor"] and var["estimator"] == ev.SR_VARIANCE_ESTIMATOR
    spread = {f"m{k}": rng.normal(0.01 * k, 0.03, size=40) for k in range(7)}
    assert not ev.sr_variance(spread)["floored"]
    with pytest.raises(ValueError, match="at least 2"):
        ev.sr_variance({"m0": common})
    # simulated false passes of 7 highly correlated noise strategies stay rare with the floor (N = 30, T = 33)
    hits = 0
    for seed in range(300):
        r = np.random.default_rng(1000 + seed)
        f = r.normal(0, 0.03, size=33)
        trials = {f"m{k}": 0.95 * f + math.sqrt(1 - 0.95 ** 2) * r.normal(0, 0.03, size=33) for k in range(7)}
        best = max(trials, key=lambda k: ev.sharpe_moments(trials[k])["sharpe"])
        v = ev.sr_variance(trials)["value"]
        hits += ev.deflated_sharpe(trials[best], n_trials=30, sr_variance=v)["dsr"] >= 0.95
    assert hits / 300 < 0.02


def test_real_registry_count_is_readable():
    count = ev.registry_trial_count()
    assert count["trials"] >= 1 and count["market"] == "HOSE" and count["statuses"] == ["planned", "run"]


# ------------------------------------------------------------------ PBO


def test_pbo_of_pure_noise_is_near_one_half():
    # A single noise sample gives a widely spread PBO (the 12870 splits share data); its mean over samples is 1/2
    # for an even N. With N = 7 a median rank (logit 0) has probability 1/7: pbo -> 4/7, pboStrict -> 3/7.
    even = [ev.pbo_cscv(np.random.default_rng(seed).normal(0, 0.03, size=(320, 10))) for seed in range(40)]
    assert 0.42 < float(np.mean([o["pbo"] for o in even])) < 0.62
    assert all(o["pbo"] == o["pboStrict"] for o in even)
    odd = [ev.pbo_cscv(np.random.default_rng(100 + seed).normal(0, 0.03, size=(96, 7))) for seed in range(40)]
    assert 0.45 < float(np.mean([o["pboStrict"] for o in odd])) + 1 / 14 < 0.6
    assert float(np.mean([o["pbo"] - o["pboStrict"] for o in odd])) == pytest.approx(1 / 7, abs=0.05)


def test_pbo_of_a_planted_edge_is_low():
    m = np.random.default_rng(4).normal(0, 0.03, size=(320, 7))
    m[:, 3] += 0.02
    out = ev.pbo_cscv(pd.DataFrame(m, columns=[f"m{i}" for i in range(7)]))
    assert out["pbo"] < 0.05 and out["combinations"] == 12870 and out["blocks"] == 16
    assert out["chosenInSample"]["m3"] > 12000 and out["probOosLoss"] < 0.05


def test_pbo_block_layout_and_validation():
    m = np.random.default_rng(0).normal(size=(70, 7))
    out = ev.pbo_cscv(m)
    assert (out["rowsPerBlock"], out["periodsUsed"], out["droppedOldest"]) == (4, 64, 6)
    assert 0.0 <= out["pboStrict"] <= out["pbo"] <= 1.0
    with pytest.raises(ValueError):
        ev.pbo_cscv(np.zeros((31, 3)))
    with pytest.raises(ValueError):
        ev.pbo_cscv(np.zeros((64, 1)))
    with pytest.raises(ValueError):
        ev.pbo_cscv(np.full((64, 3), np.nan))


def test_multiple_testing_report(tmp_path):
    path = _registry(tmp_path, ("HOSE", "run", 30))
    rng = np.random.default_rng(9)
    series = {f"m{k}": rng.normal(0, 0.03, size=96) for k in range(7)}
    out = ev.multiple_testing_report(series, "m2", registry_path=path)
    assert out["dsr"]["nTrials"] == 38 and out["pbo"]["strategies"] == 7
    gates = out["gates"]
    assert gates["pbo"] == out["pbo"]["gatePbo"] == out["pbo"]["pbo"] and gates["pboMax"] == 0.3
    assert gates["pboPass"] == (gates["pbo"] < 0.3) and gates["dsrPass"] == (out["dsr"]["dsr"] >= 0.95)
    assert gates["srVarianceEstimator"] == ev.SR_VARIANCE_ESTIMATOR
    assert out["provenance"] == ev.evaluator_provenance()
    with pytest.raises(ValueError):
        ev.multiple_testing_report(series, "nope", registry_path=path)
    with pytest.raises(ValueError):
        ev.multiple_testing_report({**series, "m7": series["m0"][:50]}, "m2", registry_path=path)


# ------------------------------------------------------------------ holdout lock


SHA = "a" * 64
NOW = datetime(2026, 11, 6, 10, 0, tzinfo=timezone.utc)


def test_holdout_lock_is_read_once_per_protocol(tmp_path):
    root = tmp_path / "vn30-2"
    lock = ev.HoldoutLock("vn30-2", SHA, root)
    assert not lock.locked
    record = lock.acquire(root / "run-1", run_id="run-1", registry_head="h", models=["M-ridge"], now=NOW)
    assert record["openedAtUtc"] == "2026-11-06T10:00:00+00:00"
    on_disk = json.loads((root / "run-1" / ev.LOCK_NAME).read_text(encoding="utf-8"))
    assert on_disk["protocolId"] == "vn30-2" and on_disk["protocolSha256"] == SHA and on_disk["models"] == ["M-ridge"]
    assert lock.locked
    with pytest.raises(ev.HoldoutAlreadyRead):
        lock.acquire(root / "run-1", run_id="run-1")
    with pytest.raises(ev.HoldoutAlreadyRead):
        ev.HoldoutLock("vn30-2", "b" * 64, root).acquire(root / "run-2", run_id="run-2")
    assert not (root / "run-2" / ev.LOCK_NAME).exists()


def test_holdout_lock_survives_a_deleted_run_record_and_isolates_protocols(tmp_path):
    root = tmp_path / "evidence"
    ev.HoldoutLock("vn30-2", SHA, root).acquire(root / "r1", run_id="r1")
    (root / "r1" / ev.LOCK_NAME).unlink()
    with pytest.raises(ev.HoldoutAlreadyRead):
        ev.HoldoutLock("vn30-2", SHA, root).acquire(root / "r2", run_id="r2")
    (root / ".holdout-vn30-2.lock").unlink()
    (root / "r1" / ev.LOCK_NAME).write_text(json.dumps({"protocolId": "vn30-2"}), encoding="utf-8")
    with pytest.raises(ev.HoldoutAlreadyRead):
        ev.HoldoutLock("vn30-2", SHA, root).acquire(root / "r2", run_id="r2")
    other = tmp_path / "other"
    ev.HoldoutLock("vn30-3", SHA, other).acquire(other / "r1", run_id="r1")


def test_holdout_lock_defaults_to_the_main_checkout_evidence_root(tmp_path):
    lock = ev.HoldoutLock("vn30-2", SHA)
    main = ev.repo_main_checkout()
    assert lock.evidence_root == main / "docs" / "research" / "evidence" / "vn30-2"
    assert not (main / ".git").is_file()  # never a worktree (whose .git is a file)
    repo = tmp_path / "main"
    (repo / ".git" / "worktrees" / "wt").mkdir(parents=True)
    (repo / ".git" / "worktrees" / "wt" / "commondir").write_text("../..\n", encoding="utf-8")
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {repo / '.git' / 'worktrees' / 'wt'}\n", encoding="utf-8")
    assert ev.repo_main_checkout(wt) == repo.resolve()
    assert ev.repo_main_checkout(repo) == repo


def test_holdout_lock_validates_inputs(tmp_path):
    with pytest.raises(ValueError):
        ev.HoldoutLock("../x", SHA, tmp_path)
    with pytest.raises(ValueError):
        ev.HoldoutLock("vn30-2", "nothex", tmp_path)
    lock = ev.HoldoutLock("vn30-2", SHA, tmp_path / "root")
    with pytest.raises(ValueError, match="direct child"):
        lock.acquire(tmp_path / "elsewhere", run_id="r")
    with pytest.raises(ValueError, match="run id"):
        lock.acquire(tmp_path / "root" / "r", run_id="bad id")
    assert not lock.locked
