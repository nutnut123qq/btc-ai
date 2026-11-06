"""vn30_2_run: point-in-time features, panel/targets, gates, preflight and one full synthetic run (dev + holdout)."""

import hashlib
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import trial_registry as tr
import vn30_2_run as run
import vn30_panel_eval as ev
import vn_raw_prices as vr


# ------------------------------------------------------------------ synthetic market


def synthetic_market(n_symbols=26, seed=7, end="2026-10-30"):
    rng = np.random.default_rng(seed)
    cal = [d.date().isoformat() for d in pd.bdate_range("2016-01-04", end)]
    syms = [f"S{j:02d}" for j in range(n_symbols)]
    bars = {}
    for j, s in enumerate(syms):
        close = 20_000.0 * (1 + j / 40)
        rows = {}
        for k, day in enumerate(cal):
            if s == "S05" and 1500 <= k < 1504:  # a short suspension
                continue
            open_ = close * (1 + rng.normal(0, 0.004))
            close = max(1_000.0, open_ * (1 + rng.normal(0.0002, 0.018)))
            high = max(open_, close) * (1 + abs(rng.normal(0, 0.004)))
            low = min(open_, close) * (1 - abs(rng.normal(0, 0.004)))
            rows[day] = (open_, high, low, close, float(rng.integers(100_000, 1_000_000)))
        bars[s] = rows
    periods = [("2018-01-22", frozenset(syms[0:22])), ("2021-02-01", frozenset(syms[2:24])),
               ("2024-02-01", frozenset(syms[4:26]))]
    vn30 = {d: float(np.mean([bars[s][d][3] for s in syms if d in bars[s]])) / 20 for d in cal}
    return run.Market(tuple(cal), bars, periods, {"VN30": vn30})


@pytest.fixture(scope="module")
def market():
    return synthetic_market()


def _registry(tmp_path, *, sha=None, status="planned", extra=()):
    path = tmp_path / "registry.jsonl"
    tr.append({"id": "other-hose", "registeredAt": "2026-11-06", "market": "HOSE", "family": "x", "status": "run",
               "trials": 6, "trialsExact": True, "description": "d", "source": "s"}, path)
    tr.append({"id": run.REGISTRY_ENTRY_ID, "registeredAt": "2026-11-06", "market": "HOSE",
               "family": "hose-vn30-cross-section", "status": status, "trials": 8, "trialsExact": True,
               "description": "d", "source": "docs/research/protocols/vn30-2.json",
               "specSha256": sha or ev.protocol_sha256()}, path)
    for e in extra:
        tr.append(e, path)
    return path


# ------------------------------------------------------------------ features


def _series(closes, highs=None, vols=None):
    n = len(closes)
    dates = [f"2020-{1 + k // 28:02d}-{1 + k % 28:02d}" if k < 336 else f"2021-{1 + (k - 336) // 28:02d}-"
             f"{1 + (k - 336) % 28:02d}" for k in range(n)]
    return run.SymbolSeries(dates, np.asarray(highs if highs is not None else closes, dtype=float),
                            np.asarray(closes, dtype=float), np.asarray(vols if vols is not None else [1.0] * n,
                                                                        dtype=float))


def test_raw_features_known_values():
    closes = np.arange(1, 301, dtype=float)  # close[k] = k + 1
    ser = _series(closes, highs=closes * 1.01, vols=[10.0] * 300)
    f = run.raw_features(ser, ser.dates[-1])  # k = 299
    assert f["mom_12_1"] == pytest.approx(closes[278] / closes[47] - 1)
    assert f["mom_6"] == pytest.approx(300 / 174 - 1)
    assert f["mom_3"] == pytest.approx(300 / 237 - 1)
    assert f["rev_1"] == pytest.approx(-(300 / 279 - 1))
    assert f["vol_60"] == pytest.approx(-np.std(np.diff(np.log(closes[239:300])), ddof=1))
    assert f["liq_20"] == pytest.approx(math.log(np.median(closes[280:300] * 10)))
    assert f["dist_52w_high"] == pytest.approx(300 / (300 * 1.01) - 1)
    assert f["trend_ma200"] == pytest.approx(300 / np.mean(closes[100:300]) - 1)
    short = run.raw_features(ser, ser.dates[100])  # k = 100: only the short windows exist
    assert math.isnan(short["mom_12_1"]) and math.isnan(short["trend_ma200"]) and math.isfinite(short["mom_3"])
    assert all(math.isnan(v) for v in run.raw_features(ser, "2019-01-01").values())


def test_features_are_point_in_time(market):
    day = market.calendar[1200]
    later = dict(market.bars["S01"])
    for d in market.calendar[1201:]:  # a wild future must not move any feature at d
        o, h, l, c, v = later[d]
        later[d] = (o * 5, h * 5, l * 5, c * 5, v * 9)
    full = run.symbol_series({"S01": later})["S01"]
    cut = run.symbol_series(market.truncate(day).bars)["S01"]
    assert run.raw_features(full, day) == run.raw_features(cut, day)


def test_rank_normalize_bounds_ties_and_missing():
    out = run.rank_normalize([3.0, 1.0, np.nan, 2.0, 2.0])
    assert out.tolist() == pytest.approx([0.5, -0.5, 0.0, 0.0, 0.0])
    assert run.rank_normalize([np.nan, 4.0]).tolist() == [0.0, 0.0]
    assert run.rank_normalize([]).tolist() == []


def test_panel_targets_and_universe(market):
    decisions = run.decision_indices(market)
    assert decisions[0] >= 0 and all(b - a == 20 for a, b in zip(decisions, decisions[1:]))
    first_day = market.calendar[decisions[0]]
    assert first_day >= "2018-01-22"
    cut = market.truncate(market.calendar[decisions[30] + 10])
    panel = run.build_panel(cut, decisions[:31])
    last = panel[panel["session"] == decisions[30]]
    assert last["target"].isna().all()  # label (open of d+21) lies beyond the truncated market
    one = panel[panel["session"] == decisions[5]]
    in_force = run.members_on(market.periods, market.calendar[decisions[5]])
    assert set(one["symbol"]) <= in_force
    i = decisions[5]
    entry, exit_ = market.calendar[i + 1], market.calendar[i + 21]
    rets = {s: market.bars[s][exit_][0] / market.bars[s][entry][0] - 1 for s in in_force
            if entry in market.bars[s] and exit_ in market.bars[s]}
    row = one.iloc[0]
    assert row["target"] == pytest.approx(rets[row["symbol"]] - np.mean(list(rets.values())))
    for col in run.FCOLS:
        assert one[col].between(-0.5, 0.5).all()


def test_history_filter_matches_backtester(market):
    decisions = run.decision_indices(market, min_history=260)
    cut = market.truncate(market.calendar[decisions[3] + 30])
    assert run.build_panel(cut, decisions[:3], min_history=1300).empty  # nobody has 1300 bars in 2018
    panel = run.build_panel(cut, decisions[:3], min_history=260)
    assert panel.groupby("session")["symbol"].count().tolist() == [22, 22, 22]


# ------------------------------------------------------------------ models


def test_ridge_learns_a_planted_signal_and_needs_no_label_in_test():
    rng = np.random.default_rng(3)
    n = 600
    df = pd.DataFrame({c: rng.uniform(-0.5, 0.5, n) for c in run.FCOLS})
    df["target"] = 0.05 * df["f_mom_6"] + rng.normal(0, 0.01, n)
    pred = run.fit_predict_ridge(df.iloc[:400], df.iloc[400:].drop(columns=["target"]))
    assert np.corrcoef(pred, df["f_mom_6"].iloc[400:])[0, 1] > 0.9
    lg = run.fit_predict_lgbm(df.iloc[:400], df.iloc[400:].drop(columns=["target"]))
    assert len(lg) == 200 and np.corrcoef(lg, df["f_mom_6"].iloc[400:])[0, 1] > 0.5
    again = run.fit_predict_lgbm(df.iloc[:400], df.iloc[400:].drop(columns=["target"]))
    assert np.array_equal(lg, again)  # fixed seed, deterministic


def test_baselines_use_declared_signs():
    test = pd.DataFrame({c: [0.5, -0.5] for c in run.FCOLS})
    assert run.FIT_PREDICT["B-lowvol"](None, test).tolist() == [0.5, -0.5]
    assert run.FIT_PREDICT["B-composite"](None, test).tolist() == [0.5, -0.5]
    assert run.FIT_PREDICT["B-ew"](None, test).tolist() == [0.0, 0.0]


# ------------------------------------------------------------------ gates


def _ic(means, ts):
    return {m: {"summary": {"mean": means[m], "t": ts.get(m), "datesNonOverlapping": 60}} for m in means}


def test_development_gate():
    base = {"B-composite": 0.02}
    g = run.development_gate(_ic({**base, "M-ridge": 0.03, "M-lgbm": 0.05}, {"M-ridge": 2.1, "M-lgbm": 1.9}))
    assert g["winner"] == "M-ridge" and not g["models"]["M-lgbm"]["pass"]
    g = run.development_gate(_ic({**base, "M-ridge": 0.01, "M-lgbm": 0.04}, {"M-ridge": 3.0, "M-lgbm": 2.5}))
    assert g["winner"] == "M-lgbm" and not g["models"]["M-ridge"]["meanIcAboveComposite"]
    g = run.development_gate(_ic({"B-composite": -0.05, "M-ridge": -0.01, "M-lgbm": None}, {"M-ridge": -3.0}))
    assert g["winner"] is None and g["verdict"] == "no-go"
    g = run.development_gate(_ic({**base, "M-ridge": 0.06, "M-lgbm": 0.05}, {"M-ridge": 2.0, "M-lgbm": 4.0}))
    assert g["winner"] == "M-ridge"  # both pass: the higher development IC


def test_holdout_gate():
    assert run.holdout_gate(None, {})["verdict"] == "no-go"
    win = {"ic": {"M-ridge": {"summary": {"mean": 0.02}}},
           "portfolio": {"metrics": {"M-ridge": {"vsEqualWeight": {"netExcessAnnualized": 0.01}}}},
           "multipleTesting": {"gates": {"dsr": 0.97, "dsrPass": True, "pbo": 0.1, "pboPass": True}}}
    assert run.holdout_gate("M-ridge", win)["pass"]
    win["multipleTesting"]["gates"].update(dsr=0.5, dsrPass=False)
    g = run.holdout_gate("M-ridge", win)
    assert not g["pass"] and g["verdict"] == "no-go" and g["meanIcPositive"]


# ------------------------------------------------------------------ preflight / registry


def test_check_protocol_against_registry(tmp_path):
    good = _registry(tmp_path / "a")
    out = run.check_protocol(registry_path=good)
    assert out["protocolSha256"] == ev.protocol_sha256() and out["registryEntry"] == "hose-vn30-2"
    with pytest.raises(run.RunRefused, match="specSha256"):
        run.check_protocol(registry_path=_registry(tmp_path / "b", sha="0" * 64))
    superseded = _registry(tmp_path / "c", extra=[{
        "id": "hose-vn30-2-run-x", "registeredAt": "2026-11-06", "market": "HOSE", "family": "f", "status": "run",
        "trials": 8, "trialsExact": True, "description": "d", "source": "docs/research/protocols/vn30-2.json",
        "supersedes": "hose-vn30-2", "specSha256": ev.protocol_sha256()}])
    with pytest.raises(run.RunRefused, match="not active"):
        run.check_protocol(registry_path=superseded)


def test_raw_sensitivity_registration_is_planned_once(tmp_path):
    path = _registry(tmp_path)
    assert not run.raw_trial_planned(path)
    e = run.register_raw_sensitivity(registry_path=path, today="2026-11-06")
    assert e["status"] == "planned" and e["trials"] == 8 and run.raw_trial_planned(path)
    assert ev.registry_trial_count(path)["trials"] == 6 + 8 + 8
    # the sensitivity is not mistaken for the protocol's own entry (source is the runner, not vn30-2.json)
    assert ev.protocol_registry_entry(tr.load(path))["id"] == "hose-vn30-2"
    with pytest.raises(run.RunRefused, match="already"):
        run.register_raw_sensitivity(registry_path=path)


def test_preflight_refuses_raw_without_registration(tmp_path):
    cfg = run.RunConfig(registry_path=_registry(tmp_path), raw_dir=tmp_path / "raw", evidence_root=tmp_path / "ev")
    with pytest.raises(run.RunRefused, match="register-raw-sensitivity"):
        run.preflight(cfg)


def test_dry_run_refuses_the_evidence_root(tmp_path, market):
    cfg = run.RunConfig(registry_path=_registry(tmp_path), raw_dir=None, evidence_root=tmp_path / "ev")
    with pytest.raises(run.RunRefused, match="evidence root"):
        run.dry_run(cfg, tmp_path / "ev" / "x", market=(market, "synthetic"))


# ------------------------------------------------------------------ raw limits


def _raw_record(day, o, h, l, c, ref):
    d = datetime.fromisoformat(day)
    change = (c - ref) / 1000
    return {"Ngay": d.strftime("%d/%m/%Y"), "GiaMoCua": o / 1000, "GiaCaoNhat": h / 1000, "GiaThapNhat": l / 1000,
            "GiaDongCua": c / 1000, "GiaDieuChinh": c / 1000, "ThayDoi": f"{change:.2f} (0%)",
            "KhoiLuongKhopLenh": 1000, "GiaTriKhopLenh": 1.0, "KLThoaThuan": 0, "GtThoaThuan": 0}


def test_load_raw_limits_verifies_hashes_and_builds_bands(tmp_path):
    cal = ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05")
    bars = {"AAA": {d: (20_000.0, 21_000.0, 19_500.0, 20_000.0, 1.0) for d in cal}}
    m = run.Market(cal, bars, [("2018-01-22", frozenset({"AAA"}))])
    recs = [_raw_record("2024-01-02", 20000, 20000, 20000, 20000, 20000),
            _raw_record("2024-01-03", 21400, 21400, 21400, 21400, 20000),  # opened at the ceiling
            _raw_record("2024-01-04", 0, 21400, 21000, 21000, 21400)]  # no opening print -> proxy
    raw = tmp_path / "raw"
    raw.mkdir()
    p = vr.write_symbol_file(raw, "AAA", recs, "2016-01-01", "2026-11-06")
    (raw / "manifest.json").write_text(json.dumps({"symbols": {"AAA": {"sha256": hashlib.sha256(
        p.read_bytes()).hexdigest()}}}), encoding="utf-8")
    limits, info = run.load_raw_limits(raw, m, ["AAA"], "2024-01-03", "2024-01-05")
    assert limits["AAA"]["2024-01-03"] == (21400.0, 18600.0, 21400.0)
    assert "2024-01-04" not in limits["AAA"] and info["sessionsFallingBackToProxy"]["noOpenOrOutsideRange"] == 1
    assert info["sessionsFallingBackToProxy"]["noRawBar"] == 1  # 2024-01-05 missing at the provider
    p.chmod(0o666)
    p.write_bytes(p.read_bytes() + b" ")
    _, bad = run._raw_limits_for(raw, m, "2024-01-03", "2024-01-05")
    assert bad["used"] is False and "sha256" in bad["reason"]


# ------------------------------------------------------------------ full synthetic run


def test_full_run_dev_then_holdout_once(tmp_path, market):
    reg = _registry(tmp_path)
    root = tmp_path / "evidence" / "vn30-2"
    cfg = run.RunConfig(registry_path=reg, raw_dir=None, evidence_root=root)
    now = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
    out = run.official_run(cfg, run_id="test-run", now=now, market=(market, "synthetic"))
    run_dir = root / "test-run"
    for name in ("dev-results.json", "results.json", "report.md", "daily-nav.csv", ev.LOCK_NAME):
        assert (run_dir / name).exists(), name
    res = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert res["schema"] == run.RESULTS_SCHEMA
    assert res["protocol"]["sha256"] == ev.protocol_sha256()
    assert res["protocol"]["evaluatorSha256"] == ev.evaluator_sha256()
    dev_doc = json.loads((run_dir / "dev-results.json").read_text(encoding="utf-8"))
    assert dev_doc["evaluatorSha256"] == ev.evaluator_sha256()
    assert res["holdout"]["window"]["multipleTesting"]["provenance"]["evaluatorSha256"] == ev.evaluator_sha256()
    assert hashlib.sha256((run_dir / "results.json").read_bytes()).hexdigest() == out["resultsSha256"]
    assert res["development"]["resultsFileSha256"] == hashlib.sha256(
        (run_dir / "dev-results.json").read_bytes()).hexdigest()
    dev, hold = res["development"]["window"], res["holdout"]["window"]
    # windows never read past their end; the post-holdout regime is never read
    assert res["development"]["market"]["lastSession"] <= "2023-12-31"
    assert dev["portfolio"]["end"] <= "2023-12-31" and dev["portfolio"]["firstDecision"] >= "2019-01-01"
    assert res["holdout"]["market"]["lastSession"] <= "2026-09-18" and hold["portfolio"]["end"] <= "2026-09-18"
    assert hold["portfolio"]["firstDecision"] >= "2024-01-01"
    assert [f["year"] for f in dev["folds"]] == [2019, 2020, 2021, 2022, 2023]
    assert [f["year"] for f in hold["folds"]] == [2024, 2025, 2026]
    for m in run.RANKED:
        assert dev["ic"][m]["summary"]["datesNonOverlapping"] == dev["ic"]["B-composite"]["summary"][
            "datesNonOverlapping"] > 50
    assert set(dev["portfolio"]["metrics"]) == set(run.MODELS)
    assert hold["multipleTesting"]["dsr"]["nTrials"] == 14  # 6 + 8 run/planned HOSE trials in this registry
    assert hold["multipleTesting"]["pbo"]["strategies"] == 7 and hold["multipleTesting"]["pbo"]["blocks"] == 16
    gate = res["holdout"]["gate"]
    assert gate["applied"] == (res["development"]["gate"]["winner"] is not None)
    assert res["verdict"].startswith("no-go") or res["verdict"].startswith("pass")
    # the lock record is the one stored with the results; the registry entry supersedes the planned one, status run
    lock = json.loads((run_dir / ev.LOCK_NAME).read_text(encoding="utf-8"))
    assert lock == res["holdout"]["lock"] and lock["models"] == list(run.MODELS)
    registry = tr.load(reg)
    last = registry.entries[-1]
    assert (last["id"], last["status"], last["supersedes"], last["trials"]) == (
        "hose-vn30-2-run-test-run", "run", "hose-vn30-2", 8)
    assert last["resultsSha256"] == out["resultsSha256"] and last["specSha256"] == ev.protocol_sha256()
    assert last["result"].startswith("verdict:")
    assert ev.protocol_registry_entry(registry)["id"] == "hose-vn30-2-run-test-run"
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Development gate" in report and "semi-sealed" in report
    # read once: a second run is refused (registry entry no longer planned, lock taken)
    with pytest.raises(run.RunRefused):
        run.official_run(cfg, run_id="again", now=now, market=(market, "synthetic"))
    reg2 = _registry(tmp_path / "fresh")
    with pytest.raises(run.RunRefused, match="already opened"):
        run.official_run(run.RunConfig(registry_path=reg2, raw_dir=None, evidence_root=root), run_id="again",
                         now=now, market=(market, "synthetic"))


def test_ic_block_scores_only_guarded_fold_rows(market):
    dev = market.truncate("2023-12-31")
    decisions = run.decision_indices(dev)
    panel = run.build_panel(dev, decisions)
    folds = run.window_folds(panel, dev, "2018-01-22", "2023-12-31")
    scores = run.score_models(panel, folds)
    ic = run.ic_block(panel, scores, folds)
    tested = {s for f in folds for s in f.test_sessions}
    overrun = {s for f in folds for s in f.dropped_label_overrun}
    assert overrun and not overrun & tested
    for m in run.RANKED:  # baselines and fitted models alike: only fold test dates
        assert {r["session"] for r in ic[m]["byDate"]} <= tested
    # a full-panel baseline column passed as scores still only reaches the fold test rows
    raw = {m: panel[f"f_{run.BASELINE_FEATURE.get(m, 'mom_6')}"] for m in run.RANKED}
    raw_ic = run.ic_block(panel, raw, folds)
    assert all({r["session"] for r in raw_ic[m]["byDate"]} <= tested for m in run.RANKED)
    assert raw_ic["B-mom_6"]["summary"] == ic["B-mom_6"]["summary"]
    hand = [ev.Fold(f.year, f.test_sessions, f.train_sessions, f.purged_sessions) for f in folds]
    with pytest.raises(ev.UnguardedFold):
        run.ic_block(panel, scores, hand)
    with pytest.raises(ev.UnguardedFold):
        run.score_models(panel, hand)
    with pytest.raises(ev.HoldoutSealed):  # the holdout window needs the acquired lock
        run.window_folds(run.build_panel(market.truncate("2026-09-18"), run.decision_indices(
            market.truncate("2026-09-18"))), market.truncate("2026-09-18"), "2024-01-01", "2026-09-18")


def test_dry_run_exercises_holdout_code_inside_development(tmp_path, market):
    cfg = run.RunConfig(registry_path=_registry(tmp_path), raw_dir=None, evidence_root=tmp_path / "ev")
    doc = run.dry_run(cfg, tmp_path / "dry", market=(market, "synthetic"))
    assert doc["pseudoHoldout"]["window"]["portfolio"]["end"] <= "2023-12-31"
    assert [f["year"] for f in doc["pseudoHoldout"]["window"]["folds"]] == [2022, 2023]
    assert not (tmp_path / "ev").exists()  # no lock, nothing in the evidence root
    assert len(tr.load(cfg.registry_path).entries) == 2  # no registry write
