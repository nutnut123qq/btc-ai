"""equity_base_rates (EQX P1, protocol equity-base-rates-protocol-v1).

Fixture candles only — never touches the DB or the network.
"""

import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import equity_base_rates as ebr
import vietnam_market_calendar as vn


@pytest.fixture(autouse=True)
def _explicit_fixture_calendar(monkeypatch):
    """EQX v2 A1: pure runs without a calendar fail closed, so every fixture here
    declares one explicitly via the env override (a date outside all fixture ranges)."""
    monkeypatch.setenv(vn.NON_TRADING_DATES_ENV, "2000-01-03")


FEE, TAX = 0.0015, 0.001
COSTS = ebr.Costs(FEE, FEE, TAX)


def _trading_days(start: date, count: int) -> list[date]:
    days, cursor = [], start
    while len(days) < count:
        if vn.is_trading_day(cursor):
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _row(day: date, o: float, h: float, l: float, c: float, v: float = 1_000_000.0, **extra):
    open_ms = vn.daily_open_ms(day)
    return {
        "OpenTimeMs": open_ms,
        "CloseTimeMs": vn.session_close_ms("1d", open_ms),
        "Open": o,
        "High": h,
        "Low": l,
        "Close": c,
        "Volume": v,
        "PriceBasis": "adjusted",
        **extra,
    }


def _bars(ohlc: list[tuple[float, float, float, float]]) -> ebr.Bars:
    days = _trading_days(date(2026, 8, 3), len(ohlc))
    rows = [_row(d, *bar) for d, bar in zip(days, ohlc)]
    bars, _, _ = ebr.prepare_bars(rows, cutoff_ms=10**15)
    return bars


def _net(entry: float, exit_: float) -> float:
    return exit_ * (1 - FEE - TAX) / (entry * (1 + FEE)) - 1


CFG = ebr.BarrierConfig("test-p5-s4-h3", 0.05, 0.04, 3)
FLAT = (100.0, 100.5, 99.5, 100.0)


# --- barrier labelling --------------------------------------------------------


def _label_first(ohlc):
    labels = ebr.label_barriers(_bars(ohlc), CFG, COSTS)
    return labels, 0  # signal bar 0 -> entry bar 1


def test_target_intrabar_exits_at_target():
    labels, t = _label_first([FLAT, (100, 101, 99, 100), (100.5, 106, 99.5, 104), FLAT, FLAT])
    assert labels.label[t] == ebr.TARGET
    assert labels.resolve_idx[t] == 2
    assert labels.net_return[t] == pytest.approx(_net(100, 105))
    assert not labels.gap_exit[t] and not labels.double_touch[t]


def test_stop_intrabar_exits_at_stop():
    labels, t = _label_first([FLAT, (100, 101, 95.5, 97), FLAT, FLAT, FLAT])
    assert labels.label[t] == ebr.STOP
    assert labels.resolve_idx[t] == 1  # entry bar itself
    assert labels.net_return[t] == pytest.approx(_net(100, 96))


def test_gap_down_through_stop_exits_at_open():
    labels, t = _label_first([FLAT, (100, 101, 99, 100), (93, 99, 92, 98), FLAT, FLAT])
    assert labels.label[t] == ebr.STOP and labels.gap_exit[t]
    assert labels.net_return[t] == pytest.approx(_net(100, 93))
    assert labels.net_return[t] < _net(100, 96)  # worse than the nominal stop


def test_gap_up_through_target_exits_at_target_not_open():
    labels, t = _label_first([FLAT, (100, 101, 99, 100), (107, 108, 106, 107), FLAT, FLAT])
    assert labels.label[t] == ebr.TARGET and labels.gap_exit[t]
    assert labels.net_return[t] == pytest.approx(_net(100, 105))


def test_same_bar_double_touch_is_stop_first_and_flagged():
    labels, t = _label_first([FLAT, (100, 106, 95, 101), FLAT, FLAT, FLAT])
    assert labels.label[t] == ebr.STOP and labels.double_touch[t]
    assert labels.net_return[t] == pytest.approx(_net(100, 96))


def test_timeout_exits_at_last_horizon_close():
    labels, t = _label_first([FLAT, (100, 101, 99, 100), (100, 102, 98, 101), (101, 103, 99, 102.5), FLAT])
    assert labels.label[t] == ebr.TIMEOUT
    assert labels.resolve_idx[t] == 3  # entry 1 + H - 1
    assert labels.net_return[t] == pytest.approx(_net(100, 102.5))


def test_immature_when_window_runs_past_data_without_touch():
    labels = ebr.label_barriers(_bars([FLAT, (100, 101, 99, 100), (100, 102, 98, 101)]), CFG, COSTS)
    assert labels.label[0] == -1 and labels.resolve_idx[0] == -1 and np.isnan(labels.net_return[0])
    assert labels.label[2] == -1 and labels.entry_idx[2] == -1  # last bar: no entry bar


def test_early_touch_matures_before_full_horizon():
    labels = ebr.label_barriers(_bars([FLAT, (100, 106, 99.5, 105)]), CFG, COSTS)
    assert labels.label[0] == ebr.TARGET and labels.resolve_idx[0] == 1


def test_costs_come_from_vn_execution_model():
    costs = ebr.Costs.from_execution_model()
    assert (costs.buy_fee, costs.sell_fee, costs.sell_tax) == (0.0015, 0.0015, 0.001)
    # Round trip at an unchanged price loses fees + tax.
    assert costs.net_return(100.0, 100.0) == pytest.approx(0.9975 / 1.0015 - 1)


def test_ev_is_probability_weighted_class_returns():
    labels = ebr.Labels(
        label=np.array([0, 1, 1, 2]),
        resolve_idx=np.array([1, 2, 3, 4]),
        double_touch=np.zeros(4, bool),
        gap_exit=np.zeros(4, bool),
        net_return=np.array([0.047, -0.042, -0.060, 0.010]),
        entry_idx=np.array([1, 2, 3, 4]),
    )
    r = ebr._class_returns(labels, np.arange(4), CFG, COSTS)
    assert r == pytest.approx([0.047, -0.051, 0.010])
    probs = np.array([0.5, 0.3, 0.2])
    assert float(np.dot(probs, r)) == pytest.approx(0.5 * 0.047 - 0.3 * 0.051 + 0.2 * 0.010)
    # Class missing from the eligible set -> nominal barrier fallback.
    r2 = ebr._class_returns(labels, np.array([1]), CFG, COSTS)
    assert r2[0] == pytest.approx(_net(1, 1.05)) and r2[2] == pytest.approx(_net(1, 1))


# --- synthetic random-walk series for walk-forward tests ----------------------


def _random_walk_rows(n: int, seed: int, start: date = date(2021, 1, 4)):
    rng = np.random.default_rng(seed)
    days = _trading_days(start, n)
    price = 20_000.0
    rows = []
    for i, d in enumerate(days):
        o = price * (1 + rng.normal(0, 0.004))
        c = o * (1 + rng.normal(0, 0.018))
        h = max(o, c) * (1 + abs(rng.normal(0, 0.008)))
        l = min(o, c) * (1 - abs(rng.normal(0, 0.008)))
        v = float(rng.integers(500_000, 3_000_000))
        rows.append(_row(d, o, h, l, c, v, Revision=i % 3, UpdatedAtUtc=datetime(2026, 10, 1, tzinfo=timezone.utc)
                         + timedelta(minutes=i)))
        price = c
    return rows


H20 = ebr.CONFIGS[0]


def _evaluate(rows, cfg=H20):
    bars, _, _ = ebr.prepare_bars(rows, cutoff_ms=10**15)
    X = ebr.compute_features(bars)
    return ebr.evaluate_config(bars, X, cfg, COSTS), bars, X


@pytest.fixture(scope="module")
def walk_rows():
    return _random_walk_rows(900, seed=7)


@pytest.fixture(scope="module")
def walk_result(walk_rows):
    return _evaluate(walk_rows)


def _oos_map(result):
    oos = result["_oos"]
    return {int(t): (float(pk), float(pp)) for t, pk, pp in zip(oos["t"], oos["probTargetKnn"], oos["probTargetPrior"])}


def test_walk_forward_produces_oos_and_yearly_folds(walk_result):
    result, bars, _ = walk_result
    v = result["validation"]
    assert v["nOos"] > 0
    assert v["oosStartSignalOpenTimeMs"] is not None
    years = [f["year"] for f in result["_folds"]]
    assert years == sorted(years) and len(set(years)) == len(years)
    for fold in result["_folds"]:
        assert fold["kSelection"]["k"] in ebr.K_GRID
    # OOS start: the first forecast bar has >= 500 eligible matured samples.
    first_t = int(result["_oos"]["t"][0])
    labels = ebr.label_barriers(bars, H20, COSTS)
    has = np.all(np.isfinite(ebr.compute_features(bars)), axis=1)
    assert int(np.sum(ebr.eligible_mask(labels, has, first_t))) >= ebr.MIN_MATURED_FOR_OOS


def test_leakage_appending_or_perturbing_future_bars_keeps_earlier_forecasts(walk_rows, walk_result):
    full = _oos_map(walk_result[0])
    cut = 820
    truncated = _oos_map(_evaluate(walk_rows[:cut])[0])
    perturbed_rows = walk_rows[:cut] + [
        dict(r, Open=r["Open"] * 1.3, High=r["High"] * 1.5, Low=r["Low"] * 0.6, Close=r["Close"] * 0.7)
        for r in walk_rows[cut:]
    ]
    perturbed = _oos_map(_evaluate(perturbed_rows)[0])
    assert truncated, "expected OOS forecasts before the cut"
    for t, value in truncated.items():
        assert full[t] == value
        assert perturbed[t] == value


def test_neighbors_resolved_strictly_before_forecast_bar(walk_result):
    result, bars, X = walk_result
    labels = ebr.label_barriers(bars, H20, COSTS)
    has = np.all(np.isfinite(X), axis=1)
    for t in result["_oos"]["t"][::37]:
        fc = ebr.knn_forecast(X, labels, has, int(t), 50)
        assert fc is not None
        assert np.all(labels.resolve_idx[fc["neighbors"]] < t)
        assert np.all(labels.resolve_idx[fc["eligible"]] >= 0)


def _toy(n_eligible: int = 30):
    """Toy labels: rows 0..n-1 eligible at t=n+5; row n resolves AT t; row n+1 immature."""
    rng = np.random.default_rng(3)
    t = n_eligible + 5
    n = t + 1
    X = rng.normal(size=(n, len(ebr.FEATURES)))
    resolve = np.full(n, -1)
    resolve[:n_eligible] = np.arange(n_eligible) + 1
    resolve[n_eligible] = t  # resolves at the forecast bar -> not eligible
    label = np.full(n, -1)
    label[:n_eligible] = rng.integers(0, 3, n_eligible)
    label[n_eligible] = ebr.TARGET
    labels = ebr.Labels(label, resolve, np.zeros(n, bool), np.zeros(n, bool), np.zeros(n), np.arange(n) + 1)
    X[n_eligible] = X[t]  # identical features: would be the nearest neighbor if it leaked
    X[n_eligible + 1] = X[t]
    return X, labels, np.ones(n, bool), t, n_eligible


def test_sample_resolving_at_forecast_bar_is_never_a_neighbor():
    X, labels, has, t, n_el = _toy()
    fc = ebr.knn_forecast(X, labels, has, t, 1)
    assert n_el not in fc["neighbors"] and n_el + 1 not in fc["neighbors"]
    assert set(fc["eligible"].tolist()) == set(range(n_el))
    assert ebr.knn_forecast(X, labels, has, t, n_el + 1) is None  # abstains, never pads


def test_scaler_fit_only_on_eligible_rows():
    X, labels, has, t, n_el = _toy()
    fc = ebr.knn_forecast(X, labels, has, t, 5)
    np.testing.assert_allclose(fc["scalerMean"], X[:n_el].mean(axis=0))
    np.testing.assert_allclose(fc["scalerStd"], X[:n_el].std(axis=0))
    X2 = X.copy()
    X2[n_el] = 1e9  # ineligible (resolves at t)
    X2[n_el + 1] = -1e9  # ineligible (immature)
    X2[t + 0] = X[t]
    fc2 = ebr.knn_forecast(X2, labels, has, t, 5)
    np.testing.assert_array_equal(fc2["scalerMean"], fc["scalerMean"])
    np.testing.assert_array_equal(fc2["probs"], fc["probs"])


def test_prior_uses_same_eligibility():
    _, labels, has, t, n_el = _toy()
    pr = ebr.prior_forecast(labels, has, t)
    expected = np.bincount(labels.label[:n_el], minlength=3) / n_el
    np.testing.assert_allclose(pr, expected)


# --- metrics -----------------------------------------------------------------


def test_n_effective_counts_non_overlapping_windows():
    idx = list(range(100))
    assert ebr.non_overlapping_count(idx, 20) == 5
    assert ebr.non_overlapping_count(idx, 1) == 100
    assert ebr.non_overlapping_count(idx, 20) < len(idx)


def test_block_bootstrap_respects_overlap():
    rng = np.random.default_rng(11)
    # Strongly overlapping series: each value repeated H times (like overlapping horizons).
    H = 20
    series = np.repeat(rng.normal(0, 1, 15), H)
    ci_block = ebr.block_bootstrap_ci(series, H)
    ci_iid = ebr.block_bootstrap_ci(series, 1)
    assert (ci_block[1] - ci_block[0]) > 2 * (ci_iid[1] - ci_iid[0])
    # Deterministic with the fixed seed.
    assert ebr.block_bootstrap_ci(series, H) == ci_block


def test_reliability_bins_and_ece():
    probs = np.array([0.1, 0.1, 0.9, 0.9, 0.5])
    outcomes = np.array([0, 0, 1, 1, 1])
    bins, ece = ebr.reliability(probs, outcomes)
    assert len(bins) == 5 and sum(b["n"] for b in bins) == 5
    assert ece == pytest.approx((2 * 0.1 + 2 * 0.1 + 0.5) / 5)


def test_gate_requires_all_three_conditions():
    ok = {"nOos": 100, "nEffective": 30, "brierDiffCi90": [-0.02, -0.001], "ece": 0.05}
    assert ebr.evaluate_gate(ok) == (True, [])
    assert not ebr.evaluate_gate({**ok, "nEffective": 29})[0]
    assert not ebr.evaluate_gate({**ok, "brierDiffCi90": [-0.02, 0.0]})[0]
    assert not ebr.evaluate_gate({**ok, "ece": 0.0501})[0]


# --- artifact ----------------------------------------------------------------


@pytest.fixture(scope="module")
def built(walk_rows):
    return ebr.build_artifact(walk_rows, "bid", cutoff_ms=10**15, non_trading_dates=frozenset())


def test_artifact_scope_and_schema(built, walk_rows):
    artifact, report = built
    assert artifact["schemaVersion"] == "equity-base-rates-artifact-v1"
    assert artifact["symbol"] == "BID"
    assert artifact["timeframe"] == "1d"
    assert artifact["priceBasis"] == "adjusted"
    assert artifact["provider"] == "VCI"
    assert artifact["dataAsOfOpenTimeMs"] == walk_rows[-1]["OpenTimeMs"]
    ds = artifact["dataset"]
    assert ds["barCount"] == len(walk_rows)
    assert ds["firstOpenTimeMs"] == walk_rows[0]["OpenTimeMs"]
    assert ds["lastOpenTimeMs"] == walk_rows[-1]["OpenTimeMs"]
    assert ds["maxRevision"] == 2
    assert ds["maxUpdatedAtUtc"] == (
        datetime(2026, 10, 1, tzinfo=timezone.utc) + timedelta(minutes=len(walk_rows) - 1)
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    assert len(ds["sha256"]) == 64
    assert artifact["method"]["version"] == "equity-knn-v1"
    assert artifact["method"]["features"] == list(ebr.FEATURES)
    assert artifact["method"]["kGrid"] == [20, 50, 100]
    assert artifact["protocol"]["version"] == "equity-base-rates-protocol-v1"
    assert artifact["protocol"]["frozenAt"] == "2026-10-16"
    assert [c["configId"] for c in artifact["configs"]] == ["tb-p5-s4-h20", "tb-p10-s6-h60"]
    for c in artifact["configs"]:
        assert set(c) == {"configId", "targetPct", "stopPct", "horizonSessions", "validation", "current"}
        v = c["validation"]
        for key in ("gatePassed", "gateReasons", "nOos", "nEffective", "brierKnn", "brierPrior",
                    "brierDiffCi90", "ece", "calibrationBins", "coverage", "abstentionRate",
                    "evNetOosMean", "evNetOosWhenPositive"):
            assert key in v
        cur = c["current"]
        assert cur["signalBarOpenTimeMs"] == artifact["dataAsOfOpenTimeMs"]
        for key in ("probTargetFirst", "probStopFirst", "probTimeout", "evNetFraction", "k", "nEligible", "prior"):
            assert key in cur
        if not cur["abstained"]:
            assert cur["probTargetFirst"] + cur["probStopFirst"] + cur["probTimeout"] == pytest.approx(1.0)
            assert cur["k"] in ebr.K_GRID
    # Report = same artifact + per-fold breakdown, JSON-serializable.
    assert report["artifact"] is artifact
    assert [p["configId"] for p in report["perConfig"]] == ["tb-p5-s4-h20", "tb-p10-s6-h60"]
    json.dumps(report, allow_nan=False)


def test_fingerprint_independent_of_input_order(walk_rows, built):
    shuffled = list(reversed(walk_rows))
    _, fp, _ = ebr.prepare_bars(shuffled, cutoff_ms=10**15)
    assert fp["sha256"] == built[0]["dataset"]["sha256"]


def test_unfinalized_bar_excluded_from_dataset(walk_rows):
    cutoff = walk_rows[-1]["CloseTimeMs"] - 1  # last bar still forming
    _, fp, _ = ebr.prepare_bars(walk_rows, cutoff_ms=cutoff)
    assert fp["barCount"] == len(walk_rows) - 1
    assert fp["lastOpenTimeMs"] == walk_rows[-2]["OpenTimeMs"]


@pytest.mark.parametrize("symbol", ["VCB", "CTG", "VNINDEX", "BTCUSDT", "ETHUSDT", ""])
def test_non_decision_symbols_rejected(symbol, walk_rows):
    with pytest.raises(ebr.NonDecisionSymbolError):
        ebr.build_artifact(walk_rows, symbol, cutoff_ms=10**15)


def test_mixed_basis_and_insufficient_data_fail_closed(walk_rows):
    bad = [dict(r, PriceBasis="raw") for r in walk_rows[:5]] + walk_rows[5:]
    with pytest.raises(ebr.BaseRatesError, match="price basis"):
        ebr.build_artifact(bad, "BID", cutoff_ms=10**15)
    with pytest.raises(ebr.InsufficientDataError) as exc:
        ebr.build_artifact([], "BID", cutoff_ms=10**15)
    assert exc.value.reason == "no-finalized-candles"
    with pytest.raises(ebr.InsufficientDataError) as exc:
        ebr.build_artifact(walk_rows[:150], "MBB", cutoff_ms=10**15)
    assert exc.value.reason == "insufficient-bars"


def test_short_history_builds_abstained_unvalidated_artifact(walk_rows):
    artifact, _ = ebr.build_artifact(walk_rows[:400], "MBB", cutoff_ms=10**15, non_trading_dates=frozenset())
    for c in artifact["configs"]:
        assert c["validation"]["gatePassed"] is False
        assert "no-oos-samples" in c["validation"]["gateReasons"]
        assert c["current"]["abstained"] is True
        assert c["current"]["probTargetFirst"] is None


def test_atomic_write_leaves_no_partial_file(tmp_path):
    target = ebr.artifact_path(tmp_path, "BID")
    ebr.write_json_atomic(target, {"ok": 1})
    with pytest.raises(ValueError):
        ebr.write_json_atomic(target, {"bad": float("nan")})  # allow_nan=False
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": 1}
    assert [p.name for p in target.parent.iterdir()] == ["base-rates.json"]
    fresh = tmp_path / "equity-base-rates" / "MBB" / "base-rates.json"
    with patch.object(ebr.os, "replace", side_effect=OSError("disk")):
        with pytest.raises(OSError):
            ebr.write_json_atomic(fresh, {"ok": 2})
    assert not fresh.exists()
    assert list(fresh.parent.iterdir()) == []


def test_cli_fixture_writes_artifact_and_report(tmp_path, walk_rows):
    fixture = tmp_path / "bid.json"
    fixture.write_text(json.dumps([{k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in r.items()}
                                   for r in walk_rows]), encoding="utf-8")
    code = ebr.main(["--symbol", "BID", "--candles-json", str(fixture), "--out-root", str(tmp_path / "ev"),
                     "--report-dir", str(tmp_path / "rep"), "--cutoff-ms", str(10**15)])
    assert code == 0
    art = json.loads((tmp_path / "ev" / "equity-base-rates" / "BID" / "base-rates.json").read_text(encoding="utf-8"))
    rep = json.loads((tmp_path / "rep" / "BID-report.json").read_text(encoding="utf-8"))
    assert art["symbol"] == "BID" and rep["artifact"]["dataset"]["sha256"] == art["dataset"]["sha256"]
    assert ebr.main(["--symbol", "VCB", "--candles-json", str(fixture), "--out-root", str(tmp_path / "ev")]) == 2
    assert not (tmp_path / "ev" / "equity-base-rates" / "VCB").exists()


# --- FastAPI endpoint --------------------------------------------------------


class TestEndpoint:
    def setup_method(self):
        from fastapi.testclient import TestClient

        self.client = TestClient(__import__("main").app)

    def test_build_writes_artifact_and_returns_summary(self, tmp_path, walk_rows):
        with patch.dict(os.environ, {"EQUITY_BASE_RATES_EVIDENCE_ROOT": str(tmp_path)}), \
                patch.object(ebr, "load_candles_db", return_value=walk_rows) as loader:
            response = self.client.post("/api/equity-base-rates/build?symbol=bid")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "built" and body["symbol"] == "BID"
        assert body["dataAsOfOpenTimeMs"] == walk_rows[-1]["OpenTimeMs"]
        assert {c["configId"] for c in body["configs"]} == {"tb-p5-s4-h20", "tb-p10-s6-h60"}
        assert all(isinstance(c["gatePassed"], bool) for c in body["configs"])
        assert loader.call_args[0][0] == "BID"
        assert (tmp_path / "equity-base-rates" / "BID" / "base-rates.json").exists()

    @pytest.mark.parametrize("symbol", ["VCB", "VNINDEX", "BTCUSDT", "XYZ"])
    def test_non_decision_symbol_is_400_without_db_access(self, symbol):
        with patch.object(ebr, "load_candles_db") as loader:
            response = self.client.post(f"/api/equity-base-rates/build?symbol={symbol}")
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "UNSUPPORTED_CAPABILITY"
        loader.assert_not_called()

    def test_missing_data_is_422_with_reason(self, tmp_path):
        with patch.dict(os.environ, {"EQUITY_BASE_RATES_EVIDENCE_ROOT": str(tmp_path)}), \
                patch.object(ebr, "load_candles_db", return_value=[]):
            response = self.client.post("/api/equity-base-rates/build?symbol=MBB")
        assert response.status_code == 422
        assert response.json()["detail"]["reason"] == "no-finalized-candles"
        assert not (tmp_path / "equity-base-rates").exists()

    def test_db_failure_is_503(self):
        with patch.object(ebr, "load_candles_db", side_effect=RuntimeError("down")):
            response = self.client.post("/api/equity-base-rates/build?symbol=MBB")
        assert response.status_code == 503
