"""Roadmap 1.1-raw: raw HOSE prices - tick sizes, ceiling/floor, reference price, limit-locked opens, traded value,
and the polite, write-once CafeF fetcher (exercised with a fake provider; no network)."""
import json
import sys
from datetime import date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

AI_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(AI_ROOT))

import vn_raw_prices as vr  # noqa: E402

# A provider record exactly as served (MBB, 2026-11-06).
MBB_REC = {"Symbol": "MBB", "Ngay": "06/11/2026", "GiaDieuChinh": 19.25, "GiaDongCua": 19.25,
           "ThayDoi": "-0,25 (-1,28%)", "KhoiLuongKhopLenh": 12496300, "GiaTriKhopLenh": 239.78,
           "KLThoaThuan": 0, "GtThoaThuan": 0, "GiaMoCua": 19, "GiaCaoNhat": 19.35, "GiaThapNhat": 19}


def rec(day: str, close: float, *, adj: float | None = None, change: str | None = None, o: float | None = None,
        h: float | None = None, lo: float | None = None, vol: int = 1000, value: float | None = None,
        pt_vol: int = 0, pt_value: float = 0.0, symbol: str = "AAA") -> dict:
    """Provider-shaped record; prices in thousand VND, values in billion VND, day ISO."""
    o = close if o is None else o
    h = max(o, close) if h is None else h
    lo = min(o, close) if lo is None else lo
    value = round(vol * close / 1e6, 2) if value is None else value
    return {"Symbol": symbol, "Ngay": datetime.strptime(day, "%Y-%m-%d").strftime("%d/%m/%Y"),
            "GiaDieuChinh": close if adj is None else adj, "GiaDongCua": close,
            "ThayDoi": change if change is not None else "0 (0 %)", "KhoiLuongKhopLenh": vol,
            "GiaTriKhopLenh": value, "KLThoaThuan": pt_vol, "GtThoaThuan": pt_value, "GiaMoCua": o,
            "GiaCaoNhat": h, "GiaThapNhat": lo}


def chg(new: float, ref: float) -> str:
    d = new - ref
    return f"{'+' if d >= 0 else ''}{d:.2f} ({d / ref * 100:+.2f}%)".replace(".", ",")


def series(*recs: dict, symbol: str = "AAA") -> vr.RawSeries:
    return vr.RawSeries.from_records(symbol, list(recs))


# ------------------------------------------------------------------------------------------- tick and band rules

@pytest.mark.parametrize("price,tick", [(10, 10), (9_990, 10), (9_999.9, 10), (10_000, 50), (49_950, 50),
                                        (49_999, 50), (50_000, 100), (250_000, 100)])
def test_hose_tick_size_follows_the_three_price_levels(price, tick):
    assert vr.hose_tick_size(price) == tick


def test_tick_size_rejects_non_positive_prices():
    with pytest.raises(ValueError):
        vr.hose_tick_size(0)


def test_rounding_to_tick_uses_the_tick_of_the_resulting_price_level():
    assert vr.round_down_to_tick(50_290) == 50_200  # 100-VND level
    assert vr.round_down_to_tick(49_999) == 49_950
    assert vr.round_down_to_tick(10_049) == 10_000
    assert vr.round_up_to_tick(9_995) == 10_000
    assert vr.round_up_to_tick(49_960) == 50_000
    assert vr.round_up_to_tick(2_883) == 2_890
    assert vr.round_down_to_tick(2_890) == 2_890 and vr.round_up_to_tick(2_890) == 2_890


def test_price_limits_round_inward_to_valid_ticks():
    # worked example of the HOSE rule: reference 95,000 -> ceiling 101,600, floor 88,400
    assert vr.price_limits(95_000) == (88_400, 101_600)
    assert vr.price_limits(3_100) == (2_890, 3_310)
    assert vr.price_limits(47_000) == (43_750, 50_200)  # ceiling crosses into the 100-VND level
    assert vr.price_limits(10_000) == (9_300, 10_700)
    floor, ceiling = vr.price_limits(24_400, 0.20)
    assert (floor, ceiling) == (19_550, 29_250)


def test_price_limits_nearest_mode_and_validation():
    assert vr.price_limits(3_100, rounding="nearest") == (2_880, 3_320)
    with pytest.raises(ValueError):
        vr.price_limits(0)
    with pytest.raises(ValueError):
        vr.price_limits(10_000, 1.5)
    with pytest.raises(ValueError):
        vr.price_limits(10_000, rounding="outward")


def test_band_default_comes_from_the_execution_model():
    from vn_execution_model import DEFAULT_PRICE_BAND_FRACTION
    assert DEFAULT_PRICE_BAND_FRACTION == 0.07
    assert vr.FIRST_DAY_BAND_FRACTION == 0.20


# ------------------------------------------------------------------------------------------------------ parsing

@pytest.mark.parametrize("text,value", [("-0,25 (-1,28%)", -0.25), ("+0,05 (+0,20%)", 0.05), ("0 (0 %)", 0.0),
                                        ("+2,1 (+2,3%)", 2.1), ("1.5 (1%)", 1.5), ("", None), (None, None),
                                        ("n/a", None)])
def test_parse_change(text, value):
    assert vr.parse_change(text) == value


def test_parse_record_converts_provider_units_to_vnd():
    bar = vr.parse_record(MBB_REC)
    assert bar.date == "2026-11-06"
    assert (bar.open, bar.high, bar.low, bar.close) == (19_000, 19_350, 19_000, 19_250)
    assert bar.adj_close == 19_250
    assert bar.provider_reference == 19_500  # 19.25 - (-0.25)
    assert bar.matched_volume == 12_496_300
    assert bar.matched_value_vnd == 239_780_000_000
    assert bar.put_through_value_vnd == 0
    # implied matched VWAP lies inside the session range: confirms "billion VND" for values
    assert bar.low <= bar.matched_value_vnd / bar.matched_volume <= bar.high


def test_validate_records_flags_ohlc_and_vwap_anomalies():
    good = rec("2026-01-05", 20.0, o=19.8, h=20.2, lo=19.7, vol=1_000_000, value=20.0)
    bad_range = rec("2026-01-06", 20.0, o=21.0, h=20.2, lo=19.7)
    bad_vwap = rec("2026-01-07", 20.0, o=19.8, h=20.2, lo=19.7, vol=1_000_000, value=40.0)
    assert vr.validate_records("AAA", [good, bad_range, bad_vwap]) == ["2026-01-06", "2026-01-07"]
    with pytest.raises(ValueError):
        vr.validate_records("AAA", [rec("2026-01-05", 0.0)])


# ----------------------------------------------------------------------------------------------------- fetching

def test_query_url_uses_month_first_dates_and_the_provider_page_cap():
    q = parse_qs(urlparse(vr.query_url("MBB", date(2025, 1, 2), date(2025, 7, 1), 3)).query)
    assert q == {"Symbol": ["MBB"], "StartDate": ["01/02/2025"], "EndDate": ["07/01/2025"], "PageIndex": ["3"],
                 "PageSize": ["30"]}
    assert vr.query_url("MBB", date(2025, 1, 2), date(2025, 7, 1), 1).startswith(vr.SOURCE_URL + "?")


def test_windows_are_half_years_that_overlap_by_one_day():
    w = vr.windows(date(2016, 1, 1), date(2017, 3, 10))
    assert w == [(date(2016, 1, 1), date(2016, 7, 1)), (date(2016, 7, 1), date(2017, 1, 1)),
                 (date(2017, 1, 1), date(2017, 3, 10))]
    assert vr.windows(date(2016, 8, 15), date(2016, 9, 1)) == [(date(2016, 8, 15), date(2016, 9, 1))]


def test_requests_start_at_least_delay_apart():
    assert vr.DELAY_S >= 1.0
    assert vr.pause_after(0.0) == vr.DELAY_S
    assert vr.pause_after(0.7) == pytest.approx(vr.DELAY_S - 0.7)
    assert vr.pause_after(5.0) == vr.MIN_PAUSE_S


def test_fetch_cutoff_keeps_the_fetch_day_only_after_the_session_is_final():
    assert vr.fetch_cutoff(datetime(2026, 11, 6, 15, 59)) == date(2026, 11, 6)
    assert vr.fetch_cutoff(datetime(2026, 11, 6, 16, 0)) == date(2026, 11, 7)


class FakeProvider:
    """Serves provider-shaped pages for one or more symbols; records every URL asked."""

    def __init__(self, data: dict[str, list[dict]], end_inclusive: bool = True, total_bias: int = 0):
        self.data, self.end_inclusive, self.total_bias, self.urls = data, end_inclusive, total_bias, []

    def __call__(self, url: str) -> dict:
        self.urls.append(url)
        q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        start = datetime.strptime(q["StartDate"], "%m/%d/%Y").date()
        end = datetime.strptime(q["EndDate"], "%m/%d/%Y").date()
        rows = [dict(r) for r in self.data.get(q["Symbol"], [])
                if start <= datetime.strptime(r["Ngay"], "%d/%m/%Y").date() <= end
                and (self.end_inclusive or datetime.strptime(r["Ngay"], "%d/%m/%Y").date() < end)]
        rows.sort(key=lambda r: datetime.strptime(r["Ngay"], "%d/%m/%Y"), reverse=True)  # newest first
        page, size = int(q["PageIndex"]), min(int(q["PageSize"]), 30)
        return {"Data": {"TotalCount": len(rows) + self.total_bias, "Data": rows[(page - 1) * size: page * size]},
                "Message": None, "Success": True}


def _daily(symbol: str, start: date, n: int) -> list[dict]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(rec(d.isoformat(), 20.0 + len(out) * 0.05, symbol=symbol))
        d = date.fromordinal(d.toordinal() + 1)
    return out


@pytest.mark.parametrize("end_inclusive", [True, False])
def test_fetch_symbol_pages_dedupes_overlap_and_applies_the_cutoff(end_inclusive):
    rows = _daily("AAA", date(2016, 6, 1), 120)  # spans the 2016-07-01 window boundary
    fake = FakeProvider({"AAA": rows}, end_inclusive=end_inclusive)
    sleeps = []
    got = vr.fetch_symbol("AAA", date(2016, 1, 1), date(2016, 10, 3), fake, sleeps.append)
    days = [vr.parse_record(r).date for r in got]
    expected = [vr.parse_record(r).date for r in rows if vr.parse_record(r).date < "2016-10-03"]
    assert days == expected  # complete, unique, oldest first, cutoff excluded
    assert len(fake.urls) == len(sleeps) and all(s >= vr.MIN_PAUSE_S for s in sleeps)  # one pause per request
    assert any("PageIndex=2" in u for u in fake.urls)


def test_fetch_symbol_fails_on_short_pages_or_disagreeing_overlap():
    rows = _daily("AAA", date(2016, 6, 1), 40)
    with pytest.raises(ValueError, match="TotalCount"):
        vr.fetch_symbol("AAA", date(2016, 1, 1), date(2016, 9, 1), FakeProvider({"AAA": rows}, total_bias=5),
                        lambda s: None)

    class Disagree(FakeProvider):
        def __call__(self, url):
            out = super().__call__(url)
            if "StartDate=07%2F01%2F2016" in url:
                for r in out["Data"]["Data"]:
                    if r["Ngay"] == "01/07/2016":
                        r["GiaDongCua"] = 99.0
            return out
    with pytest.raises(ValueError, match="disagree"):
        vr.fetch_symbol("AAA", date(2016, 1, 1), date(2016, 9, 1), Disagree({"AAA": rows}), lambda s: None)


def test_fetch_symbol_collapses_zero_volume_shadow_rows():
    # GVR (UPCoM, 2018) is served with two rows per session: one with the matched volume, one with zeros
    rows = _daily("AAA", date(2018, 6, 25), 5)
    shadow = [r | {"KhoiLuongKhopLenh": 0, "GiaTriKhopLenh": 0} for r in rows]
    stats = {}
    got = vr.fetch_symbol("AAA", date(2018, 1, 1), date(2018, 8, 1), FakeProvider({"AAA": rows + shadow}),
                          lambda s: None, stats)
    assert [r["KhoiLuongKhopLenh"] for r in got] == [1000] * 5
    assert stats["duplicatesCollapsed"] == 5
    with pytest.raises(ValueError, match="disagree"):
        vr.collapse_duplicate("AAA", "2018-06-25", rows[0], rows[0] | {"GiaDongCua": 1.0})


def test_fetch_symbol_rejects_records_of_another_symbol_and_bad_payloads():
    with pytest.raises(ValueError, match="another symbol"):
        vr.fetch_symbol("AAA", date(2016, 1, 1), date(2016, 3, 1),
                        lambda url: {"Success": True, "Data": {"TotalCount": 1,
                                                               "Data": [rec("2016-01-05", 20.0, symbol="BBB")]}},
                        lambda s: None)
    with pytest.raises(ValueError, match="unexpected payload"):
        vr.fetch_symbol("AAA", date(2016, 1, 1), date(2016, 3, 1), lambda url: {"Success": False}, lambda s: None)


def test_run_fetch_is_write_once_resumable_and_records_missing_symbols(tmp_path):
    fake = FakeProvider({"AAA": _daily("AAA", date(2016, 1, 4), 30), "BBB": _daily("BBB", date(2016, 1, 4), 5)})
    m = vr.run_fetch(["AAA", "BBB", "ZZZ"], tmp_path, date(2016, 1, 1), date(2016, 3, 1), fake, lambda s: None,
                     log=lambda s: None, fetched_on="2016-03-01")
    assert sorted(m["symbols"]) == ["AAA", "BBB"] and m["missing"] == {"ZZZ": "provider returned no bars"}
    assert m["cutoffExclusive"] == "2016-03-01"
    entry = m["symbols"]["AAA"]
    assert entry["bars"] == 30 and entry["first"] == "2016-01-04"
    doc = json.loads((tmp_path / "AAA.json").read_text(encoding="utf-8"))
    assert doc["schema"] == vr.SCHEMA and doc["records"][0]["Ngay"] == "04/01/2016"  # verbatim provider records
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved == m
    # resume: existing files are kept (no request), the missing symbol is retried, the cutoff stays the snapshot's
    fake.urls.clear()
    m2 = vr.run_fetch(["AAA", "ZZZ"], tmp_path, date(2016, 1, 1), date(2016, 6, 1), fake, lambda s: None,
                      log=lambda s: None)
    assert all("Symbol=ZZZ" in u for u in fake.urls)
    assert m2["symbols"]["AAA"] == entry and m2["cutoffExclusive"] == "2016-03-01"
    # a tampered file is refused, never silently replaced
    (tmp_path / "AAA.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the manifest"):
        vr.run_fetch(["AAA"], tmp_path, date(2016, 1, 1), date(2016, 3, 1), fake, lambda s: None, log=lambda s: None)
    with pytest.raises(ValueError, match="another --from"):
        vr.run_fetch(["AAA"], tmp_path, date(2015, 1, 1), date(2016, 3, 1), fake, lambda s: None, log=lambda s: None)


def test_write_symbol_file_never_overwrites(tmp_path):
    vr.write_symbol_file(tmp_path, "AAA", [], "2016-01-01", "2016-03-01")
    with pytest.raises(FileExistsError):
        vr.write_symbol_file(tmp_path, "AAA", [], "2016-01-01", "2016-03-01")


def test_tickers_are_every_symbol_ever_in_the_membership_file():
    t = vr.tickers(AI_ROOT / "contracts" / "vn30-membership.json")
    assert len(t) == len(set(t)) and {"MBB", "BID", "ROS", "TCX"} <= set(t)


# --------------------------------------------------------------------------------- reference, limits, value

def _snapshot(tmp_path: Path, symbol: str, records: list[dict]) -> vr.RawPrices:
    m = vr.run_fetch([symbol], tmp_path, date(2016, 1, 1), date(2030, 1, 1), FakeProvider({symbol: records}),
                     lambda s: None, log=lambda s: None, fetched_on="2026-11-06")
    assert symbol in m["symbols"]
    return vr.RawPrices(tmp_path)


def test_reference_price_is_the_previous_raw_close_and_exchange_basis_carries_ex_right_adjustments():
    s = series(rec("2025-01-06", 24.4, adj=18.45),
               # ex-right session: the exchange reference is 21.22, not the previous close 24.4
               rec("2025-01-07", 21.65, adj=18.82, change="+0,43 (+2,03%)", o=21.7, h=21.7, lo=21.2),
               rec("2025-01-08", 21.7, adj=18.86, change="+0,05 (+0,23%)"))
    assert vr.reference_price(s, "2025-01-07") == 24_400
    assert vr.reference_price(s, "2025-01-07", "exchange") == 21_220
    assert vr.reference_price(s, "2025-01-08", "exchange") == 21_650
    assert vr.reference_price(s, "2025-01-06") is None  # first bar: unknown
    with pytest.raises(ValueError):
        vr.reference_price(s, "2025-01-07", "vwap")


def test_exchange_reference_ignores_provider_rounding_noise_but_bridges_provider_holes():
    s = series(rec("2026-03-19", 52.4), rec("2026-03-20", 48.85, change="-3,52 (-6,72%)"),  # 30 VND off: noise
               rec("2026-01-28", 26.55), rec("2026-02-02", 27.3, change="+0,10 (+0,37%)"))  # 29/30 Jan missing
    assert vr.reference_price(s, "2026-03-20", "exchange") == 52_400
    assert vr.reference_price(s, "2026-02-02", "exchange") == 27_200  # close of the missing session
    assert vr.reference_noise_vnd(52_400) == 100 and vr.reference_noise_vnd(200_000) == 200


def test_exchange_reference_rejects_a_provider_change_that_contradicts_the_session_range():
    # ACB 2021-08-25 as served: ThayDoi -2,55 implies a reference of 35,000, but 31,900..32,500 is outside its band
    s = series(rec("2021-08-24", 32.3, adj=13.05), rec("2021-08-25", 32.45, adj=13.11, change="-2,55 (-7,29%)",
                                                        o=32.2, h=32.5, lo=31.9))
    assert vr.reference_candidates(s.bar("2021-08-25"), s.bar("2021-08-24")) == [35_000, 32_300]
    assert vr.reference_price(s, "2021-08-25", "exchange") == 32_300


def test_exchange_reference_uses_the_adjustment_ratio_when_the_change_field_missed_a_corporate_action():
    # ACB 2021-06-10: 25% stock dividend, the change field still reports the unadjusted previous close
    s = series(rec("2021-06-09", 42.45, adj=13.72), rec("2021-06-10", 34.1, adj=13.78, change="-8,35 (-19,67%)",
                                                        o=34.5, h=35.35, lo=34.1))
    ref = vr.reference_price(s, "2021-06-10", "exchange")
    assert ref == round(42_450 * (13.72 / 42.45) / (13.78 / 34.1))
    floor, ceiling = vr.price_limits(ref)
    assert floor <= 34_100 and 35_350 <= ceiling
    assert vr.reference_price(s, "2021-06-10") == 42_450  # the plain basis stays the previous raw close


def test_limit_locked_open_by_side(tmp_path):
    p = _snapshot(tmp_path, "AAA", [
        rec("2026-01-05", 10.0),
        rec("2026-01-06", 10.7, o=10.7, h=10.7, lo=10.7, change="+0,70 (+7,00%)"),  # opens at the ceiling
        rec("2026-01-07", 9.96, o=9.96, h=10.2, lo=9.96, change="-0,74 (-6,92%)"),  # opens at the floor (9,960)
        rec("2026-01-08", 10.0, o=10.0, change="+0,04 (+0,40%)"),
        rec("2026-01-09", 10.0, o=10.5, h=10.2, lo=9.9),
    ])
    assert p.limit_locked_open("AAA", "2026-01-06", "buy") is True
    assert p.limit_locked_open("AAA", "2026-01-06", "sell") is False
    assert p.price_limits("AAA", "2026-01-07") == (9_960, 11_400)
    assert p.limit_locked_open("AAA", "2026-01-07", "sell") is True
    assert p.limit_locked_open("AAA", "2026-01-07", "buy") is False
    assert p.limit_locked_open("AAA", "2026-01-08", "buy") is False
    assert p.limit_locked_open("AAA", "2026-01-05", "buy") is False  # first bar: the provider's reference
    assert p.limit_locked_open("AAA", "2026-01-05", "buy", basis="previous-close") is None  # no previous bar
    assert p.limit_locked_open("AAA", "2026-02-01", "buy") is None  # no bar
    assert p.limit_locked_open("AAA", "2026-01-09", "buy") is None  # provider open outside low..high
    with pytest.raises(ValueError):
        p.limit_locked_open("AAA", "2026-01-06", "short")


def test_no_opening_print_is_treated_as_locked():
    bar = vr.parse_record(rec("2026-01-06", 10.0, o=0, h=10.0, lo=10.0))
    assert vr.limit_locked_open_bar(bar, 10_000, "buy") and vr.limit_locked_open_bar(bar, 10_000, "sell")


def test_traded_value_is_matched_value_plus_optional_put_through(tmp_path):
    p = _snapshot(tmp_path, "MBB", [MBB_REC | {"Ngay": "05/11/2026"},
                                    MBB_REC | {"KLThoaThuan": 100000, "GtThoaThuan": 2.02}])
    assert p.traded_value("MBB", "2026-11-06") == 239_780_000_000
    assert p.traded_value("MBB", "2026-11-06", include_put_through=True) == 241_800_000_000
    assert p.traded_value("MBB", "2026-11-07") is None


def test_raw_prices_refuses_a_file_that_does_not_match_the_manifest(tmp_path):
    p = _snapshot(tmp_path, "AAA", [rec("2026-01-05", 10.0)])
    path = tmp_path / "AAA.json"
    path.write_text(path.read_text(encoding="utf-8").replace("10.0", "11.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        p.series("AAA")


def test_band_fraction_is_20pct_on_listing_and_after_a_long_suspension():
    market = [f"2026-{m:02d}-{d:02d}" for m in (1, 2, 3) for d in range(1, 29)]
    s = series(rec("2026-01-10", 10.0), rec("2026-01-11", 10.1), rec("2026-02-20", 10.2))
    assert vr.band_fraction(s, "2026-01-10", market) == 0.20  # first bar after the market already traded
    assert vr.band_fraction(s, "2026-01-11", market) == 0.07
    assert vr.band_fraction(s, "2026-02-20", market) == 0.20  # 36 market sessions missed
    assert vr.band_fraction(s, "2026-01-11") == 0.07  # without a market calendar only the default


# ------------------------------------------------------------------------------------------------- cross-checks

def test_ratio_steps_separates_corporate_actions_from_one_day_glitches():
    pairs = [("d1", 10_000, 8_000), ("d2", 10_100, 8_080),
             ("d3", 9_000, 9_000),  # ratio 0.8 -> 1.0 and stays: a corporate action
             ("d4", 9_100, 9_100),
             ("d5", 9_200, 9_800),  # one-day deviation, back to 1.0 next day: a data error
             ("d6", 9_300, 9_300)]
    out = vr.ratio_steps(pairs, lambda adj: 0.001)
    assert [s["date"] for s in out["steps"]] == ["d3"]
    assert [t["date"] for t in out["transients"]] == ["d5"]


def test_crosscheck_symbol_reports_steps_references_band_and_proxy(tmp_path):
    recs = [rec("2026-01-05", 20.0, adj=10.0), rec("2026-01-06", 20.2, adj=10.1, change="+0,20 (+1,00%)"),
            rec("2026-01-07", 10.6, adj=10.6, change="+0,50 (+4,95%)", o=10.5, h=10.6, lo=10.5),  # 2:1 split
            rec("2026-01-08", 11.3, adj=11.3, change="+0,70 (+6,60%)", o=11.3, h=11.3, lo=11.3)]  # ceiling 11,300
    adjusted = {"2026-01-05": {"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 1},
                "2026-01-06": {"open": 10.1, "high": 10.1, "low": 10.1, "close": 10.1, "volume": 1},
                "2026-01-07": {"open": 10.5, "high": 10.6, "low": 10.5, "close": 10.6, "volume": 1},
                "2026-01-08": {"open": 11.3, "high": 11.3, "low": 11.3, "close": 11.3, "volume": 1},
                "2026-01-09": {"open": 11.3, "high": 11.3, "low": 11.3, "close": 11.3, "volume": 1}}
    out = vr.crosscheck_symbol(series(*recs), adjusted)
    assert out["cafefSteps"] == ["2026-01-07"] and out["vciSteps"] == ["2026-01-07"]
    assert out["referenceSource"] == {"provider": 1, "ratio": 0, "previous-close": 2}
    assert out["providerReferenceRejected"] == [] and out["providerReferenceWithoutStep"] == []
    assert out["stepsWithoutProviderReference"] == [] and out["stepsOnlyVci"] == []
    assert out["bandViolations"] == [] and out["unexplainedSessions"] == []
    assert out["plainSessions"] == 2
    assert out["limitLockedOpens"] == {"buy": 1, "sell": 0}
    assert out["sessionsOnlyVciCount"] == 0  # 2026-01-09 is after the raw series ends
    assert out["proxyVsRawLock"]["buy"]["both"] == 1
