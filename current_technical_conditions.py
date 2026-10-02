"""Current technical conditions for the latest finalized contiguous BTCUSDT bar.

Implements the frozen ``current-conditions-v1`` contract
(``docs/research/current-conditions-contract.md``).  Detector semantics are
reused from ``technical_event_modules`` and
``technical_event_descriptive_evidence`` — this module only filters the latest
finalized contiguous Klines window, re-runs the contract detectors on it, and
classifies each result into ``triggeredOnBar`` / ``state`` / ``activeZone`` /
``operativeLeg`` conditions at ``asOfMs``.  No formulas are re-implemented here
and no predictive claim is made anywhere in the payload.
"""

from __future__ import annotations

import json
import time
from bisect import bisect_right
from typing import Any, Mapping, Sequence

from technical_event_modules import (
    _direction,
    _ema,
    _profile,
    _rsi,
    build_causal_technical_events,
)
from technical_event_descriptive_evidence import (
    Candle,
    _LifecycleLevelIndex,
    _causal_contexts,
    _smc_lifecycle,
    _table_columns,
    sha256_bytes,
)


SCHEMA_VERSION = "current-conditions-v1"
SYMBOL = "BTCUSDT"
TIMEFRAME_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
WINDOW_BARS = 2_000

# Canonical immutable-decision schema required from CausalSmartMoneyEvents
# (same set the descriptive-evidence pipeline verifies before trusting a row).
CANONICAL_SMC_COLUMNS = frozenset(
    {
        "EventId",
        "EventType",
        "OriginTimeMs",
        "AvailableTimeMs",
        "CalculationVersion",
        "DecisionSourceOpenTimeMsJson",
        "DecisionEvidenceJson",
        "DecisionEvidenceSha256",
    }
)

_DIRECTION_LABEL = {1: "bullish", -1: "bearish", 0: "neutral"}
_FVG_EVENT_TYPES = frozenset({"FVG_BULL", "FVG_BEAR"})

# Presentation cap: an unbounded count of far-off-price zones can stay
# "unmitigated" for years; emit only the nearest-to-close zones per type.
ACTIVE_ZONE_LIMIT_PER_TYPE = 25


class UnknownTimeframeError(ValueError):
    """Raised when timeframe is outside the contract scope (1h/4h/1d)."""


class ConditionsUnavailableError(RuntimeError):
    """Raised when finalized klines cannot be loaded for the timeframe."""


def _now_ms() -> int:
    return int(time.time() * 1000)


def _load_inputs(
    timeframe: str, now_ms: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], str | None]:
    """Load klines + causal SMC rows inside one read-only snapshot.

    Returns ``(kline_rows_asc, causal_smc_rows, causal_smc_columns,
    causal_smc_error)``.  Kline rows are the newest finalized, exact-duration
    bars (ascending) capped at ``WINDOW_BARS + 1`` so the caller can detect
    truncation at the window edge.  A failure of the causal-SMC read is
    reported through ``causal_smc_error`` instead of raising so the module is
    marked unavailable rather than failing the whole payload.
    """
    from db_config import get_db_connection

    interval = TIMEFRAME_MS[timeframe]
    connection = get_db_connection()
    try:
        # Same consistency pattern as the descriptive-evidence PostgreSQL
        # loader: separate statements must share one repeatable-read snapshot.
        connection.set_session(
            readonly=True,
            autocommit=False,
            isolation_level="REPEATABLE READ",
        )
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT "OpenTimeMs","CloseTimeMs","Open","High","Low","Close","Volume" '
                'FROM "Klines" WHERE "Symbol"=%s AND "Timeframe"=%s '
                'AND "CloseTimeMs"<=%s AND "CloseTimeMs"-"OpenTimeMs"+1=%s '
                'ORDER BY "OpenTimeMs" DESC LIMIT %s',
                (SYMBOL, timeframe, now_ms, interval, WINDOW_BARS + 1),
            )
            names = ("OpenTimeMs", "CloseTimeMs", "Open", "High", "Low", "Close", "Volume")
            klines = [dict(zip(names, row)) for row in cursor.fetchall()]
            klines.reverse()

            causal_columns: set[str] = set()
            causal_rows: list[dict[str, Any]] = []
            causal_error: str | None = None
            try:
                causal_columns = _table_columns(cursor, "CausalSmartMoneyEvents")
                if CANONICAL_SMC_COLUMNS.issubset(causal_columns) and klines:
                    as_of_ms = int(klines[-1]["CloseTimeMs"])
                    selected = [
                        name
                        for name in (
                            "Id",
                            "EventId",
                            "EventType",
                            "OriginTimeMs",
                            "AvailableTimeMs",
                            "ReferenceTimeMs",
                            "Price",
                            "HighPrice",
                            "LowPrice",
                            "State",
                            "MitigatedAtMs",
                            "CalculationVersion",
                            "DecisionSourceCandleCount",
                            "DecisionSourceOpenTimeMsJson",
                            "DecisionEvidenceJson",
                            "DecisionEvidenceSha256",
                        )
                        if name in causal_columns
                    ]
                    quoted = ",".join(f'"{name}"' for name in selected)
                    cursor.execute(
                        f'SELECT {quoted} FROM "CausalSmartMoneyEvents" '
                        'WHERE "Symbol"=%s AND "Timeframe"=%s AND "AvailableTimeMs"<=%s '
                        'ORDER BY "AvailableTimeMs", "EventId"',
                        (SYMBOL, timeframe, as_of_ms),
                    )
                    causal_rows = [dict(zip(selected, row)) for row in cursor.fetchall()]
            except Exception as exc:  # module degradation, not a 503
                causal_error = f"causalSmc read failed: {type(exc).__name__}"
                causal_columns = set()
                causal_rows = []
        connection.rollback()
        return klines, causal_rows, sorted(causal_columns), causal_error
    finally:
        connection.close()


def _verified_causal_smc(
    rows: Sequence[Mapping[str, Any]], as_of_ms: int
) -> list[tuple[dict[str, Any], dict[str, Any], list[int]]]:
    """Verify canonical causal-SMC rows exactly like the evidence pipeline.

    ``sha256(DecisionEvidenceJson)`` must equal ``DecisionEvidenceSha256`` and
    the immutable decision fields must agree with the hashed evidence.  Any
    mismatch raises ``ValueError`` — callers must mark the whole module
    unavailable, never silently skip the row.  Rows with unknown or future
    availability are skipped the same way the pipeline's cutoff filter skips
    them (they are not yet observable).
    """
    verified: list[tuple[dict[str, Any], dict[str, Any], list[int]]] = []
    ordered = sorted(
        rows,
        key=lambda row: (
            row.get("AvailableTimeMs") if isinstance(row.get("AvailableTimeMs"), int) else 2**63 - 1,
            str(row.get("EventId") or ""),
        ),
    )
    for raw in ordered:
        row = dict(raw)
        available = row.get("AvailableTimeMs")
        if not isinstance(available, int) or available > as_of_ms:
            continue
        evidence_json = row.get("DecisionEvidenceJson")
        evidence_sha = str(row.get("DecisionEvidenceSha256") or "").lower()
        if not isinstance(evidence_json, str) or sha256_bytes(evidence_json.encode("utf-8")) != evidence_sha:
            raise ValueError("CausalSmartMoneyEvents decision evidence hash mismatch")
        try:
            evidence = json.loads(evidence_json)
            source_times = json.loads(str(row.get("DecisionSourceOpenTimeMsJson") or "[]"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("CausalSmartMoneyEvents decision evidence JSON is invalid") from exc
        if (
            not isinstance(evidence, Mapping)
            or not isinstance(source_times, list)
            or not source_times
            or any(not isinstance(value, int) for value in source_times)
            or len(set(source_times)) != len(source_times)
        ):
            raise ValueError("CausalSmartMoneyEvents decision source list is invalid")
        evidence_sources = evidence.get("sourceCandles")
        if not isinstance(evidence_sources, list) or [
            item.get("openTimeMs") for item in evidence_sources
        ] != source_times:
            raise ValueError("CausalSmartMoneyEvents source list disagrees with decision evidence")
        if row.get("DecisionSourceCandleCount") is not None and int(row["DecisionSourceCandleCount"]) != len(source_times):
            raise ValueError("CausalSmartMoneyEvents decision source count mismatch")
        for column, evidence_key in (
            ("EventId", "eventId"),
            ("EventType", "eventType"),
            ("OriginTimeMs", "originTimeMs"),
            ("AvailableTimeMs", "availableTimeMs"),
            ("CalculationVersion", "calculationVersion"),
        ):
            if evidence.get(evidence_key) != row.get(column):
                raise ValueError(f"CausalSmartMoneyEvents immutable decision mismatch: {column}")
        event_type = str(row["EventType"])
        expected_state = "active" if event_type in _FVG_EVENT_TYPES else "confirmed"
        if evidence.get("stateAtAsOf") != expected_state or evidence.get("mitigatedAtMs") is not None:
            raise ValueError("CausalSmartMoneyEvents decision evidence contains future lifecycle state")
        verified.append((row, evidence, list(source_times)))
    return verified


def _condition(
    *,
    module: str,
    event_type: str,
    kind: str,
    event_id: Any,
    formed_ms: int,
    available_ms: int,
    context: Mapping[str, Any] | None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "module": module,
        "eventType": event_type,
        "kind": kind,
        "direction": _DIRECTION_LABEL[_direction(event_type)],
        "eventId": event_id,
        "formedTimeMs": int(formed_ms),
        "availableTimeMs": int(available_ms),
        "context": dict(context or {}),
        "details": dict(details or {}),
    }


def _event_condition(event: Mapping[str, Any], kind: str) -> dict[str, Any]:
    return _condition(
        module=str(event.get("module")),
        event_type=str(event["eventType"]),
        kind=kind,
        event_id=event.get("eventId"),
        formed_ms=int(event["formedTimeMs"]),
        available_ms=int(event["availableTimeMs"]),
        context=event.get("context"),
        details={"sourceCandleOpenTimesMs": list(event.get("sourceCandleOpenTimesMs") or [])},
    )


def _state_condition(
    module: str,
    event_type: str,
    as_of_ms: int,
    context: Mapping[str, Any],
    details: Mapping[str, Any],
) -> dict[str, Any]:
    return _condition(
        module=module,
        event_type=event_type,
        kind="state",
        event_id=None,
        formed_ms=as_of_ms,
        available_ms=as_of_ms,
        context=context,
        details=details,
    )


def _state_conditions(
    window: Sequence[Mapping[str, Any]],
    contexts: Sequence[Mapping[str, str]],
    metadata: Mapping[str, Any],
    as_of_ms: int,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Persistent conditions at ``asOfMs`` evaluated with contract helpers."""
    conditions: list[dict[str, Any]] = []
    closes = [float(row["Close"]) for row in window]
    last_context = dict(contexts[-1])
    indicator_params = metadata["technicalIndicators"]["parameters"]
    profile_params = metadata["volumeProfile"]["parameters"]

    # Regime pair from the causal per-bar context of the decision bar.
    trend_token = {"up": "BULL", "down": "BEAR", "sideways": "SIDEWAYS"}.get(last_context["trend"])
    volatility_token = {"high": "HIGH", "normal": "NORMAL", "low": "LOW"}.get(
        last_context["volatility"]
    )
    if trend_token and volatility_token:
        conditions.append(
            _state_condition(
                "marketRegime",
                f"REGIME_{trend_token}_{volatility_token}",
                as_of_ms,
                last_context,
                {"trend": last_context["trend"], "volatility": last_context["volatility"]},
            )
        )
    else:
        warnings.append("marketRegime state skipped: insufficient volatility history")

    # RSI zone vs the contract bands (strict inequalities; equality stays on
    # the pre-cross side per the contract crossingRule).
    rsi_period = int(indicator_params["rsiPeriod"])
    rsi_lower = float(indicator_params["rsiLower"])
    rsi_upper = float(indicator_params["rsiUpper"])
    rsi_now = _rsi(closes, rsi_period)
    if rsi_now is None:
        warnings.append("technicalIndicators RSI state skipped: insufficient bars")
    else:
        if rsi_now < rsi_lower:
            zone_type, zone = "RSI_ZONE_OVERSOLD", "oversold"
        elif rsi_now > rsi_upper:
            zone_type, zone = "RSI_ZONE_OVERBOUGHT", "overbought"
        else:
            zone_type, zone = "RSI_ZONE_NEUTRAL", "neutral"
        conditions.append(
            _state_condition(
                "technicalIndicators",
                zone_type,
                as_of_ms,
                last_context,
                {
                    "rsi": rsi_now,
                    "rsiPeriod": rsi_period,
                    "rsiLower": rsi_lower,
                    "rsiUpper": rsi_upper,
                    "zone": zone,
                },
            )
        )

    # Close vs SMA(50) over the same finite window the detector uses.
    sma_period = int(indicator_params["smaPeriod"])
    if len(closes) < sma_period:
        warnings.append("technicalIndicators SMA state skipped: insufficient bars")
    else:
        sma_now = sum(closes[-sma_period:]) / sma_period
        close_now = closes[-1]
        if close_now > sma_now:
            event_type = "CLOSE_ABOVE_SMA50"
        elif close_now < sma_now:
            event_type = "CLOSE_BELOW_SMA50"
        else:
            event_type = "CLOSE_AT_SMA50"
        conditions.append(
            _state_condition(
                "technicalIndicators",
                event_type,
                as_of_ms,
                last_context,
                {"close": close_now, "smaPeriod": sma_period, "sma": sma_now},
            )
        )

    # EMA fast vs slow seeded on the finite emaSourceBars window.
    ema_source_bars = int(indicator_params["emaSourceBars"])
    if len(closes) < ema_source_bars:
        warnings.append("technicalIndicators EMA state skipped: insufficient bars")
    else:
        source = closes[-ema_source_bars:]
        fast = _ema(source, int(indicator_params["emaFast"]))
        slow = _ema(source, int(indicator_params["emaSlow"]))
        if fast > slow:
            event_type = "EMA_FAST_ABOVE_SLOW"
        elif fast < slow:
            event_type = "EMA_FAST_BELOW_SLOW"
        else:
            event_type = "EMA_FAST_EQUALS_SLOW"
        conditions.append(
            _state_condition(
                "technicalIndicators",
                event_type,
                as_of_ms,
                last_context,
                {
                    "emaFastPeriod": int(indicator_params["emaFast"]),
                    "emaSlowPeriod": int(indicator_params["emaSlow"]),
                    "emaSourceBars": ema_source_bars,
                    "emaFast": fast,
                    "emaSlow": slow,
                },
            )
        )

    # Close vs POC/VAH/VAL from _profile over the last windowBars bars.
    profile_bars = int(profile_params["windowBars"])
    bins = int(profile_params["bins"])
    fraction = float(profile_params["valueAreaFraction"])
    if len(window) < profile_bars:
        warnings.append("volumeProfile state skipped: insufficient bars")
    else:
        profile = _profile(window, len(window) - profile_bars, len(window), bins, fraction)
        if profile is None:
            warnings.append("volumeProfile state skipped: degenerate profile")
        else:
            poc, vah, val = profile
            close_now = closes[-1]
            for label, level in (("POC", poc), ("VAH", vah), ("VAL", val)):
                if close_now > level:
                    event_type = f"VOLUME_PROFILE_CLOSE_ABOVE_{label}"
                elif close_now < level:
                    event_type = f"VOLUME_PROFILE_CLOSE_BELOW_{label}"
                else:
                    event_type = f"VOLUME_PROFILE_CLOSE_AT_{label}"
                conditions.append(
                    _state_condition(
                        "volumeProfile",
                        event_type,
                        as_of_ms,
                        last_context,
                        {
                            "close": close_now,
                            "poc": poc,
                            "vah": vah,
                            "val": val,
                            "level": label,
                            "windowBars": profile_bars,
                            "bins": bins,
                            "valueAreaFraction": fraction,
                        },
                    )
                )
    return conditions


def build_conditions_payload(
    timeframe: str,
    kline_rows: Sequence[Mapping[str, Any]],
    causal_smc_rows: Sequence[Mapping[str, Any]] = (),
    causal_smc_columns: Sequence[str] = (),
    *,
    now_ms: int,
    causal_smc_error: str | None = None,
    window_bars: int = WINDOW_BARS,
) -> dict[str, Any]:
    """Pure computation of the current-conditions payload (no generatedAtMs).

    ``kline_rows`` is any superset of the latest contiguous finalized run —
    the function re-applies finalization, exact-duration and contiguity rules,
    so fixture rows and database rows share one code path.
    """
    if timeframe not in TIMEFRAME_MS:
        raise UnknownTimeframeError(f"timeframe must be one of {sorted(TIMEFRAME_MS)}")
    interval = TIMEFRAME_MS[timeframe]
    warnings: list[str] = []
    unavailable: list[dict[str, str]] = []

    # Keep only finalized, exact-duration bars and take the latest contiguous
    # run ending at the newest finalized bar; stop at the first gap.
    finalized = sorted(
        (
            row
            for row in kline_rows
            if int(row["CloseTimeMs"]) <= now_ms
            and int(row["CloseTimeMs"]) - int(row["OpenTimeMs"]) + 1 == interval
        ),
        key=lambda row: int(row["OpenTimeMs"]),
    )
    if not finalized:
        raise ConditionsUnavailableError(
            f"no finalized contiguous Klines for {SYMBOL} {timeframe}"
        )
    run_start = len(finalized) - 1
    while (
        run_start > 0
        and int(finalized[run_start]["OpenTimeMs"]) - int(finalized[run_start - 1]["OpenTimeMs"])
        == interval
    ):
        run_start -= 1
    run = finalized[run_start:]
    window = run[-window_bars:]
    first_open = int(window[0]["OpenTimeMs"])
    as_of_ms = int(window[-1]["CloseTimeMs"])

    if len(run) > len(window) or run_start > 0:
        warnings.append("fibonacci-anchor-may-predate-window")
    if run_start > 0:
        warnings.append("analysis-window-truncated-at-non-contiguous-gap")

    contexts = _causal_contexts(window, interval)
    events, metadata = build_causal_technical_events(timeframe, interval, window, contexts)
    contract_meta = metadata["contract"]

    conditions: list[dict[str, Any]] = []
    for event in events:
        # Equality — a plain <= would leak events decided on older bars.
        if int(event["availableTimeMs"]) == as_of_ms:
            conditions.append(_event_condition(event, "triggeredOnBar"))

    conditions.extend(_state_conditions(window, contexts, metadata, as_of_ms, warnings))

    leg_events = [
        event
        for event in events
        if str(event["eventType"]).startswith("FIBONACCI_LEG_")
        and int(event["availableTimeMs"]) <= as_of_ms
    ]
    if leg_events:
        operative = max(
            leg_events, key=lambda e: (int(e["availableTimeMs"]), str(e["eventId"]))
        )
        conditions.append(_event_condition(operative, "operativeLeg"))

    # causalSmc: canonical schema gate + hash verification, then lifecycle.
    if causal_smc_error:
        unavailable.append({"module": "causalSmc", "reason": causal_smc_error})
    elif not CANONICAL_SMC_COLUMNS.issubset(set(causal_smc_columns)):
        unavailable.append(
            {
                "module": "causalSmc",
                "reason": "CausalSmartMoneyEvents canonical schema is unavailable",
            }
        )
    else:
        try:
            verified = _verified_causal_smc(causal_smc_rows, as_of_ms)
        except ValueError as exc:
            unavailable.append({"module": "causalSmc", "reason": str(exc)})
        else:
            candles = [
                Candle(
                    int(row["OpenTimeMs"]),
                    int(row["CloseTimeMs"]),
                    float(row["Open"]),
                    float(row["High"]),
                    float(row["Low"]),
                    float(row["Close"]),
                    float(row.get("Volume") or 0.0),
                    int(row["CloseTimeMs"]),
                    contexts[index],
                )
                for index, row in enumerate(window)
            ]
            close_to_index = {candle.close_ms: index for index, candle in enumerate(candles)}
            context_by_close = {
                candle.close_ms: contexts[index] for index, candle in enumerate(candles)
            }
            level_index = _LifecycleLevelIndex(candles, interval)
            close_times = [candle.close_ms for candle in candles]
            zone_conditions: list[dict[str, Any]] = []
            for row, evidence, source_times in verified:
                event_type = str(row["EventType"])
                available = int(row["AvailableTimeMs"])
                if available == as_of_ms:
                    conditions.append(
                        _condition(
                            module="causalSmc",
                            event_type=event_type,
                            kind="triggeredOnBar",
                            event_id=str(row["EventId"]),
                            formed_ms=int(row["OriginTimeMs"]),
                            available_ms=available,
                            context=context_by_close.get(available, {}),
                            details={"sourceCandleOpenTimesMs": source_times},
                        )
                    )
                if event_type not in _FVG_EVENT_TYPES:
                    continue
                lifecycle = _smc_lifecycle(evidence)
                if lifecycle is None:
                    continue
                mitigation = lifecycle["mitigationLevel"]
                zone_low = float(evidence["lowPrice"])
                zone_high = float(evidence["highPrice"])
                price = float(mitigation["price"])
                operator = str(mitigation["operator"])
                decision_index = close_to_index.get(available)
                details: dict[str, Any] = {
                    "semanticsVersion": lifecycle["semanticsVersion"],
                    "zoneLowPrice": zone_low,
                    "zoneHighPrice": zone_high,
                    "mitigationLevel": dict(mitigation),
                }
                if decision_index is not None:
                    hit = level_index.first_crossing(decision_index, operator, price, as_of_ms)
                    post_decision = len(candles) - 1 - decision_index
                    if hit is not None:
                        continue  # lifecycle terminated before asOfMs
                    details["postDecisionBarsObserved"] = post_decision
                    details["decisionPredatesWindow"] = False
                    if post_decision == 0:
                        details["mitigationObservation"] = (
                            "no post-decision finalized candle observed yet"
                        )
                    else:
                        details["mitigationObservation"] = (
                            "no post-decision crossing observed in window"
                        )
                else:
                    # Zone decision candle is not in the window; every in-window
                    # candle with close > AvailableTimeMs is still post-decision.
                    start_index = bisect_right(close_times, available)
                    mitigated = False
                    for index in range(start_index, len(candles)):
                        candle = candles[index]
                        if (operator == "at_or_below" and candle.low <= price) or (
                            operator == "at_or_above" and candle.high >= price
                        ):
                            mitigated = True
                            break
                    if mitigated:
                        continue
                    details["postDecisionBarsObserved"] = len(candles) - start_index
                    details["decisionPredatesWindow"] = True
                    details["mitigationObservation"] = (
                        "decision candle predates analysis window; "
                        "mitigation before firstOpenTimeMs is unobservable"
                    )
                zone_conditions.append(
                    _condition(
                        module="causalSmc",
                        event_type=event_type,
                        kind="activeZone",
                        event_id=str(row["EventId"]),
                        formed_ms=int(row["OriginTimeMs"]),
                        available_ms=available,
                        context=context_by_close.get(available, {}),
                        details=details,
                    )
                )
            # Zones far from the current price can stay "unmitigated" for years;
            # cap the emitted list at the zones nearest to the as-of close while
            # reporting the honest total so nothing is silently dropped.
            close_now = float(window[-1]["Close"])
            for event_type in _FVG_EVENT_TYPES:
                zones = [z for z in zone_conditions if z["eventType"] == event_type]
                if not zones:
                    continue
                zones.sort(
                    key=lambda z: (
                        abs(
                            (z["details"]["zoneLowPrice"] + z["details"]["zoneHighPrice"]) / 2
                            - close_now
                        ),
                        str(z["eventId"]),
                    )
                )
                kept = zones[:ACTIVE_ZONE_LIMIT_PER_TYPE]
                for zone in kept:
                    zone["details"]["unmitigatedTotal"] = len(zones)
                    zone["details"]["zoneOrdering"] = "nearest-to-asof-close"
                conditions.extend(kept)
                if len(zones) > len(kept):
                    warnings.append(
                        f"{event_type}: {len(zones)} unmitigated zones; "
                        f"showing {len(kept)} nearest to the as-of close"
                    )

    return {
        "schemaVersion": SCHEMA_VERSION,
        "symbol": SYMBOL,
        "timeframe": timeframe,
        "asOfMs": as_of_ms,
        "latestClosedBar": {
            "openTimeMs": int(window[-1]["OpenTimeMs"]),
            "closeTimeMs": int(window[-1]["CloseTimeMs"]),
            "open": float(window[-1]["Open"]),
            "high": float(window[-1]["High"]),
            "low": float(window[-1]["Low"]),
            "close": float(window[-1]["Close"]),
            "volume": float(window[-1].get("Volume") or 0.0),
        },
        "analysisWindow": {
            "firstOpenTimeMs": first_open,
            "bars": len(window),
            "contiguous": True,
            "windowBars": window_bars,
        },
        "contract": {
            "contractVersion": contract_meta["contractVersion"],
            "definitionsSha256": contract_meta["definitionsSha256"],
            "rawFileSha256": contract_meta["rawFileSha256"],
        },
        "conditions": conditions,
        "warnings": warnings,
        "unavailableModules": unavailable,
    }


# In-process cache keyed by (timeframe, asOfMs); a newly closed bar changes the
# key and therefore invalidates automatically.
_PAYLOAD_CACHE: dict[tuple[str, int], dict[str, Any]] = {}
_CACHE_MAX_ENTRIES = 16


def clear_cache() -> None:
    _PAYLOAD_CACHE.clear()


def get_current_conditions(timeframe: str, *, now_ms: int | None = None) -> dict[str, Any]:
    """Contract payload for ``GET /api/current-conditions``."""
    normalized = str(timeframe).strip().lower()
    if normalized not in TIMEFRAME_MS:
        raise UnknownTimeframeError(
            f"timeframe must be one of {sorted(TIMEFRAME_MS)}"
        )
    now = _now_ms() if now_ms is None else int(now_ms)
    try:
        klines, causal_rows, causal_columns, causal_error = _load_inputs(normalized, now)
    except Exception as exc:
        raise ConditionsUnavailableError(
            f"klines unavailable for {SYMBOL} {normalized}: {type(exc).__name__}"
        ) from exc
    if not klines:
        raise ConditionsUnavailableError(
            f"no finalized contiguous Klines for {SYMBOL} {normalized}"
        )
    as_of_ms = int(klines[-1]["CloseTimeMs"])
    key = (normalized, as_of_ms)
    cached = _PAYLOAD_CACHE.get(key)
    if cached is not None:
        return {**cached, "generatedAtMs": now}
    payload = build_conditions_payload(
        normalized,
        klines,
        causal_rows,
        causal_columns,
        now_ms=now,
        causal_smc_error=causal_error,
    )
    if len(_PAYLOAD_CACHE) >= _CACHE_MAX_ENTRIES:
        _PAYLOAD_CACHE.pop(next(iter(_PAYLOAD_CACHE)))
    _PAYLOAD_CACHE[key] = payload
    return {**payload, "generatedAtMs": now}


__all__ = [
    "CANONICAL_SMC_COLUMNS",
    "ConditionsUnavailableError",
    "SCHEMA_VERSION",
    "SYMBOL",
    "TIMEFRAME_MS",
    "UnknownTimeframeError",
    "WINDOW_BARS",
    "build_conditions_payload",
    "clear_cache",
    "get_current_conditions",
]
