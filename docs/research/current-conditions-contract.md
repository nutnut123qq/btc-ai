# Contract — Current Technical Conditions + Evidence Linkage

> Frozen contract for Đ1. Workers MUST implement exactly these shapes.
> Violations of semantics = rework. Owner: coordinator.

## 1. Semantics (do not renegotiate)

- `asOfMs` = `CloseTimeMs` of the **latest finalized** contiguous bar for the
  timeframe. A bar is finalized when `CloseTimeMs <= now`. Never use the
  in-flight bar.
- `kind` classes per condition:
  - `triggeredOnBar`: event whose `availableTimeMs == asOfMs`
    (`availableTimeMs <= asOfMs` is NOT sufficient — it must equal the latest
    closed bar's close time).
  - `state`: a persistent condition that holds at `asOfMs`, evaluated by
    re-running the contract detector state functions on the analysis window
    (e.g., current regime pair, RSI zone, close vs SMA50/POC/VAH/VAL,
    EMA fast vs slow). Not "latest event row".
  - `activeZone`: a previously emitted zone event whose lifecycle is not
    terminated at `asOfMs`. For FVG (`smc-causal-fvg-zone-v1`): mitigation =
    **first candle `Low <= zone low` (FVG_BULL) / `High >= zone high`
    (FVG_BEAR)** — wick/extreme crossing per `_LifecycleLevelIndex`, NOT close.
  - `operativeLeg`: most recent `FIBONACCI_LEG_*` event with
    `availableTimeMs <= asOfMs` (the leg remains operative until an opposite
    confirmed leg supersedes it — it is not bound to the latest bar).
- Direction uses the same token mapping as the contract `confluence`
  `voteMapping`: bullish = BULL, ABOVE, EXIT_OVERSOLD, ENTER_OVERSOLD;
  bearish = BEAR, BELOW, EXIT_OVERBOUGHT, ENTER_OVERBOUGHT; else neutral.
  Reuse `technical_event_modules._direction` — do not re-implement.
- Detectors: reuse `build_causal_technical_events`, `_causal_contexts`,
  `_indicator_events` state helpers (`_ema`, `_rsi`), `_profile`,
  `technical_event_descriptive_evidence` lifecycle semantics. NO new formulas.
- causalSmc: read `CausalSmartMoneyEvents` (same canonical hash verification
  as the pipeline: `DecisionEvidenceSha256` must match
  `sha256(DecisionEvidenceJson)`; a mismatch = whole module unavailable with
  reason, never silently skip the row).
- Analysis window: last **2000** finalized contiguous bars ending at `asOfMs`
  (fibonacci anchor may predate the window — declare it in `warnings`
  when the earliest window bar is a potential pivot source). All bars in the
  window MUST be contiguous (`OpenTimeMs` deltas == interval); stop the window
  at the first gap.
- No scoring, voting aggregation into a verdict, probabilities, BUY/SELL,
  or predictive claims anywhere in either service or UI copy.

## 2. AI endpoint

`GET http://AI/api/current-conditions?timeframe=1h|4h|1d` → 200:

```json
{
  "schemaVersion": "current-conditions-v1",
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "asOfMs": 0,
  "latestClosedBar": {"openTimeMs":0,"closeTimeMs":0,"open":0,"high":0,"low":0,"close":0,"volume":0},
  "generatedAtMs": 0,
  "analysisWindow": {"firstOpenTimeMs":0,"bars":0,"contiguous":true,"windowBars":2000},
  "contract": {"contractVersion":"...","definitionsSha256":"...","rawFileSha256":"..."},
  "conditions": [
    {
      "module": "technicalIndicators",
      "eventType": "RSI_ENTER_OVERSOLD",
      "kind": "triggeredOnBar|state|activeZone|operativeLeg",
      "direction": "bullish|bearish|neutral",
      "eventId": "...",                    // null for pure state conditions
      "formedTimeMs": 0,
      "availableTimeMs": 0,                // state conditions: == asOfMs
      "context": {"trend":"up","volatility":"normal"},
      "details": { }                       // optional, e.g. FVG zone bounds,
                                         // RSI value, POC/VAH/VAL levels
    }
  ],
  "warnings": ["..."],
  "unavailableModules": [{"module":"causalSmc","reason":"..."}]
}
```

- Deterministic: same DB state → identical payload except `generatedAtMs`.
- Cache per `(timeframe, asOfMs)` in-process; a new closed bar invalidates.
- Errors: 400 for bad timeframe; 503 `{detail}` when klines unavailable.
- Compute budget: single bounded window, no bootstrap/statistics — must be
  well under a second after warm; measure in the test run and report actual ms.

## 3. Backend endpoint

`GET /api/research/current-conditions?timeframe=1h|4h|1d` → 200:

```json
{
  "timeframe": "1h",
  "asOfMs": 0,
  "conditions": [ /* AI conditions verbatim, plus */ ],
  // each condition gains:
  //   "evidence": {
  //     "1": {"forwardReturn": {...cell}, "mfe": {...cell}, "mae": {...cell}},
  //     "3": {...}, "6": {...}
  //   },
  // cell = {"tested":true,"rawP":..,"adjustedQValue":..,"passesDeclaredFdr":..,
  //         "sufficientSample":..,"nonOverlappingPairs":..,"effect":..,
  //         "ciLower":..,"ciUpper":..,"meanPairedDifference":..}
  //   or {"tested":false,"reason":"untested"|"insufficient-sample"|...}
  "evidence": {
    "available": true,
    "reason": null,
    "runId": "...",
    "manifestSha256": "...",
    "specSha256": "...",
    "cutoffMs": 0,
    "evidenceAgeBars": 0
  },
  "conflicts": [
    {"horizon":1,"metric":"forwardReturn",
     "bullish":["module:EVT"],"bearish":["module:EVT"]}
  ],
  "warnings": ["..."],
  "generatedAtMs": 0
}
```

Join keys: `(module, eventType, horizon, metric)` against the current verified
report for the timeframe (via `ResearchEvidenceCatalog` already-loaded
artifacts — do not re-parse files).

`evidence` block semantics:
- `available=false` + `reason` when: no verified bundle, hash/semantic
  verification fails, or spec/contract sha mismatch with the conditions
  payload. Staleness alone NEVER makes evidence unavailable —
  `evidenceAgeBars = floor((asOfMs - cutoffMs)/intervalMs)` is informational
  and rendered by the UI as "nghiên cứu cắt tại …".
- `conflicts`: populated only when same `(horizon, metric=forwardReturn)` has
  `passesDeclaredFdr==true` on both a bullish and a bearish condition.

Failure modes (must all be covered by tests):
- AI down/timeout/5xx → 502 envelope `{Code:"CONDITIONS_SOURCE_UNAVAILABLE"}`,
  retryable=true.
- AI payload schema mismatch → 502 `{Code:"CONDITIONS_PAYLOAD_INVALID"}`.
- No verified evidence → 200 with `evidence.available=false` + reason.
- Bad timeframe → 400.

## 4. UI (Evidence Center — inside existing screen, no new dashboard)

- Timeframe selector reuses the existing pattern (BTCUSDT only; tf = 1h/4h/1d).
- Two rows of meta: `asOf` market time + `evidence cutoff` (study age),
  rendered as two separate labeled values — never merged.
- Grouped by module; each condition shows: eventType label, kind badge,
  direction chip, per-horizon evidence cells (effect + CI + q + nonOverlap +
  gate flag); unavailable/insufficient cells say why.
- `triggeredOnBar` conditions get a "mới trên nến đóng gần nhất" badge.
- Conflicts section renders `conflicts` verbatim with explanation text —
  no resolution, no "winner".
- Link per condition → opens existing dossier for that manifest.
- States: loading skeleton, empty ("không có điều kiện nào thỏa trên nến đóng
  gần nhất"), error, success. Copy in Vietnamese, consistent with existing
  panel wording; every claim traceable — no invented percentages.

## 5. Non-goals

- No scheduled task, no new DB tables, no writes, no ModelPredictions.
- No changes to EnsembleService, paper trader, or evidence pipeline.
- Do not touch `Program.cs`, `appsettings*.json`, `TASKS.md`,
  `AGENTS.md` — coordinator owns wiring/shared files.
