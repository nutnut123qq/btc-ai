# BTC technical-event descriptive evidence

`technical_event_descriptive_evidence.py` creates an immutable, reproducible
description of what happened after causally available technical events. It is
restricted to `BTCUSDT` and finalized `1h`, `4h`, or `1d` candles. In addition
to causal SMC rows, it reconstructs indicator crossings, candle shapes, volume
anomalies, market-regime transitions, confirmed Fibonacci legs, rolling volume
profile crossings, and same-close confluence directly from finalized candles.
The byte-identical backend/Python definition is
`contracts/technical-module-contract.json`; bundles record its recursively
canonicalized `definitionsSha256` separately from the raw file hash.

This workflow does not train a model, issue a prediction, choose a signal,
simulate a trade, calculate PnL, or promote an event type. Its report fixes all
of these fields to `false`: `predictiveEvidence`, `probabilityClaim`,
`economicClaim`, and `promotionAllowed`.

## Input contract

The JSON input must contain:

- `symbol`: exactly `BTCUSDT`.
- `timeframe`: `1h`, `4h`, or `1d`.
- dataset `lineage`: non-empty `source`, `sourceVersion`, and a 64-character
  `contentSha256`.
- candles with valid OHLC, exact timeframe duration, `finalized: true`, an
  explicit `availableTimeMs`, and context fields used for matching controls.
- events with a unique `eventId`, `eventType`, `formedTimeMs`,
  `confirmedTimeMs`, `availableTimeMs`, `sourceCandleOpenTimesMs`, context, and
  the same three required lineage fields.

Unknown dataset lineage or candle availability aborts the run. Unknown event
lineage, unavailable source bars, and other event-level problems remain visible
as excluded ledger rows with an exact reason. Data that was unavailable at the
declared cutoff cannot enter an outcome.

An event may additionally declare lifecycle semantics:

```json
{
  "lifecycle": {
    "semanticsVersion": "smc-zone-v1",
    "touchLevel": {"operator": "at_or_below", "price": 61250.0},
    "mitigationLevel": {"operator": "at_or_below", "price": 60500.0},
    "invalidationUnavailableReason": "no validated invalidation rule"
  }
}
```

The two supported operators are `at_or_above` and `at_or_below`; they are
evaluated against each finalized bar's high or low. Events without declared
zone semantics report lifecycle status `unavailable` with a reason. The
evaluator never guesses a touch, mitigation, or invalidation rule from an
event name.

Persisted legacy technical tables are not silently trusted. Their layer audit
is `unavailable` when point-in-time availability and calculation version are
not both contractually guaranteed. The evaluator instead uses the versioned
causal reconstruction and labels the new six-close regime incomparable with
the legacy ADX/ATR/Bollinger regime.

## Measurements and counts

Each retained event has a row-level ledger entry. At 1, 3, and 6 contiguous
bars after the decision close it records:

- elapsed wall-clock milliseconds;
- close-to-close forward return;
- maximum favorable price excursion (`max(high / decisionClose - 1)`);
- maximum adverse price excursion (`min(low / decisionClose - 1)`).

MFE and MAE above are raw upward and downward excursions, not direction-normalized
trade outcomes. No direction, order, fee, or fill is assumed.

The report separately publishes stored, eligible, excluded, and realized counts.
A causally valid event near the cutoff stays eligible while its unrealized
horizons remain `null`; they are never filled with zero. Exclusions are grouped
by reason. Lifecycle coverage and observed time to first touch, mitigation, and
invalidation are also shown per event type when the event contract defines those
states.

## Dependence, controls, and uncertainty

The declared horizons, metrics, event types, and context keys form one published
family. The workflow retains all eligible types; it does not select types from
their observed results.

Every declared sensitivity grid is actually executed one axis at a time,
including its baseline value. For every variant the report publishes stored,
eligible, excluded, and realized counts, overlap exclusions, eligible decision-
time Jaccard versus baseline, and block-bootstrap summaries for return/MFE/MAE
at 1/3/6 bars with mean delta versus baseline. Every variant remains visible;
there is no winner selection or parameter promotion.

To reduce repeated counting, it keeps the earliest event with the same event
type and declared context within `dedupBars` (six by default). It reports how
many candidates this removed and explicitly states that remaining outcome
windows may still overlap. Mean intervals use a chronological moving-block
bootstrap over event rows, so the report never claims independent observations.

For context comparison, each event is paired deterministically with the most
recent unused earlier candle that has exactly the same declared context. Bars
around any event are excluded from the control pool, and the full six-bar
control outcome must end strictly before the event decision
(`controlIndex + 6 < eventDecisionIndex`). The report shows the
event-minus-control difference as a historical description. It is not a
randomized counterfactual or a probability for a current event.

The additive `statisticalEvidence` section is governed by
`contracts/technical-evidence-statistical-spec.json` and emitted according to
`contracts/technical-evidence-statistics.schema.json`. Its hypothesis identity
is `(module,eventType,horizonBars,metric)`. The family is enumerated before
reading the ledger from all module-contract event types plus the spec-declared
causal SMC family. The current 42 module/event identities therefore emit 378
hypotheses per timeframe even when no event was observed. Unknown observed
identities fail closed. Every identity reports paired raw
and standardized effect sizes, a centered two-sided moving-block-bootstrap
p-value, year/regime stability, the maximum non-overlapping window count and
the observations excluded to form that set, plus an event-order
autocorrelation effective-sample diagnostic. Negative, non-significant,
insufficient-pair, and no-pair identities remain present with nullable
statistics; null never means zero.

Bootstrap blocks are fixed-length blocks in chronological event order. This
assumes dependence is predominantly local and paired differences are
sufficiently stationary within an event family. Event-order blocks are not
equal-duration calendar blocks; long memory, sparse regimes, and structural
breaks can invalidate nominal intervals and p-values. Benjamini-Yekutieli
correction covers every testable identity in the timeframe bundle. It tolerates
arbitrary dependence but is conservative. The declared `q=0.05` indicator is
descriptive only and cannot promote a module, threshold, signal, prediction,
or trade.

## Artifacts and verification

One bundle writes five content-addressed files:

1. cutoff-filtered snapshot JSON;
2. row-level ledger JSONL;
3. aggregate report JSON;
4. semantic manifest JSON binding the preceding hashes, sizes, configuration,
   source lineage, evaluator/modules, output schemas, and versioned contracts;
5. runtime sidecar JSON containing interpreter, platform, NumPy, and Git state.

The semantic manifest excludes runtime/environment fields, so identical frozen
input and code/spec bytes have the same semantic manifest across runtimes. The
run index separately binds the runtime sidecar hash to that manifest. A missing,
modified, or cross-bound sidecar fails run-index verification without changing
the definition of the semantic evidence hash.

Existing content-addressed artifacts are reused only when their bytes are
identical. Conflicting overwrite is refused. The verifier checks all hashes and
then reconstructs every derived technical event and its module metadata from
the frozen candles, exact-compares that reconstruction with the snapshot, and
rebuilds the complete ledger and report. Stored SMC events receive a separate
lineage check because their upstream table rows are not reconstructed. Changing
an event or outcome and merely updating file hashes therefore still fails
verification.

Build and immediately verify a bundle:

```powershell
.\venv\Scripts\python.exe technical_event_descriptive_evidence.py `
  --input-json .\technical-events-snapshot.json `
  --cutoff-ms 1789948799999 `
  --output-dir .\docs\research\evidence\technical-descriptive
```

The normal repository workflow can build the same input directly from
PostgreSQL in a read-only transaction. It reads finalized `Klines`, derives the
six-close trend and trailing range-volatility context causally, and imports
`smc-causal-v2` rows only at their explicit `AvailableTimeMs`:

```powershell
.\venv\Scripts\python.exe technical_event_descriptive_evidence.py `
  --from-postgresql `
  --timeframe 4h `
  --cutoff-ms 1789948799999 `
  --output-dir .\docs\research\evidence\technical-descriptive
```

The normal scheduled entrypoint is finite and produces all three timeframes in
one lock-protected invocation. It stages and semantically verifies every bundle,
publishes content-addressed artifacts first and manifests last, retains older
bundles, runs the canonical golden semantic fixture, then writes an immutable
three-timeframe run index and atomically replaces the self-hashed
`latest-success.json` pointer as the final discovery operation. Partial or
orphan manifests are not members of the published run. Pipeline status records
the same run-index hash for the Evidence Center API:

```powershell
.\scripts\run_technical_evidence_pipeline.ps1
```

Install the daily Windows task explicitly (the repository never installs it on
import or startup):

```powershell
.\scripts\install_technical_evidence_task.ps1 -DailyAt "02:20"
```

If a process is killed or the machine loses power, the exclusive lock is
deliberately not removed automatically. After inspecting local logs, an admin
can conservatively recover only a lock older than the configured age whose PID
is no longer alive on the same host:

```powershell
.\scripts\recover_technical_evidence_lock.ps1 -MaxLockAgeHours 12
```

The timeframe is part of every snapshot and artifact hash. Legacy SMC rows without explicit availability stay
in the row ledger as `unknown_availability` exclusions. If the required causal
columns or causal rows are absent, `modules.causalSmc` reports `unavailable`.
The loader does not reconstruct availability from `CreatedAtUtc`, `TimeMs`, or
the current mitigation flag.

Kline rows whose stored duration does not exactly match the requested timeframe
are excluded before contexts are calculated. Their count is published under
`modules.finalizedCandles.excludedInvalidDurationRows`; events depending on the
missing interval then fail the normal source/contiguity checks.

For causal FVG rows, first touch at the near boundary and complete-fill
mitigation at the far boundary are derived from stored high/low boundaries under
`smc-causal-fvg-zone-v1`. That backend contract has no validated FVG invalidation
rule, so invalidation remains `null` with an explicit reason. Stored
`IsMitigated`/`MitigatedAtMs` are neither selected nor hashed; lifecycle is
recomputed from finalized candles after the event decision. This prevents a
future mitigation update from changing the artifact for an earlier cutoff.
Swing, BOS, and CHoCH currently report lifecycle unavailable because no universal
touch, mitigation, or invalidation semantics have been declared for them.

For BOS/CHoCH lineage, source candles include the full five-bar reference pivot
(`reference ± 2 bars`) plus the break bar. This mirrors the causal backend replay
and makes the pivot confirmation independently inspectable.

Verify an existing bundle independently:

```powershell
.\venv\Scripts\python.exe technical_event_descriptive_evidence.py `
  --verify-manifest .\docs\research\evidence\technical-descriptive\<sha256>.manifest.json
```

The producer of the input snapshot remains responsible for exporting the
backend's causal event state at `asOf`. In particular, a later mitigation must
not be written into an earlier snapshot. The content hash and event algorithm
version make that producer lineage visible and auditable.
