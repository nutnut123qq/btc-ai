# Forward-paper fill/outcome observer (`ai/paper_observer.py`)

Status: not wired into `paper_trader.py`; scheduled separately through `ops/run-ai-job.ps1 -Job PaperObserver` (see *Scheduling*).
Version tag: `forward-paper-observer-v1`.

The forward-paper recorder (`forward_paper_recorder.py`, driven by
`paper_trader.py --mode forward-paper`) writes one immutable decision row per
finalized BTCUSDT 4h signal bar into `PaperObservations`, leaving
`FillPrice/FillObservedAtUtc/FillSource` and
`OutcomeReturn/OutcomeObservedAtUtc/OutcomeHorizon` NULL. This observer is the
"separate timestamped observer" that module refers to: it fills those columns
**only from a live quote it actually observed inside a bounded window**, and
leaves them NULL forever otherwise.

What it is not: not an exchange execution, not an order, no trading keys, no
private endpoints. It is a *simulated paper fill* priced from a public quote.

## Schema reuse

No new table, column or migration. The observer writes only the six nullable
fill/outcome columns that already exist. The database already enforces:

- `CK_PaperObservations_FillCoherence`: a fill only on `long`/`short`, price > 0,
  `FillObservedAtUtc >= ObservedAtUtc`, non-blank `FillSource`.
- `CK_PaperObservations_OutcomeCoherence`: an outcome only after a fill, with
  `OutcomeObservedAtUtc >= FillObservedAtUtc` and non-blank `OutcomeHorizon`.
- Trigger `prevent_paper_observation_rewrite`: decision columns immutable, fill
  and outcome write-once, no DELETE.

The observer adds its own guards in the `WHERE` clause of each UPDATE so that a
violating write is a 0-row no-op rather than an exception; the constraints and
trigger remain the backstop.

## Semantics

### Eligibility

- `Decision IN ('long','short')` and `FillPrice IS NULL` → fill candidate.
- `FillPrice IS NOT NULL` and `OutcomeReturn IS NULL` → outcome candidate.
- `abstain` rows (every production row today: the model is quarantined, so all
  123 production rows are `abstain/model-unavailable`) are never selected and
  never get a fill or PnL. With no candidates the observer does not even fetch
  a quote.
- Eligibility depends only on the recorded decision, not on the model's
  *current* status. A directional decision taken while a model was active is
  still filled and scored if that model is quarantined later: dropping it would
  discard prospective evidence and bias the journal toward models that are
  still in favor.

### Quote

- Source: `forward_paper_recorder.fetch_binance_spot_quote` (Binance spot
  `bookTicker`, public, no key). `LiveQuote` carries only the **bid/ask mid**
  (`source="binance-spot-bookTicker-bid-ask-mid"`), not bid and ask separately,
  so both long and short are priced at the mid. The half-spread is not modeled
  in the fill price; it belongs to the per-side slippage of the cost spec (see
  Net return). Reusing the recorder's fetcher keeps fill price on the same
  basis as the decision-time `QuotePrice`.
- One quote is fetched per run, after the candidate rows are read. The clock is
  read again after the fetch (`now`). The quote is rejected — and nothing is
  written — when:
  - the fetch raises (network, HTTP, invalid book);
  - `price <= 0`, or `received_at_utc` is naive, or `received_time_ms`
    disagrees with `received_at_utc` by more than 1 ms;
  - it is stale: `now - received_time_ms > 5 s` (`QUOTE_MAX_AGE_MS`);
  - it is from the future: `received_time_ms > now + 1 s`.
- All window checks below use the **quote's received time**, not the run's
  start time, so a slow fetch cannot push a quote outside its window unnoticed.

### Fill

- Window: `AvailableTimeMs <= quoteMs <= AvailableTimeMs + 15 min`
  (`FILL_WINDOW_MS`), and also `quoteReceivedAtUtc >= ObservedAtUtc` (the
  constraint compares timestamps; `AvailableTimeMs` is the millisecond floor
  of `ObservedAtUtc`). 15 minutes matches the `BTCPaperTrader` 15-minute
  cadence: a poll in the next cadence slot after the decision can fill it.
- Written values: `FillPrice = quote.price`, `FillObservedAtUtc =
  quote.received_at_utc`, `FillSource =
  "<quote.source>|receivedTimeMs=<ms>|observer=forward-paper-observer-v1"`.
- Outside the window → never filled. No backfill after downtime, no catch-up
  from stored data. The row stays NULL, which the journal reads as "fill not
  observed" (unavailable), not as a zero return.

### Outcome

- Horizon: 4h from the **fill**. Stored as `OutcomeHorizon = '4h-gross'` so
  the row itself states that the return is gross of costs.
- Window: `FillObservedAtUtc + 4h <= quoteAt <= FillObservedAtUtc + 4h + 15 min`.
- `OutcomeReturn = side * (exitMid / FillPrice - 1)`, `side = +1` long,
  `-1` short. `OutcomeObservedAtUtc = exit quote received_at_utc`.
- This is a **gross** directional return. **Costs are NOT deducted.**
- Missed window → `OutcomeReturn` NULL forever (unavailable, not zero).
- The exit price is not stored as its own column (no column exists); it is
  recoverable as `FillPrice * (1 + side * OutcomeReturn)` up to float
  precision. The exit quote uses the same fetcher/source as the fill, recorded
  in `FillSource`.

### Net return (derived, not stored)

`ConfigProvenanceJson.costSpec` on each row records
`{feePerSideBps, slippagePerSideBps, roundTripBps}` (currently 10/5/30). A
reader derives net with the same model as backtests,
`execution_engine.calculate_round_trip(side, entry_reference_price=FillPrice,
exit_reference_price=FillPrice*(1+side*OutcomeReturn), costs=ExecutionCostSpec(fee, slip),
contract=PAPER_EXECUTION_CONTRACT).net_return`. To first order,
`net ≈ OutcomeReturn - roundTripBps/10000`.

### Lookahead rules

- Never reads `Klines`, the next bar's open, or any stored bar as a fill/exit.
- Never uses information later than the observation: each write is priced by a
  quote received at the moment of the run, and the run only writes when that
  moment falls inside the row's window.
- Does not touch decision columns (the UPDATE sets only fill or only outcome
  columns; the trigger also rejects anything else).

### Server-clock guard (no backdating)

The Python freshness checks trust the process clock, so the database clock is
the authority. Both UPDATEs additionally require

```sql
quoteReceivedAtUtc BETWEEN NOW() - INTERVAL '10000 milliseconds'
                       AND NOW() + INTERVAL '1000 milliseconds'
```

so a caller with a fake clock, or a `LiveQuote` rebuilt from a stored close,
cannot backfill a fill or outcome after downtime: the write matches 0 rows.
`NOW()` is the UPDATE's transaction start; the observer commits after its
SELECT and after each UPDATE, so each UPDATE gets a fresh `NOW()`. If the
observer host's clock runs more than 1 s ahead of the database, writes are
refused (fails closed). The public `observe_once(conn)` takes no clock or
fetcher (real clock + `fetch_binance_spot_quote` only); `_observe_once` with
injectable clock/fetcher exists for tests and is still bounded by the guard.

### Idempotency and concurrency

Each write is a conditional UPDATE:

```sql
UPDATE "PaperObservations" SET fill...
WHERE "Id" = %s AND "Decision" IN ('long','short') AND "FillPrice" IS NULL
  AND <window and chronology guards>
```

(and `"OutcomeReturn" IS NULL AND "FillPrice" IS NOT NULL ...` for outcomes).
A retry after a successful write matches 0 rows. Two concurrent runs: under
PostgreSQL READ COMMITTED the second UPDATE blocks on the row lock, then
re-checks `FillPrice IS NULL` against the committed row and matches 0 rows, so
exactly one fill wins and is never overwritten. Each UPDATE is committed on its
own.

## Tests

`tests/unit/test_paper_observer.py`: pure planner/quote tests with a fake
clock and fetcher (including exact window boundaries), plus PostgreSQL tests
on a per-test temporary `btc_observer_test_<hex>` database on localhost that
apply the recorder `SCHEMA_SQL` (constraints + trigger). Because writes are
bounded by `NOW()`, PG tests use the real clock and place rows in the past via
an explicit `CreatedAtUtc` (keeping `CK_PaperObservations_ProspectiveClock`
satisfied). They cover backdated/future quotes and every window guard by
executing the UPDATE SQL directly, bypassing the planner. They skip if
psycopg2 or local PostgreSQL is unavailable.

## CLI

```
python paper_observer.py --once
```

Runs one observation pass against the database from `db_config`
(`ai/.env`, overridden by the ops secrets in the environment), prints one JSON
line (`ranAtUtc`, `candidates`, `fills`, `outcomes`, `quote`), and exits 0. If the
`PaperObservations` table does not exist it does nothing (it never creates
schema). Any Python/DB error prints one JSON line with `error`/`message` on
stdout (not a stderr traceback, so the ops wrapper logs it) and exits 1.

## Scheduling

- Entry point: `backend/ops/run-ai-job.ps1 -Job PaperObserver` (secrets via
  `Import-OpsSecrets`, log `.ops/logs/paperobserver-job.log`, exit code passed
  to Task Scheduler).
- Task: `backend/ops/install-paper-observer-task.ps1 [-Enable]` registers
  `BTCPaperObserver` every 5 minutes (≤ the 15-minute fill/outcome windows),
  `MultipleInstances=IgnoreNew`, 4-minute time limit, same interactive
  deploy-user principal as the other application tasks. It is separate from
  `BTCPaperTrader`/`BTC_Paper_Trading_Runner`, whose cadence and action are unchanged.
- Rollback: `Disable-ScheduledTask BTCPaperObserver`. Written fills/outcomes are
  write-once journal entries and are never deleted.
- Operational status (installed/enabled/verified) is tracked in `TASKS.md` and
  the deployment evidence, not here.

## Notes

- The backend `PaperObservationReader` already returns `FillPrice`,
  `FillObservedAtUtc`, `OutcomeReturn`, `OutcomeObservedAtUtc`,
  `OutcomeHorizon`; it does not return `FillSource`.
- In production today the observer would write nothing (no directional rows).
