# Technical evidence operations protocol

This protocol covers BTCUSDT descriptive technical evidence for `1h`, `4h`,
and `1d`. It does not authorize prediction, parameter promotion, trading, PnL,
or a production-database write.

## Evidence contract

Every bundle freezes finalized candles, canonical causal SMC decision rows, and
the seven reconstructed technical modules. The report exposes:

- exhaustive profiles by module, UTC year, timeframe, and declared
  trend/volatility regime;
- event, candle, horizon, context, lineage, and historical-control missingness;
- unique-module and pairwise same-close overlap plus confluence dependence;
- every fixed sensitivity variant from the contract, including variants with
  negative or unavailable outcomes;
- every `(module,eventType)` family, including zero-sample and non-positive
  families.

The machine contract is `contracts/technical-evidence-profile.schema.json` and
the consumer fixture is `contracts/technical-evidence-profile.example.json`.
`profileDefinitionsSha256` binds the fixed dimensions, horizons, module scope,
and metrics. `thresholdSelectionAllowed`, `outcomeDrivenSelectionAllowed`, and
`rankingOrWinnerSelection` are always false.

The additive statistical output is `report.statisticalEvidence`, version
`btc-technical-evidence-statistics/v1`. Its fixed methodology is in
`contracts/technical-evidence-statistical-spec.json`; its consumer schema and
fixture are `contracts/technical-evidence-statistics.schema.json` and
`contracts/technical-evidence-statistics.example.json`. The null uses a
strict-prior, without-replacement candle matched on trend and volatility whose
maximum-horizon outcome was finalized before the event. Moving blocks follow
chronological event order rather than equal calendar duration and assume local
dependence plus within-family stationarity. Benjamini-Yekutieli corrects the
complete testable family under arbitrary dependence. Q-values are descriptive
diagnostics and never selection or promotion gates.

The statistical family is contract-first, not row-first: all event types in
`technical-module-contract.json` plus the spec-declared causal SMC family are
crossed with all three horizons and metrics. The current 42 identities produce
378 rows per timeframe even for an empty ledger. Zero-event rows retain null
inferential values; an observed identity absent from the declared family aborts
the run.

Canonical SMC evidence comes only from `CausalSmartMoneyEvents`. The loader
verifies the exact decision-evidence bytes against `DecisionEvidenceSha256`,
uses the persisted event id and decision source candles, and gates an empty
result with the versioned rebuild checkpoint. `State` and `MitigatedAtMs` are
latest-known fields and are never copied into an earlier cutoff; lifecycle is
recomputed from frozen candles. `SmartMoneyStructures` remains visible only as
legacy excluded audit input.

## Atomic publication and recovery

One finite run stages and semantically verifies all three bundles. Immutable
artifacts and manifests may be copied first, but readers discover a run only
through the self-hashed `latest-success.json` pointer written last. It names one
content-addressed run index containing exactly `1h`, `4h`, and `1d`. Interrupted
copies are orphans and cannot replace the prior valid trio.

The pipeline lock is exclusive. Scheduled runs never remove a stale lock.
Recovery is an explicit operation that requires the same host, an expired lock,
and a dead PID. A failed run writes a bounded, redacted pipeline status and the
PowerShell scheduler wrapper writes a separate bounded local log/status under
ignored `ai/.ops/technical-evidence/`.

Each run index also binds one content-addressed runtime sidecar per timeframe.
The semantic manifest contains deterministic data/code/spec/schema provenance
only; interpreter, platform, NumPy, and Git state live in the sidecar and are
excluded from the semantic hash. Readers reject a referenced sidecar whose
hash is missing, modified, or bound to another manifest.

Use the production-DB-free harness before deployment:

```powershell
cd D:\code\btc\ai
.\venv\Scripts\python.exe technical_evidence_operational_harness.py `
  --bars 240 --iterations 2 --bootstrap-samples 5 `
  --output-dir docs/research/evidence/technical-operational-harness
```

It injects a second-bundle publication failure, proves the active pointer stays
unchanged, retries the run, copies the evidence directory, verifies the restored
trio, checks repeatable semantic hashes, and enforces a Python allocation cap.
It is not a PostgreSQL capacity test.

## Capacity and reference-aware retention

Snapshots and ledgers are intentionally large. Never delete files by age or
glob. `technical_evidence_retention.py` follows validated run-index → manifest
→ artifact references, protects the active pointer graph, and retains at least
two successful runs. Unknown files are untouched. Planning is read-only:

```powershell
.\venv\Scripts\python.exe technical_evidence_retention.py `
  --output-dir docs/research/evidence/technical-descriptive `
  --keep-successful-runs 2
```

Application requires the exact SHA-256 from a newly reviewed plan:

```powershell
.\venv\Scripts\python.exe technical_evidence_retention.py `
  --output-dir docs/research/evidence/technical-descriptive `
  --keep-successful-runs 2 --apply --confirm-plan-sha256 PLAN_SHA
```

Every candidate is re-hashed immediately before deletion. Retention refuses to
delete anything until the minimum successful-run count exists. The scheduler
runner can perform the same two-phase flow only when explicitly installed with
`-ApplyRetention`; it is off by default.

## Backup and restore verification design

Database backup and evidence-artifact backup are separate gates:

1. The database task must invoke `backend/ops/guarded-backup.ps1`, never the raw
   `backup.ps1`. The guard checks disk headroom, serializes runs, verifies the
   new dump, retains at least two complete sets, and rotates only verified sets.
2. Before migration, verify dump/manifest checksums and `pg_restore --list`.
   Quarterly, perform the backend `Full` restore drill into a unique isolated
   database, reconcile exact public-table row counts, and drop the drill DB in
   `finally`. Never target the configured source database.
3. Copy the complete evidence directory to the backup destination. On restore,
   run `read_verified_success` (or the operational harness) against the copied
   directory. It must resolve the pointer, run index, exact trio, manifests,
   artifact hashes, and semantic recomputation.
4. Record RPO, RTO, backup task exit code, guarded-wrapper status, restore drill
   timestamp, source identity, and artifact run-index SHA. A successful dump
   without a restore drill is not a proven recovery path.

The known local task `Bitcoin Analyst DB Backup` must be updated from its old
raw `backup.ps1` action with the reviewed installer; do not edit Task Scheduler
actions manually:

```powershell
& "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
  -NoProfile -ExecutionPolicy Bypass `
  -File "D:\code\btc\backend\ops\install-backup-task.ps1" `
  -TaskName "Bitcoin Analyst DB Backup" `
  -OutputDirectory "D:\code\btc\backend\.ops\backups" `
  -RetentionCount 2 -MinimumFreeGiB 15 -DailyAt "02:00" -Enable
```

Review the generated action before applying. A bounded smoke test starts the
task once, polls it with a deadline shorter than its execution limit, checks
`LastTaskResult == 0`, then validates the wrapper status JSON and the newest
complete backup set. Do not infer success from Task Scheduler history because
the Operational channel can be disabled.

## Honest limitations

- These profiles are descriptive and dependent; overlapping outcomes are not
  independent trials.
- Historical context matching is not a randomized counterfactual.
- OHLCV volume profile is an approximation, not observed price-by-volume.
- The synthetic harness does not prove production PostgreSQL performance,
  database restore correctness, or disk capacity.
- A checkpoint marked partial/unavailable means absence of stored SMC events
  must not be interpreted as evidence that no events occurred.
