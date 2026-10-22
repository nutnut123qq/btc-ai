# Journal note — `POST /api/equities/{SYMBOL}/research-notes`

Recorded with `research_snapshot.py journal record --symbol SYM --file note.json --snapshot export\snapshot.json`
(AdminGuard: needs `EQX_ADMIN_KEY`). Notes are **immutable** after insert. Shape as implemented in
`backend/Services/Models/EquityResearchSnapshotModels.cs` (`EquityResearchNoteRequest`) and validated by
`backend/Services/EquityResearchNoteService.cs` (2026-10-18).

## Request (`note.json`)

**Omit** `requestedAsOf`, `signalBarOpenTimeMs`, `snapshotId` and `snapshotSha256`: `--snapshot` fills
them from the export (it only sets keys that are absent — a placeholder value would be sent as-is and
rejected).

```json
{
  "mode": "forward",
  "horizonSessions": 20,
  "skillVersion": "investment-research-v1.2",
  "methodVersion": "investment-research-method-v1",
  "direction": "sideways",
  "actionView": "wait",
  "actionBasis": "market-only",
  "body": {
    "schemaVersion": "equity-research-note-v1",
    "thesis": "one paragraph, <= 4000 chars",
    "evidence": [{ "claim": "…", "evidenceIds": ["snapshot:…", "script:rs.py#excess20"] }],
    "counterEvidence": ["plain string", { "claim": "…", "evidenceIds": ["…"] }],
    "scenarios": [
      { "name": "base", "horizonSessions": 20, "description": "…", "invalidation": ["daily close < 38,500"] }
    ],
    "limitations": ["prices latest-adjusted; quant abstained"],
    "missingData": ["NPL/COVERAGE/CAR no PIT source"],
    "outcomeCheck": { "how": "R = close(bar signal+20 sessions)/signal close - 1 (adjusted). Convention tb-p5-s4-h20: R >= +5% up, R <= -4% down, else sideways; also record which scenario invalidation triggered first", "dueAfterSessions": 20 },
    "probability": null
  }
}
```

Enums: `mode` forward|retrospective · `direction` up|down|sideways|no-view · `actionView`
buy-more|hold|reduce|sell|wait|none · `actionBasis` position-aware|market-only|insufficient-input
(`insufficient-input` ⇒ `actionView: "none"`). `horizonSessions` 1..250. `body` keys outside the list
above are rejected; `thesis`, `evidence`, `counterEvidence`, `scenarios`, `limitations`, `missingData`,
`outcomeCheck` are required.

## Forward vs retrospective

| | forward | retrospective |
|---|---|---|
| signal bar | must be the latest expected finalized 1d bar, stored | any past as-of |
| time window | before the next session's 09:00 open | any time |
| scored as forecast | yes, if it carries a probability | never |
| errors | 409 `SIGNAL_BAR_NOT_LATEST`, `FORECAST_WINDOW_CLOSED`, `CALENDAR_COVERAGE_EXPIRED` | — |

`CALENDAR_COVERAGE_EXPIRED`: the HOSE calendar file no longer covers the next session, so the
forecast window cannot be determined. A forward note cannot be recorded until the calendar is
extended with the official notice. Record the note as `retrospective`, or stop and say so. Never
edit the calendar or any other data to get it accepted.

Both: the backend rebuilds the snapshot for (symbol, requestedAsOf); id + sha must match or 409
`SNAPSHOT_MISMATCH` (data changed since export → re-export, redo affected numbers, re-record).

## Probability (optional, rare)

```json
"probability": {
  "configId": "tb-p5-s4-h20",
  "probTargetFirst": 0.42,
  "probStopFirst": 0.38,
  "method": "beta-binomial on own-history triple-barrier hits, 2016-2026, non-overlapping",
  "methodRef": "data/research-sessions/BID/20261018-1530-forward/scripts/barrier_rates.py"
}
```

Accepted only with non-empty `method` and `methodRef` (else 400 `PROBABILITY_METHOD_REQUIRED`);
`configId` ∈ {`tb-p5-s4-h20` (+5%/−4%, 20 sessions), `tb-p10-s6-h60` (+10%/−6%, 60 sessions)};
`probTargetFirst + probStopFirst ≤ 1`. A forward note with a probability creates a `session-research`
forecast record (`Validated = false`) scored later like quant/LLM rows. Only include one when the script in
`methodRef` exists in the session folder and reproduces the numbers from the export. Otherwise
`"probability": null` — a qualitative scenario set is the honest output.

## Response

`EquityResearchNoteDto`: `noteId` (`session/{SYM}/{signalBarOpenTimeMs}/{shortid}`), `availableTimeMs`
(real write time), `snapshotVerified: true`, `forecastIds[]`. Save it as `journal-response.json` and put
`noteId` in the report.

## Hashes

`snapshotId` and `snapshotSha256` in the note are the snapshot's own content identity: what the
backend re-verifies, and what `--snapshot` fills in. The per-file sha256 values in `manifest.json`
(snapshot.json bytes, series CSVs, portfolio.json) are export integrity hashes; cite them only as file
references. If `SNAPSHOT_MISMATCH` comes back, something served changed since the export (new bar,
rebuilt base-rates artifact, newly registered document or fundamentals row): re-export, redo the
affected numbers and record again.
