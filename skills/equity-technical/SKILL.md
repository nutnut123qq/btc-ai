---
name: equity-technical
description: Technical-only (candles and indicators) analysis session for the Vietnamese bank equities MBB and BID in the btc repo. It reads one backend payload, GET /api/equities/{symbol}/technical (plus /candles for more history), forms a direction view up/down/sideways/no-view over 20 or 60 sessions without inventing any probability, records it as a forward journal note bound to the payload's snapshot, and later reads how past calls were scored by the backend rule direction-score-v1. Use it when the user asks for a technical, chart, candle or "phân tích kỹ thuật / nến" view of MBB or BID, or asks how earlier technical calls scored. Never reads fundamentals, events, documents or news; for those use investment-research. Do not use it for BTC/crypto or for changing the app's code.
---

# equity-technical (MBB / BID, candles only)

The backend computes every number (candles, indicators, patterns, conditions, base rates, scores).
This session reads them, judges a direction, writes it down once, and lets the backend score it.
It never places an order and never produces a probability.

To start it in a new session (it is not auto-discovered, same convention as `investment-research`):

> Read `D:\code\btc\ai\skills\equity-technical\SKILL.md` and follow it for: <request>.

Backend: env `BACKEND_BASE_URL`, default `http://localhost:5197` (local stack). Shell notes (UTF-8,
paths) are in `../investment-research/references/cli.md`; use only the parts this file names.
Recording and reviewing need env `EQX_ADMIN_KEY`, sent only as header `X-Admin-Key`, never printed or
written. If `[bool]$env:EQX_ADMIN_KEY` is `False`, do the analysis, save `note.json` unrecorded and
say the user must set the key themselves; do not read ops secrets or look for it elsewhere. A 401
means the key is missing or wrong.

```powershell
$base = if ($env:BACKEND_BASE_URL) { $env:BACKEND_BASE_URL } else { "http://localhost:5197" }
$h = @{ "X-Admin-Key" = $env:EQX_ADMIN_KEY }
```

First create the session folder (see "Where things go") and work inside it.

## Scope (hard limits)

- **Symbols:** `MBB` or `BID` only.
- **Endpoints, and nothing else:**
  - `GET /api/equities/{symbol}/technical`: the whole input.
  - `GET /api/equities/{symbol}/candles?timeframe=..&limit=..&endTimeMs=..`: more history (bars with
    `openTimeMs <= endTimeMs`, ascending, `limit` at most 2000).
  - `POST /api/equities/{symbol}/research-notes`: record the note (admin).
  - `GET /api/equities/{symbol}/research-notes/direction-scores`: past calls scored (admin).
- Never read fundamentals, financial facts, events, corporate-action lists, source documents, news,
  the dossier, the full research snapshot or the web. If the user asks about them, say this skill is
  technical-only and point to `investment-research`.
- Do not loop the endpoints; one `technical` read per session (again only after `SNAPSHOT_MISMATCH`).

## Step 1: Read the payload

Save the raw response bytes in the session folder
(`Invoke-WebRequest -UseBasicParsing "$base/api/equities/MBB/technical" -OutFile technical.json`;
`?bars=` sets the window, default 100, at most 500; the call can take close to a minute on a busy
machine), then read these fields by these exact names:

- `timeframes[]`: one entry per timeframe the catalog lists for the symbol. Use whatever timeframes
  are present; a timeframe that is absent has no data, so do not infer it. Each entry has:
  - `candles` (ascending; the last one is the latest finalized bar);
  - `freshness` (`isStale`, `staleRule`, `reason`): a stale timeframe goes into the note's
    `limitations`;
  - `indicators` (latest row) with `indicators.methods`: quote every indicator under the formula named
    there (e.g. Wilder vs Cutler RSI); never assume a formula;
  - `candlePatterns` in the window (the stored recognizer). The `conditions` module `candlePatterns`
    can disagree with it on the same bar; report both when they do;
  - `conditions`: if `conditions.status` is `unavailable`, give `conditions.reason` (no stand-in). The
    usable part is `conditions.payload.conditions[]`, each with `module`, `eventType`, `kind`,
    `direction`, `details`; one `eventType` can appear twice (`kind` = bar trigger vs ongoing state).
    Also read `conditions.payload.warnings`; if `conditions.alignedWithLatestCandle` is `false`, say
    the conditions are not for the latest candle;
  - `dataQuality`: copy every `dataQuality.caveats[].code` with its text into the note's `limitations`
    (e.g. `missing-expected-bars`, `session-open-1h-bars-missing`, `unexpected-session-bars`,
    `below-calendar-coverage`).
- `baseRates` (`status`, `reasons[]`, `configs[]`), `relativeStrength`, `calendar.forwardNoteOpen`,
  `calendar.nextSessionOpenUtcMs`.
- `noteBinding.snapshotId`, `noteBinding.snapshotSha256`, `noteBinding.requestedAsOf`,
  `noteBinding.signalBarOpenTimeMs`: the binding for the note (Step 4). If `noteBinding` is null, give
  `noteBindingUnavailableReason` and do not record.
- `scoring.ruleVersion` (must be `direction-score-v1`; if not, stop and report the mismatch).
- `limitations`: copy them into the note.

Cite payload items as `technical:<field path>`, and array items by their key, e.g.
`technical:timeframes[1d].conditions[REGIME_SIDEWAYS_NORMAL]`.

The signal bar is the latest finalized **1d** bar. Horizons count 1d HOSE sessions; 1h bars are context
only. Prices are latest-adjusted (corporate actions rewrite history): reason in returns, not levels
across an ex-date (`../investment-research/references/data-limitations.md` item 1).

## Step 2: Honesty rules

- **No probabilities.** Quote the backend's numbers with their field path; never turn them, a pattern
  count or a feeling into a probability or a "likely %". `body.probability` is always `null`.
- **Base rates are unvalidated.** While `baseRates.status` is `abstained` with `gate-not-passed` in
  `baseRates.reasons`, the kNN artifact failed its out-of-sample gate, so every base-rate number
  (including configs showing `abstained: false`) is research-only and **unvalidated**. Quote one only
  with that word next to it, and never as a reason on its own.
- **Disagreements stay visible.** Two RSIs or two trend labels that disagree are reported, not merged.
- **Computed numbers come from a script.** Anything not copied from the payload (returns, ranges,
  levels from more candles) comes from a script in `scripts/`, run on the saved JSON, cited as
  `script:<file>#<output>`. Iterate forward in time only; a signal at bar t uses bars <= t.

## Step 3: The judgement

- `direction`: `up | down | sideways | no-view`. `no-view` is honest when the evidence is mixed or the
  data is not usable (stale freshness, missing timeframe, `conditions` unavailable).
- `horizonSessions`: **20 or 60 only**; those are the only horizons the backend scores.
- Scored later by the backend rule `direction-score-v1`: R = close of the bar H sessions after the
  signal bar (HOSE calendar) / signal-bar close - 1, both on the stored adjusted closes; **up** if
  R >= up-threshold, **down** if R <= down-threshold (a negative number), otherwise **sideways**. Both
  `scoring.horizons[]` of `GET technical` and `horizons[]` of `direction-scores` give these as signed
  fractions, `upThresholdPct` 0.05 and `downThresholdPct` -0.04 for H = 20:

  | horizonSessions | up if R >= | down if R <= | barrier config |
  |---|---|---|---|
  | 20 | +5% | -4% | `tb-p5-s4-h20` |
  | 60 | +10% | -6% | `tb-p10-s6-h60` |

  So "up" means you expect R >= +5% (H = 20), not merely a green close. A missing target bar is not
  shifted to another session (`target-bar-missing`).
- `actionView` / `actionBasis`: from the table in `../investment-research/SKILL.md` Step 7. This skill
  reads no position, so `position-aware` never applies: `market-only` with `wait` by default, or
  `hold` only when the user says they already hold the stock; or `insufficient-input` with `none`
  when the view is `no-view` for lack of data.
  For holdings, stop levels, sellable shares or buy capacity, point to the `equity-position` skill.
- Give scenarios with invalidation levels (from the payload or a script) for the same horizon.

## Step 4: Record the note (once)

- `mode: "forward"` only when `calendar.forwardNoteOpen` is `true`: after the signal bar is stored and
  before `calendar.nextSessionOpenUtcMs` (the next session's 09:00 ICT open). Otherwise record it as
  `retrospective` (stored for the record, never scored) unless the user says not to record.
- Top level: `noteBinding` copied verbatim into `snapshotId`, `snapshotSha256`, `requestedAsOf` (a
  JSON string) and `signalBarOpenTimeMs`, plus `skillVersion: "equity-technical-v2"` and
  `methodVersion: "equity-technical-method-v1"`.
- `body` follows `equity-research-note-v1` exactly as in
  `../investment-research/references/journal-note.md` (required keys, enums, limits). Use that file
  for the `body` only: its "omit the binding, use `--snapshot`" instruction is for exported snapshots;
  this skill has no export, so the note carries the binding from `noteBinding`.
  `body.limitations` carries the payload `limitations` and every `dataQuality` caveat;
  `body.missingData` says fundamentals and news were deliberately not read.
  `body.outcomeCheck.how` states the `direction-score-v1` rule and thresholds for the chosen horizon.

Minimal shape (values are placeholders; the binding values come from `noteBinding`):

```json
{
  "mode": "forward",
  "requestedAsOf": "1792315293201",
  "signalBarOpenTimeMs": 1792281600000,
  "snapshotId": "snap/MBB/1792309500000/0123456789ab",
  "snapshotSha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "horizonSessions": 20,
  "skillVersion": "equity-technical-v2",
  "methodVersion": "equity-technical-method-v1",
  "direction": "sideways",
  "actionView": "wait",
  "actionBasis": "market-only",
  "body": {
    "schemaVersion": "equity-research-note-v1",
    "thesis": "technical-only view from GET technical",
    "evidence": [{ "claim": "close above SMA50 on 1d", "evidenceIds": ["technical:timeframes[1d].indicators"] }],
    "counterEvidence": [],
    "scenarios": [{ "name": "base", "horizonSessions": 20, "description": "range", "invalidation": ["1d close < level"] }],
    "limitations": ["base rates unvalidated (gate-not-passed)"],
    "missingData": ["fundamentals and news not read (technical-only skill)"],
    "outcomeCheck": { "how": "direction-score-v1: R = close(signal+20 sessions)/signal close - 1 (adjusted); R >= +5% up, R <= -4% down, else sideways", "dueAfterSessions": 20 },
    "probability": null
  }
}
```

Record by posting the file as bytes, so nothing is re-encoded (save `note.json` as UTF-8 without BOM):

```powershell
Invoke-WebRequest -UseBasicParsing -Method Post -Headers $h -ContentType "application/json; charset=utf-8" `
  -InFile note.json -OutFile journal-response.json -TimeoutSec 900 "$base/api/equities/MBB/research-notes"
```

The POST re-verifies the snapshot and can take several minutes. If it times out or the connection
drops, do not post again blindly: GET `direction-scores` (Step 5) and look for the note in `items[]`
first; post again only if it is not there. Without the key, save `note.json` unrecorded and say so.

On 409:

- `SIGNAL_BAR_NOT_LATEST` / `FORECAST_WINDOW_CLOSED`: record as `retrospective`, or stop.
- `SNAPSHOT_MISMATCH`: data changed since the read; GET `technical` again, redo affected numbers,
  record again.
- `CALENDAR_COVERAGE_EXPIRED`: the HOSE calendar no longer covers the next session; record as
  `retrospective` or stop and say so.
- Never edit data or the calendar to get a note accepted.

Only the **earliest** forward note per (signal bar, horizon) is scored; later ones are listed
`superseded-not-scored`. Do not re-record to "fix" a call.

## Step 5: Review past calls

`GET /api/equities/{symbol}/research-notes/direction-scores` with header `X-Admin-Key` (from
`EQX_ADMIN_KEY`, never printed):

```powershell
Invoke-WebRequest -UseBasicParsing -Headers $h -OutFile direction-scores.json "$base/api/equities/MBB/research-notes/direction-scores"
```

Report what it says, item by item where useful:

- Per horizon (`horizons[]`): `status` is `insufficient-sample` until `nEffective >= 30`
  (non-overlapping calls), `no-data` when nothing resolved, else `ok`. Below 30, `hitRate` means
  nothing; say so instead of drawing a conclusion. `hitRate`, `coverage`, `returnPct` and the
  thresholds are fractions (0.05 = 5%); hit rates are descriptive counts, not probabilities.
- A call has value only if it beats `baselinePrior` (point-in-time prior) on the same resolved set;
  `bestConstantInHindsight` is shown for reference and is not achievable.
- `no-view` counts toward coverage, not toward right or wrong. Per note (`items[]`), `status` is
  `resolved`, `pending`, `no-view`, `superseded-not-scored`, `retrospective-not-scored`,
  `out-of-window-not-scored`, `window-undetermined-not-scored` or `no-declared-thresholds`; a
  `pending` note carries `pendingReason` (`immature`, `target-bar-missing`,
  `calendar-coverage-expired`, `signal-bar-missing`).

## Where things go

`D:\code\btc\data\research-sessions\{SYMBOL}\{YYYYMMDD-HHMM}-technical\` (ICT creation time; outside
the git repos; one folder per symbol): `technical.json`, `scripts/` (scripts, their outputs and any
`/candles` reads), `note.json`, `journal-response.json`, `direction-scores.json`, and a short
`report.md` (mode, view, evidence, counter-evidence, scenarios, limitations, note id).

## Versions

`skillVersion: "equity-technical-v2"`, `methodVersion: "equity-technical-method-v1"`. Bump the skill
version when this file's rules change and the method version when the analysis procedure changes. v2
(2026-10-30, after the first trial run): record by posting the file with a long timeout and check
before re-posting, payload structure of `conditions` and `freshness`, real caveat codes, binding stays
in the note.
