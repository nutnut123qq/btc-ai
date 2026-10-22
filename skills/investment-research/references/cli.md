# CLI reference — `D:\code\btc\ai\research_snapshot.py`

Stdlib-only Python; talks to the backend HTTP API (never the DB). Run with the repo venv or system
Python 3.12:

```powershell
$py = "D:\code\btc\ai\venv\Scripts\python.exe"
$cli = "D:\code\btc\ai\research_snapshot.py"
```

Backend: global `--backend URL` (before the subcommand: `& $py $cli --backend URL readiness …`) or env `BACKEND_BASE_URL`, default `http://localhost:5197` (the local dev
stack). Do not point it at production unless the user explicitly asks — and even then everything is
read-only except `journal record`.

Admin key: env `EQX_ADMIN_KEY`, sent only as the `X-Admin-Key` header, never printed or written. Check
presence without revealing it:

```powershell
& $py -c "import os; print('admin key present:', bool(os.environ.get('EQX_ADMIN_KEY')))"
```

## Shell notes

- Commands here are **PowerShell** (`$py`, `$cli`, `$asOf`).
- In Git Bash, use forward-slash `D:/...` paths. `/d/...` paths are not understood by Windows Python
  when passed inside scripts.
- Set `PYTHONIOENCODING=utf-8` (`$env:PYTHONIOENCODING = "utf-8"`) before running analysis scripts
  that print Vietnamese, or the console code page garbles or crashes on `ư`/`ố`. Write outputs to JSON
  files with `encoding="utf-8"` and `ensure_ascii=False`.
- Do not run scripts that read downloaded issuer files from inside their folder; pass paths as
  arguments and use `python -I`.
- Retrospective date-only as-of: `$asOf = [DateTimeOffset]::Parse("2026-06-30T23:59:00+07:00").ToUnixTimeMilliseconds()`.

## Commands

| Step | Command | Output |
|---|---|---|
| readiness | `& $py $cli readiness --symbol BID --as-of $asOf` | JSON `equity-research-readiness-v1` to stdout: `purposes.{technical,fundamental,quantitative,portfolio-action}.{status,reasons,checks}` |
| export | `& $py $cli export --symbol BID --as-of $asOf --out-dir <session>\export` | `snapshot.json` (bytes as served), `series-{tf}.csv` (all bars ≤ cutoff, re-fingerprinted), `manifest.json` (sha256 per file) |
| export (action) | add `--include-portfolio` | also `portfolio.json` (position + decision; AdminGuard) — kept separate from `snapshot.json` |
| record note | `& $py $cli journal record --symbol BID --file <session>\note.json --snapshot <session>\export\snapshot.json` | created note JSON (save as `journal-response.json`) |
| list notes | `& $py $cli journal list --symbol BID [--take 50]` | `equity-research-notes-v1` |

`--as-of`: epoch ms or ISO-8601 **with offset**. Always pass it, using ONE value per session
(`$asOf = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()` for "now"). If it is omitted, every call
snapshots its own "now", so readiness and export disagree on `snapshotId`. The signal bar is the latest
1d session closed (14:45 ICT) at or before the as-of. `snapshotId` = `snap/{SYM}/{effectiveCutoffMs}/{sha12}`. `--snapshot` fills
`snapshotId`, `snapshotSha256`, `requestedAsOf` and `signalBarOpenTimeMs` from the export when the
note does not set them.

Exit codes: `0` ok · `1` HTTP error (message carries status + code) · `2` usage · `3` fingerprint
mismatch (stored bars changed since the snapshot — re-export; no manifest is written).

## Other read-only endpoints (from `snapshot.howToGoDeeper`)

- `GET /api/equities/{SYM}/source-documents` — issuer document metadata (files are not served; originals
  under `D:\code\btc\data\equity-documents\{SYM}\`, read-only, untrusted — parse with a pure-Python
  reader under `python -I` if you must open one).
- `GET /api/equities/{SYM}/financial-facts?metric=&periodType=&from=&to=&asOf=&status=`
  (`equity-financial-facts-v1`): every fact with its document ref (url, sha256, page, location),
  extraction, `availableAtUtc`, `isDerived`/`derivation`, `scale` and `selection`. `status` defaults to
  `validated`; `status=all` also returns review-needed/quarantined ones, for auditing only. Always pass
  `asOf=$asOf` so it matches the snapshot. Series points already carry `scale` and
  `normalizedValue`; use this endpoint to inspect superseded/restated values and provenance.
- `GET /api/equities/{SYM}/fundamentals/coverage[?asOf=]` (`equity-fundamentals-coverage-v1`): the
  full period × group × consolidation grid with stage, counts, `missingMetrics`, reason and
  attempt/document ids.
- `GET /api/equities/{SYM}/documents/pipeline-runs` (`equity-pipeline-runs-v1`): pipeline run
  history (status, counts). Read it to see whether a backfill is still running before concluding that
  coverage is final.
- `GET /api/equities/{SYM}/forecast-journal` — forward forecasts and resolved outcomes.
- `GET /api/equities/{SYM}/data-quality` — latest coverage (not point-in-time).

## Verified on the live test stack (2026-10-18, BE http://127.0.0.1:5297, DB btc_eqx_test)

- [x] **`readiness` returns all four purposes; `portfolio-action` = `requires-admin`.** Run with
  `--as-of 1792315293201` on BID: technical `ready`, fundamental `partial` (`metrics-unavailable`),
  quantitative `partial` (`oos-gate`), portfolio-action `requires-admin` (`position-data-requires-admin`;
  public readiness is market-only even when a key is set).
- [x] **Readiness and export with the same explicit as-of give the same snapshot.** Both returned
  `snap/BID/1792136700000/33736d4bb62c` with identical `snapshotSha256` (signal bar 1792108800000,
  2026-10-16; `asOfIsLatest` true).
- [x] **`export` writes `snapshot.json`, `series-1d.csv` and `manifest.json`.** BID: 3173 bars,
  fingerprint verified; `portfolio.json` appears only with `--include-portfolio`. Exit code 3 was not
  provoked live, because that would need a data change between snapshot and paging. It is covered
  offline by `ai/tests/unit/test_research_snapshot.py::test_export_fingerprint_mismatch_fails_without_manifest`.
- [x] **`journal record --snapshot` works end-to-end.** It was not run during the skill check (by
  instruction), but the forward-test recorded a forward MBB note: `session/MBB/1792108800000/f0e364fd`
  (`snapshotVerified: true`, `forecastIds: []` because `probability: null`; see
  `D:\code\btc\data\research-sessions\MBB\20261018-1658-forward\journal-response.json`).
  `journal list` works.
- [x] **`--as-of` with a `+07:00` offset resolves to the expected signal bar.**
  `2026-10-15T16:00:00+07:00` (after the 14:45 close) gives cutoff 1792050300000, the 10-15 bar.
  `2026-10-15T14:00:00+07:00` (session still open) gives cutoff 1791963900000, the 10-14 bar.

Example (PowerShell), validated quarterly PAT attributable to the parent at the session as-of:
`Invoke-RestMethod "http://127.0.0.1:5297/api/equities/MBB/financial-facts?metric=PROFIT_AFTER_TAX_PARENT&periodType=quarter&asOf=$asOf" | ConvertTo-Json -Depth 6 | Out-File -Encoding utf8 scripts\facts-pat-parent.json`.
Save such reads under `scripts\` and cite them like any other output.

Note: `readiness` and `research-snapshot` are on the backend's "expensive" limiter (10 requests per
minute per IP), so do not loop them.
