---
name: investment-research
description: Point-in-time investment research session for the Vietnamese bank equities MBB and BID in the btc repo. It fixes the symbol, as-of date and horizon, checks readiness, exports one verified snapshot through ai/research_snapshot.py, splits broad research across fundamentals, technical/statistical, news/macro and critic roles that all use that same snapshot, and synthesizes the disagreements. The direction view is kept separate from the buy/hold/reduce/wait action, and the session saves a report plus a journal note whose outcome can be checked later. Use it whenever the user asks to research, analyse, review or "nghiên cứu / phân tích / đánh giá" MBB or BID, asks whether to buy more, hold, reduce or wait on them, wants a forward or retrospective view at a date, or wants a research note recorded. Do not use it for BTC/crypto or for changing the app's code.
---

# investment-research (MBB / BID)

This skill turns a research request into a **reproducible, point-in-time** research record. The
backend collects, normalizes and serves the data; this session thinks, criticises its own work and
writes it down. Nothing here places an order, and no probability is invented along the way.

To start it in a new session (it is not auto-discovered):

> Read `D:\code\btc\ai\skills\investment-research\SKILL.md` and follow it for: <request>.

All paths are absolute (Windows). The CLI is `D:\code\btc\ai\research_snapshot.py`; reuse it and
never reimplement it. Its commands and the shell notes (PowerShell vs Git Bash, UTF-8 output) are in
`references/cli.md`.

## Step 0: Fix the question before touching data

Write these at the top of the working notes:

- **symbol**: `MBB` or `BID` only.
- **as-of**, pinned to ONE explicit epoch-ms value before any command:
  - "Now" is `$asOf = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()`, and only this gives a
    `forward` note.
  - A past date is `retrospective`. A date-only request ("30/06/2026") pins **23:59 ICT of that
    date** (`2026-06-30T23:59:00+07:00` converted to epoch ms), so that day's session counts as closed.
  - Pass the same `--as-of $asOf` to `readiness` and `export`. Without it, each call snapshots its own
    "now" and the snapshot ids differ. `journal record --snapshot` takes it from the export.
  - Retrospective notes are stored but never scored as forecasts.
- **horizon** in trading sessions: 20 by default (base-rate config `tb-p5-s4-h20`: +5% / −4%), or 60
  (`tb-p10-s6-h60`: +10% / −6%).
- **what is asked**: a market view only, or a **portfolio action**. Read position, budget or risk only
  for an action. That needs `EQX_ADMIN_KEY` in the environment. Check that it is present, never print
  it, and never ask for it in chat.

If forward vs retrospective is ambiguous, ask one short question: the journal rules differ.

## Steps 1 to 3: Readiness, then one snapshot

1. **Readiness.** Run `readiness --as-of $asOf`. Read each purpose separately; there is no global
   "ready" flag. `not-ready` or `partial` limits what may be claimed in that area.
   **`portfolio-action` is always `requires-admin` on this public endpoint, even with a key.** Judge
   portfolio readiness from `portfolio.json` instead (Step 7).
2. **Export.** Run `export --as-of $asOf` into a fresh session folder (see "Where things go"). Add
   `--include-portfolio` only for an action question. Its `snapshotId` must equal readiness's; if not,
   the data changed in between, so start again. Exit code 3 means the data changed during export:
   re-export, never hand-patch.
3. **Check the sources before analysis.** In `snapshot.json`, note `snapshotId`, `snapshotSha256`,
   `effectiveCutoffMs`, `pit` (incl. `barsUpdatedAfterAsOf`), `unavailable[]`, the conditions
   `warnings`, `sourceDocuments[]` (with `availableAtUtc` / `availabilityBasis` / extraction status),
   **`fundamentalsSeries`** (validated issuer facts, multi-period) and **`fundamentalsCoverage`**
   (which period × group is validated, plus gaps with reasons), and the provenance of each legacy
   fundamentals row.
   - Only `validated` facts are numbers.
   - Cite them as `fact:<factKey>@<sha12>#p<page>`.
   - Keep printed vs derived (`*_CALC`, `isDerived`), quarter/ytd/annual, PAT vs PAT-parent, gross vs
     net loans and consolidated vs separate apart.
   - Claim a trend only when coverage shows every needed period validated. Then read `references/data-limitations.md`, which explains each warning and what
   is known to be missing or not point-in-time.

Which hash to cite: **`snapshotSha256`** (with `snapshotId`) is the content hash of the snapshot.
It goes in the report, in the note, and is what the backend re-verifies. The sha256 values in
`manifest.json` only prove the integrity of the exported files' bytes; list them under References,
never as "the snapshot sha". `snapshotSha256` changes whenever any served content changes
(base-rates artifact rebuilt, document or fundamentals row registered, conditions re-computed), even
with no new bars. If anything was rebuilt or imported during the session, **re-export just before
recording** and redo any affected numbers.

## Step 4: Split the work only when it is broad

Answer narrow questions directly. For a broad thesis or an action question, use the four roles in
`references/subagents.md`:

- fundamentals and valuation
- technical and statistical
- news, events and macro
- a **critic**, run last on the other three drafts

Spawn them as subagents when the environment allows. **If it cannot, run the roles sequentially in
this session**, each written to `agents/<role>.md` before the next starts (critic last). Say so in the
report header.

**Step 5.** Every role uses the same export folder and as-of, and returns claims with evidence IDs,
counter-evidence and missing data. Nothing dated after the as-of may be used, especially in
retrospective sessions and from web sources.

## Step 6: Synthesize the disagreements; never take a majority vote

For each disagreement, give what each side rests on, what would settle it, and your conclusion with
the reason, or "unresolved". Three weak agreeing claims do not outweigh one strong contrary fact.
Unresolved disagreement becomes uncertainty in the report, not a middle position.

## Step 7: Two separate outputs: direction view and action

- **Direction view**: `up | down | sideways | no-view` over the horizon, with scenarios and
  invalidation levels.
- **Action**: `buy-more | hold | reduce | sell | wait | none`, with a basis. Pick the basis from this
  table, first matching row:

| Situation | basis | allowed actions |
|---|---|---|
| `portfolio.json` exported; lots, settled shares, shared cash pool and (for buy-more) this symbol's `buyBudgetVnd` all present; no `corporate-action-unconverted:*` | `position-aware` | any; quantities only from the decision's budget-bound capacity |
| A position exists but the pool or budget is missing, or `corporate-action-unconverted:*` makes PnL / proceeds / cost-based levels null, and the question is **hold vs reduce/sell** | `market-only` | `hold`, `reduce`, `sell` (or `wait`), justified by the market view and stated as conditional. No share count, cost-based stop or PnL; list what must be fixed (convert lots, enter the pool, declare a budget) |
| Same gaps, and the question is **buy more** | `market-only` | `wait` (never `buy-more` without a declared budget and a known pool), with the reason |
| No admin key, or no portfolio exported, **and** a market view exists | `market-only` | `wait` / `hold` as a market-only statement ("for a holder…"), and say why it is not position-aware |
| No usable market view either (`no-view`, and readiness for the needed purpose is not-ready) | `insufficient-input` | `none` only |

A bullish view does not imply buy-more. T+2 sellable shares, the next-session ±7% band (recomputed
in code; see data-limitations #12), 100-share lots and the budget decide what is actually doable.

## Step 8: Honesty rules for numbers

- **Probabilities.** Do not invent them. A probability enters the note only with a declared,
  reproducible method (`method` + `methodRef`) that you actually ran (`references/journal-note.md`).
- **Edge.** A stronger model, or more agreeing agents, is not evidence of edge; only out-of-sample
  results are.
- **Base rates.**
  - The snapshot-level `baseRates.status = abstained` with `gate-not-passed` means the kNN artifact
    failed its out-of-sample gate. That failure makes **every** base-rate number research-only and
    unvalidated, even for configs showing `abstained: false` with research numbers present.
  - Per-config `abstained` answers only "is a current forecast available for this signal bar".
  - Never present research numbers as decision-grade. Qualitative scenarios are fine, labelled
    **unvalidated**.
- **Indicators.** Two RSIs exist. Conditions' RSI14 is a simple-average (Cutler) RSI; the decision's
  `market.rsi14` is Wilder's RSI14. Name which one you quote (default: compute both in a script and
  quote both). The decision's `market.trend` and the conditions regime use different rules and may
  disagree; report that, don't pick one silently.
- **Price levels.** Prices are latest-adjusted (corporate actions rewrite history). Compare across
  ex-dates in returns, not absolute levels.

## Step 9: Compute with code, without look-ahead

Every number not copied from the snapshot (returns, drawdowns, RS, ratios, levels, bands, event
studies) comes from a script in the session's `scripts/` folder. Run it on the export, save its output
next to it, and cite it as `script:<file>#<output>`. Re-running it on the same export must give the
same number.

In hand-written scans, **iterate forward in time only**:

- A signal at bar *t* may use bars ≤ *t* only; its outcome uses bars > *t*.
- Never choose the "end of an episode" by looking ahead (e.g. the last day of an oversold run), never
  centre windows, and never use bars after the cutoff.
- Overlapping episodes are not independent samples; report the effective n.

## Step 10: Save the report and the journal note

1. Write `report.md` from `references/output-template.md`. Keep every section; write "none" or
   "missing" where needed.
2. Build `note.json` (`references/journal-note.md`) and record it with `journal record --snapshot`.
   This needs `EQX_ADMIN_KEY`; without it, save `note.json` unrecorded and say so.
   - A `forward` note is accepted only for the latest finalized 1d bar, before the next session opens.
   - On 409 `SIGNAL_BAR_NOT_LATEST` / `FORECAST_WINDOW_CLOSED`, record it as `retrospective`.
   - On 409 `SNAPSHOT_MISMATCH`, re-export, redo affected numbers and record again.
   - On 409 `CALENDAR_COVERAGE_EXPIRED`, the HOSE calendar file no longer covers the next session,
     so forward notes cannot be recorded until the calendar is extended. Record it as
     `retrospective`, or stop and report that. Never edit the calendar or any data.
   - Never alter data to get a note accepted.
3. **Outcome check** (`outcomeCheck`). Use this declared convention; it is not a probability.
   - Return R = close of the bar `horizonSessions` sessions after the signal bar ÷ signal close − 1,
     on adjusted prices.
   - With the horizon's barrier config (`tb-p5-s4-h20`: target +5%, stop −4%; `tb-p10-s6-h60`:
     +10% / −6%): **up** if R ≥ +target, **down** if R ≤ −stop, otherwise **sideways**.
   - Also say which scenario invalidation triggered first, if any.
   - Write the exact rule into `outcomeCheck.how`.

## Where things go

`D:\code\btc\data\research-sessions\{SYMBOL}\{YYYYMMDD-HHMM}-{forward|retrospective}\` is outside the
git repos and never committed. The timestamp is the folder's **creation time in ICT (UTC+7)**, not the
as-of.

```
export/        research_snapshot.py export output (snapshot.json, series-*.csv, manifest.json[, portfolio.json])
scripts/       analysis scripts + outputs (+ readiness.json, asof.txt)
agents/        each role's raw claims
report.md      the report
note.json      the journal request; journal-response.json after recording
```

## Versions

Use `skillVersion: "investment-research-v1.2"` and `methodVersion: "investment-research-method-v1"` in
notes. Bump the skill version when this file's rules change, and the method version when the analysis
procedure changes.
