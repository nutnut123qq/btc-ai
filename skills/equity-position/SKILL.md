---
name: equity-position
description: Position and risk session for the Vietnamese bank equities MBB and BID in the btc repo. It records the user's real trades and cash into the backend (PUT /api/equities/{symbol}/position, PUT /api/equities/cash), then reads the backend's rules-only decision (GET /api/equities/{symbol}/decision) and explains it - action and reasons, -8% stop and -4% reduce levels, T+2 sellable shares, the next-session ±7% band, buy-more capacity within the declared budget, and the loss if the stop is hit. Use it when the user says they bought or sold MBB/BID, sets cash or a buy budget, or asks "vị thế", "rủi ro", "cắt lỗ", "bán được bao nhiêu", "mua thêm được bao nhiêu". Never places an order, never gives a probability, never reads news or fundamentals; for a chart/candle view use equity-technical.
---

# equity-position (MBB / BID, your holdings)

The backend holds the lots, the cash pool and the declared rules, and computes every decision
number. This session writes only what the user says they actually did, then reads the decision and
explains it. It never places an order, never produces a probability, and is not investment advice:
the action is the output of fixed, declared rules (`evidenceStatus: rules-only`).

Sessions opened in `D:\code\btc` find it on their own: `.claude\skills\equity-position` (Claude Code)
and `.agents\skills\equity-position` (Codex, not checked there yet) are junctions to this folder, made by
`D:\code\btc\ai\scripts\link_skills.ps1`. Anywhere else, start it with:

> Read `D:\code\btc\ai\skills\equity-position\SKILL.md` and follow it for: <request>.

This file is self-contained: it needs no other skill's files.

Backend: env `BACKEND_BASE_URL`, default `http://localhost:5197`. Every endpoint here is admin-only
(personal data): send env `EQX_ADMIN_KEY` as header `X-Admin-Key`, never print or write it. If
`[bool]$env:EQX_ADMIN_KEY` is `False`, stop and say the user must set `EQX_ADMIN_KEY` themselves
(value from `backend/ops/show-admin-key.ps1`); do not read ops secrets or look for the key anywhere
else. A 401 means the key is missing or wrong: stop. A 404 means the symbol is outside the scope.
Shell notes:

- Commands here are **PowerShell**. In Git Bash, write paths as `D:/...`; Windows Python does not
  understand `/d/...` inside scripts.
- Before running a script that prints Vietnamese, set `$env:PYTHONIOENCODING = "utf-8"`; write JSON
  outputs with `encoding="utf-8"` and `ensure_ascii=False`.

```powershell
$base = if ($env:BACKEND_BASE_URL) { $env:BACKEND_BASE_URL } else { "http://localhost:5197" }
$h = @{ "X-Admin-Key" = $env:EQX_ADMIN_KEY }
```

First create one session folder per symbol (see "Where things go") and work inside it. The cash
pool is shared by MBB and BID: when a request covers both, read it once and say it is the same pool.

## Scope (hard limits)

- **Symbols:** `MBB` or `BID` only.
- **Endpoints, and nothing else:**
  - `GET /api/equities/{symbol}/position`: lots, settlement, shared cash, this symbol's buy budget.
  - `PUT /api/equities/{symbol}/position`: replace the lots (and the buy budget).
  - `PUT /api/equities/cash`: set the one shared cash pool.
  - `GET /api/equities/{symbol}/decision`: the rules-only decision.
- Write only what the user states. Never invent a trade, a price, a date or a cash figure, and never
  "correct" data to change a decision.
- Never read news, fundamentals, events, documents or the web. For a direction view, point to
  `equity-technical`.

## Step 1: Read the current state

Save both responses in the session folder (`Invoke-WebRequest -UseBasicParsing -Headers $h <url>
-OutFile position.json`, same for `decision.json`) and read these fields by these exact names:

- `position.json`: `lots[]` (`id`, `quantity`, `costPerShareVnd`, `acquireDate`, `settleDate`,
  `settled`, `source`), `totalShares`, `settledShares`, `pendingShares`, `avgCostPerShareVnd`,
  `sharedCashVnd`, `buyBudgetVnd`, `totalDeclaredBudgetsVnd`, `cashSource` (`shared-pool` or
  `unknown`), `cashNotes`, `updatedAtUtc`. `availableCashVnd` is a legacy mirror that reads 0 while
  the pool is unknown: use `sharedCashVnd` with `cashSource`, and never copy `availableCashVnd` into a
  PUT (a 0 there would be written to the shared pool).
- `decision.json`: `action`, `reasons[]`, `triggers[]`, `invalidation`, `position`, `buyMore`,
  `market`, `cash` (`sharedCashVnd`, `buyBudgetVnd`, `totalDeclaredBudgetsVnd`, `cashSource`,
  `spendableVnd`), `dataFreshness[]`, `evidenceStatus`, `priceBasis`, `generatedAtUtc`.

Empty `lots` (`decision.reasons` = `no-position`) means nothing has been entered yet; `cashSource:
unknown` means the cash pool has never been set. Say so; do not assume the user holds nothing. With
`no-position` the decision returns early: no trend reason, no triggers, and `buyMore` is always null
whatever the cash, so this endpoint never sizes a first buy; say that too.

## Step 2: Record what the user did (only when they say so)

Before any PUT, show the user the exact lot list and figures you are about to send, and send only
after they confirm.

- **The position PUT replaces everything.** It deletes every stored lot of the symbol and writes the
  body's lots, and it sets `buyBudgetVnd` to the body's value: an omitted or null `buyBudgetVnd`
  **clears** the budget. So always start from `position.json`: copy every existing lot
  (`quantity`, `costPerShareVnd`, `acquireDate`; drop `id`, `settleDate`, `settled`, `source`), apply
  the change, and send the current `buyBudgetVnd` back unless the user changes it. Do not send
  `availableCashVnd` here; set cash with the cash PUT.
- **Buy:** add one lot `{ "quantity", "costPerShareVnd", "acquireDate" }`: matched quantity, matched
  price per share in VND (before fees), trade date as `yyyy-MM-dd` (Vietnam date, not in the future).
  Several fills at different prices are separate lots.
- **Sell:** remove the sold quantity from lots, oldest `acquireDate` first, unless the user names the
  lots. Only settled shares can have been sold (T+2); if the user reports selling more than
  `settledShares`, or more than they hold, stop and ask instead of writing.
- **Corporate actions:** enter lots exactly as bought (original quantity and price). The backend
  converts stock dividends, bonus shares and splits itself (`decision.position.corporateActions`);
  never pre-adjust a lot.
- **Cash:** `PUT /api/equities/cash` with `{ "availableCashVnd": <settled, spendable VND> }`. One pool
  serves MBB and BID; `null` resets it to unknown.
- **Budget:** `buyBudgetVnd` in the position PUT is this symbol's share of the pool. Budgets of all
  symbols together must stay within the pool.
- On 400: `INVALID_POSITION_INPUT`, `INVALID_CASH_INPUT`, `CASH_UNKNOWN` (set cash before a budget),
  `BUDGET_EXCEEDS_CASH`. Report the message and fix the input with the user; never loosen a figure to
  pass validation.
- After a PUT, GET `position` again and check `totalShares`, `settledShares` and `buyBudgetVnd` match
  what the user expects. Save the request body and both responses.

## Step 3: Explain the decision

Re-read `decision.json` after any write. Report, with field paths:

- **Action** (`hold`, `wait`, `buy-more`, `reduce`, `sell`, `conditional-sell`) and every reason code,
  verbatim, with its meaning:

  | reason | meaning |
  |---|---|
  | `no-position` | nothing entered for this symbol |
  | `stop-loss-hit` | last 1d close at or below avg cost -8% (sell needs the whole position settled) |
  | `reduce-threshold-hit` | at or below avg cost -4% |
  | `partial-settlement-sell-settled-only` | only settled shares can go now (T+2) |
  | `trend-bearish` | `market.trend` is `down` and RSI14 < 50 |
  | `trend-constructive` | `market.trend` is `up`, 50 <= RSI14 < 70, and at least one lot is affordable |
  | `cash-unknown` / `buy-budget-undeclared` / `budget-exceeds-cash` / `cash-below-one-lot` | why no buy quantity is proposed |
  | `buy-sized-at-next-session-ceiling` | buy-more quantity priced at the next session's ceiling |
  | `no-settled-shares` / `sell-not-fillable` | a sell/reduce fired but nothing can be sold now (nothing settled, or no positive net proceeds) |
  | `stale-data` | some timeframe is stale (e.g. only 1h); on its own it does not change the action |
  | `stale-data-blocked:<action>` | a stale 1d turned sell/reduce/conditional-sell/buy-more into `wait` |
  | `corporate-action-since-acquisition` / `corporate-action-converted:<type>` / `corporate-action-unconverted:<reason>` | a corporate action went ex after the earliest lot; converted lots are already adjusted, unconverted ones null the cost figures |
  | `insufficient-data`, `indicators-warming-up`, `no-trigger` | not enough data, or nothing fired |

  Quote any other code verbatim and say it is not in this table.
- **Levels:** `triggers[]` (`name`, `priceLevelVnd`, `condition`, `direction`, `withinPriceBand`,
  `nonFillRisk`) and `invalidation`. HOSE rejects orders outside the band, it does not clamp them.
- **The band in the decision is the next session's band.** `market.ceilingPriceVnd` /
  `market.floorPriceVnd`, and `withinPriceBand` / `nonFillRisk` on triggers, are computed around
  `market.lastDailyCloseVnd` (the signal close, the next session's reference price) on the HOSE tick
  grid, ceiling rounded down and floor rounded up (backend rules `declared-rules-v4`). Quote them as
  returned; do not recompute. Journal rows stamped `declared-rules-v1` to `v3` used the previous close
  as the reference. `market.previousDailyCloseVnd` is only the close before the signal bar. This holds only
  while the 1d entry of `dataFreshness[]` has `isStale: false`: with a stale 1d bar the band is built
  on an old close, so say it is out of date and do not rely on it for an order.
- **Stop and reduce levels:** with a position and `position.costBasisComparable` true, every rule
  branch returns the `stop-loss` (avg cost -8%) and `reduce-below-cost` (avg cost -4%) triggers,
  already rounded up to a tick. Quote their `priceLevelVnd`; do not compute levels yourself. No such
  trigger means no position, a non-comparable cost basis, or an early `wait` on missing data
  (`insufficient-data`, `calendar-coverage-expired`): say which, and give no level.
- **Position:** `position.totalShares`, `settledShares`, `sellableSharesNow`, `avgCostPerShareVnd`,
  `lastPriceVnd`, `unrealizedPnlVnd`, `unrealizedPnlPct`, `netProceedsIfSellableNowVnd`. If
  `position.costBasisComparable` is `false`, every cost-based figure is null on purpose: give
  `costBasisNote` and the `corporateActions[]` with status `unconverted`, and do not estimate PnL.
- **Buying power:** `cash.spendableVnd`, `cash.buyBudgetVnd`, and `buyMore.maxBuyableShares` /
  `maxBuyableLots` / `estimatedCostVnd` (an upper bound priced at the ceiling). Never propose a quantity
  above `buyMore`, and none at all when `buyMore` is null.
- **Loss if the stop is hit:** compute with a script on the saved JSON (cite it as
  `script:<file>#<output>`): `(avgCostPerShareVnd - stop-loss priceLevelVnd) x totalShares`, plus sell fee 0.15%
  and sell tax 0.1% on the gross at the stop price. If the user gives their total capital, also give it
  as a percent of that. Practitioners often cap the loss at the stop at roughly 0.5-2% of capital per
  position; say it is a convention, not a rule of this app.
- **Freshness:** `dataFreshness[]` per timeframe (`isStale`, `reason`). A stale 1d blocks
  sell/reduce/conditional-sell/buy-more by rule; say the decision is on old data.

## Honesty rules

- **No probabilities, no forecasts.** The decision is rules-only. For a direction view, use
  `equity-technical`; do not mix its view into these numbers.
- **Fees are defaults, not the user's broker.** Broker fee 0.15% per side, sell tax 0.1%, lot 100,
  T+2, band ±7% are the backend's declared defaults (`backend/Services/VietnamExecutionRules.cs`). The user's
  real broker schedule is not entered yet; say costs are approximate.
- **Prices are adjusted.** `priceBasis` is `adjusted` (corporate actions rewrite history); compare in
  returns across an ex-date, not levels.
- The final call is the user's. Say what the rules output and why; do not tell the user to buy or sell.

## Where things go

`D:\code\btc\data\research-sessions\{SYMBOL}\{YYYYMMDD-HHMM}-position\` (ICT creation time; outside
the git repos, personal data, never commit): `position.json`, `decision.json`,
`put-request.json` / `put-response.json` for each write, `scripts/`, and a short `report.md`
(what changed, action, levels, risk at the stop, freshness, limitations).

## Versions

`skillVersion: "equity-position-v3"`. Bump it when this file's rules change. v2 (2026-10-30, after the
first trial run): next-session band recomputed, stop/reduce computed when no trigger, more reason codes,
`availableCashVnd` trap, early return on `no-position`. v3 (2026-10-30, EQDEC-1): the backend now
returns the next-session band and both cost triggers on every branch (`declared-rules-v4`), so the
skill quotes them instead of recomputing. SKILLS-1 (2026-10-30) moved the shell notes and the band
note in so the file needs no other skill, and made it discoverable from `D:\code\btc`; the rules are
unchanged, so the version stays v3.
