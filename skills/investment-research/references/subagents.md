# Subagent briefs and claim format

Use subagents for broad questions (thesis, action, "full review"). **If the environment cannot spawn
subagents, play the four roles sequentially in this session.** Write each role's output to
`agents\<role>.md` before starting the next, run the critic last against those files, and state "roles
run sequentially in-session" in the report header. Give each one: the absolute session
folder, the as-of, horizon, mode (forward/retrospective) and its brief below. All of them read the **same**
`export\` folder; none fetches its own price data. Save each raw answer under `agents\<role>.md`.

## Shared rules (paste into every brief)

- Use only data available at or before the as-of. For web sources, record URL **and** publication date;
  anything published after the as-of is excluded (retrospective sessions especially).
- Every number you derive (returns, ratios, levels) comes from a script you save under `scripts\` and
  run on the export; cite it. Scans iterate **forward in time only**: a signal at bar t sees bars ≤ t,
  and its outcome uses later bars. No "last day of the episode" selection, no centred windows.
- Return claims in the format below. No probabilities unless you computed them with a declared,
  reproducible method — and say it is unvalidated.
- Say what you could not check (`missingData`) rather than filling gaps with priors.

## Claim format

```json
{
  "role": "fundamentals",
  "claims": [
    {
      "claim": "Q2/2026 consolidated profit after tax rose vs Q1/2026",
      "evidenceIds": ["snapshot:fundamentals.items[2]", "doc:7cada4b4-…#page6"],
      "counterEvidence": ["NIM from KBS is TTM, not quarterly — growth may be mix-driven"],
      "strength": "strong | moderate | weak",
      "horizonRelevance": "how it matters over the stated horizon"
    }
  ],
  "missingData": ["NPL ratio has no point-in-time source"],
  "notes": "anything the coordinator must know (data oddities, provenance doubts)"
}
```

Evidence ID scheme (keep it resolvable by someone else):

| Prefix | Points to |
|---|---|
| `snapshot:<json-path>` | a field in `export\snapshot.json` (e.g. `snapshot:relativeStrength.benchmarks[0]`) |
| `series:<tf>@<openTimeMs>[..<openTimeMs>]` | rows in `export\series-<tf>.csv` |
| `portfolio:<json-path>` | `export\portfolio.json` (action questions only) |
| `fact:<factKey>@<sha256[:12]>#p<page>` | a validated financial fact (`snapshot:fundamentalsSeries` / `/financial-facts`): factKey + document sha256 + page; derived facts cite `#derived` plus their input factKeys |
| `doc:<documentId>#page<N>` | an issuer source document (`snapshot:sourceDocuments[]`) |
| `script:<file>#<output>` | a saved script and its output |
| `web:<url>@<publishedDate>` | an external page, published on or before the as-of |

## Briefs

**Fundamentals & valuation**: earnings trajectory, balance sheet, asset quality, capital, valuation.
- **Multi-period work uses `snapshot.fundamentalsSeries`.** Check `fundamentalsCoverage` and the
  readiness `fundamental` checks first: a trend, YoY or TTM is claimed only when every needed period
  is validated/persisted for that group. Otherwise name the gaps.
- **Only `validated` facts are numbers.** Review-needed, quarantined or extracted facts are at most
  "unvalidated value exists".
- Use `normalizedValue` (value × scale, in `unit`) for absolute amounts, never the as-printed `value`.
- Cite each number as `fact:<factKey>@<sha12>#p<page>`.
- **Keep definitions apart, and state which you use:**
  - printed vs derived (`isDerived`, `*_CALC`);
  - quarter vs ytd vs half-year vs annual;
  - `PROFIT_AFTER_TAX` vs `PROFIT_AFTER_TAX_PARENT`;
  - gross vs net customer loans;
  - consolidated only (separate statements are not extracted).
- **NPL / CAR / CASA** exist only where the reports print them; check `missingMetrics` per period, and
  don't estimate them.
- **Availability.** MB facts for 2016–2019Q1 have first-seen availability (not PIT). Legacy KBS/VCI
  rows and press figures are cross-checks, labelled as such and never merged with issuer facts.
- **Valuation (PB, P/E)** needs a price, and prices are latest-adjusted. Use the same adjustment basis
  for the price and the per-share book/earnings, or say that the valuation cannot be computed
  consistently.

**Technical & statistical** — trend/structure on 1d (and 1h for MBB only), conditions (latest as-of only),
relative strength vs VNINDEX and peers (respect `stale` benchmarks), drawdown/volatility, base rates.
Base rates: if `abstained`, research-only numbers may be quoted only as "unvalidated, abstained because
<reasons>". Compute everything from `series-*.csv` with a saved script.

**News, events & macro** — corporate actions and events from the snapshot (with `verificationStatus`),
issuer disclosures, sector/policy/rates news. Each item: date, source, why it matters over the horizon,
and whether it is already reflected in price (cite the series). Unverified corporate actions are flagged,
never used to adjust numbers.

**Critic** (runs last, gets the three drafts) — attack the strongest claims: look-ahead, survivorship,
definition mismatches, adjusted-price artefacts, small samples, stale benchmarks, correlated "independent"
evidence, missing counter-scenarios, action not doable under T+2 / ±7% band / lot / budget. Output: a list
of objections, each with the claim it targets, the evidence, and what would resolve it.

## Coordinator synthesis (after all returns)

1. Table of claims by theme; mark agreement / disagreement.
2. For each disagreement: both sides' evidence, what would settle it, your call and why — or "unresolved".
3. Adopt the critic's objections or rebut them with evidence; never silently drop them.
4. No majority vote, no averaging of opinions into a middle view.
