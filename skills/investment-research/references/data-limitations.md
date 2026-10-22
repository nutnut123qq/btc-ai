# Known data limitations (state them in every report they touch)

Snapshot-specific limitations are in `snapshot.json` → `pit.limitations`, `unavailable[]`,
`contextLimitations`; always read those first. Standing ones as of 2026-10-19 (data contract v3):

1. **Prices are latest-adjusted, not point-in-time.** VCI/KBS back-adjust the whole history at every
   ex-date (stock dividend, bonus, cash dividend). `pit.reconstruction` is at most `partial`: a level quoted
   for a past as-of differs from what was on the screen then, and every ex-date rewrites history. Returns
   are fine; absolute levels (supports, cost comparisons) across an ex-date are not.
2. **Corporate-action conversion.** Lots bought before an ex-date are converted only for verified
   stock-dividend / bonus / split with a ratio; anything else ⇒ PnL / proceeds / cost-based levels are
   `null` with reason `corporate-action-unconverted:{type}` — say so, never recompute by hand.
3. **Issuer financial facts (v3) are the primary fundamentals source.** Statements 2016 → 2026H1 for BID
   and MBB are discovered, OCR'd (local Tesseract + RapidOCR), normalized into financial facts with
   page/location provenance and validated by accounting identities. Read them through:
   - `snapshot.fundamentalsSeries`: validated facts visible at the as-of, one point per
     (metric, period, periodType, consolidation), using the *latest-available* validated fact. Audit
     adjustments and restatements win over the original once known; superseded facts are kept but
     not selected.
   - `snapshot.fundamentalsCoverage`: period × metric group × consolidation stages
     (`not-attempted < not-listed < discovered < downloaded < extracted < validated < persisted`)
     plus gaps with reasons. `persisted` means every expected catalog metric of the group was
     validated; `validated` means only some were.
   - `GET /api/equities/{SYM}/financial-facts`: every fact, including superseded and non-validated ones,
     with `selection`.

   **Rules.**
   - **Status.** Only `status: validated` is a fact. `review-needed`, `quarantined` and `extracted`
     facts failed or skipped validation (identity mismatch, unreadable OCR, conflicting comparatives).
     Never cite them as numbers. At most mention that a value exists but is unvalidated.
   - **Status is current, not as-of.** A later quarantine is not reconstructible for a past as-of.
   - **Scale.** `value` is as printed (reports usually print in triệu đồng, scale 1,000,000);
     `normalizedValue` (= value × scale, exact, in `unit`) is what to use for absolute amounts. Both
     series points and `/financial-facts` items carry `scale` and `normalizedValue`. Never mix a raw
     `value` with a `normalizedValue`.
   - **Printed vs derived.** `isDerived: true` facts are computed by the pipeline: Q4 = FY − 9M,
     quarter = YTD difference, and `*_CALC` ratios such as `NPL_RATIO_CALC`, LDR, ROE and NIM. Each
     carries `derivation.formula` and its input factKeys. A ratio the issuer *printed* (statement
     `note`) is a different metric from its `_CALC` twin. Never merge the two, and label derived
     numbers as derived.
   - **Period types.** `quarter`, `ytd` (e.g. `2026-9M`), `half-year` (`2026H1`), `annual` (`2026`) and
     `point-in-time` (balance-sheet dates) are different series. A Q4 report's YYYY ytd column is not
     the audited FY. Comparative columns (`columnRole: comparative`) are prior-period figures printed
     in a later report.
   - **Definitions that are not interchangeable:**
     - `PROFIT_AFTER_TAX` (consolidated, incl. minority interests) vs `PROFIT_AFTER_TAX_PARENT`;
     - `CUSTOMER_LOANS_GROSS` vs `CUSTOMER_LOANS_NET` (after provisions);
     - consolidated vs separate.
   - **Separate statements are downloaded but not extracted this round.** Consolidated is the only
     extracted basis. Coverage shows separate cells at `downloaded`.
   - **NPL / CAR / CASA** exist only where a report prints them; groups 3–5 come from the loan-quality
     note, and many quarterly reports omit CAR and CASA. Check the coverage cell's `missingMetrics`
     for the periods you need. The legacy provider rows (KBS/VCI ratios without a PIT date) remain
     informational only.
   - **Trends.** Before claiming a multi-period trend, YoY or TTM, confirm in
     `fundamentalsCoverage` / readiness (`facts-core-quarters`) that each needed period is `validated`
     or `persisted` for that group. Name the gaps; never interpolate.
4. **Issuer document availability (point in time).** A document, and every fact from it, counts as
   known at `availableAtUtc`:
   - **BID:** the IR listing date (date-only), so available from 00:00 ICT of the next day
     (`issuer-published`).
   - **MB:** no reliable publication date. Availability is first-seen unless a date was recovered from
     MB notices or an HTTP Last-Modified (only after mid-2019); see `availabilityBasis`. **MB documents
     for 2016–2019Q1 are first-seen, i.e. when this system downloaded them (October 2026). A
     retrospective snapshot before that date does not see them.** That is correct point-in-time
     behaviour, not data to add by hand.
   - **Revisions:** a revision (`supersedesDocumentId` set) is visible only from its own first-seen
     time.
   - Cite facts as `fact:<factKey>` together with the document sha256 and page (see
     `subagents.md`).
5. **Fundamentals sources differ by definition.** KBS CSTC, VCI ratios (TTM, no PIT date) and
   `issuer-report` rows are never interchangeable under one metric name; NIM is quarterly/TTM as served,
   do not annualize.
6. **Quant base rates: gate not passed for any of the 4 configs (2 configs × MBB/BID).**
   - Snapshot-level `baseRates.status: abstained` with `gate-not-passed` means the kNN artifact failed
     its out-of-sample gate.
   - That failure makes every number research-only and unvalidated, including configs that show
     `abstained: false` with research probabilities. Per-config `abstained` only says whether a
     current forecast exists for this signal bar; it is not a validation flag.
   - `decisionUse` is null whenever the gate failed.
   - For a past as-of, the artifact is withheld unless its `dataAsOf ≤` signal bar
     (`artifact-not-pit-for-asof`).
7. **Conditions** are only computed for the latest finalized bar; a past as-of shows them unavailable.
8. **Calendar provenance.** Non-trading days come from `vn-hose-non-trading-days.json`; older days are
   `unverified-legacy` (treated as closed to avoid fake gaps, but not backed by an official notice). Gaps
   are `market-wide-missing` (all symbols + VNINDEX absent: possibly an unverified closure) or
   `symbol-missing` (provider gap or suspension — reason not assigned).
9. **Relative strength**: a benchmark whose latest bar lags the instrument is `stale`
   (`benchmark-stale` / `aligned-window-ends-before-instrument-latest`) — do not use its returns. RS uses
   the latest ~400 stored bars, so very old as-of values may be unavailable.
10. **MBB has 1h and 1d; BID is 1d only.** 1h bars have provider quirks (missing 09:00 bars, stray 12:00
    bar) — prefer 1d for anything statistical.
11. **Volume** history differs between providers deep in the past (adjustment differs); treat old volume
    comparisons as approximate.
12. **Decision price band (known backend issue, not fixed).** The decision's ±7% ceiling/floor
    (`market.ceilingPriceVnd` / `floorPriceVnd`) appears to be based on the *previous* close, not the
    signal close, and is not rounded to HOSE price ticks. Before quoting a band for the next session,
    recompute it in a script:
    - reference = signal-bar close; ceiling = reference × 1.07 rounded **down** to a tick; floor =
      reference × 0.93 rounded **up** to a tick;
    - HOSE ticks: 10 ₫ below 10,000; 50 ₫ from 10,000 to < 50,000; 100 ₫ at ≥ 50,000;
    - adjusted prices equal raw prices only after the latest ex-date.
13. **Conditions and snapshot warnings.**
    - `below-calendar-coverage`: the analysis window reaches back before the start of the HOSE
      calendar file. Session contiguity there cannot be checked, so the window stops at the coverage
      start and long-lookback items (e.g. Fibonacci anchors, often with
      `fibonacci-anchor-may-predate-window`) may start later than the true swing. Treat long-lookback
      levels as approximate. Recent-bar indicators are unaffected.
    - `calendar-coverage-expired`: the bar to evaluate lies after the calendar's coverage end, so
      trading days are unknown and conditions fail closed (unavailable). Report it as a data gap;
      never assume weekdays are sessions.
    - `pit.barsUpdatedAfterAsOf` (per timeframe): stored bars ≤ cutoff that were written or updated
      after the as-of (ingestion, back-adjustment, revision). 0 means the stored series equals what
      was knowable then, apart from adjustment. A large count (typical for old as-of values: every
      bar rewritten) means absolute levels differ from what was on screen at the as-of. Use returns,
      and state the count in the report.
14. **Events and corporate actions availability.** Rows carry `availableAtUtc` and
    `availabilityBasis`. Availability is the provider's publish date, except that it falls back to
    `first-seen` when that date is missing or implausible. Implausible means the publish date is
    **more than 90 days before** the item's own date: the `EventDate`, or for corporate actions the
    `ExDate` or the earlier `RecordDate`. A retrospective snapshot
    only includes rows available at the as-of. Do not add later-known events by hand.
15. **Two RSIs and two trend labels.** Conditions use a simple-average (Cutler) RSI14 and their own
    regime label; the decision (`portfolio.json` → `decision.market`) uses Wilder RSI14 and its own
    `trend`. They can differ materially (seen: 26.9 vs 38.7 on BID). Always name the definition, and
    preferably compute both in a script.
