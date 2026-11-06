# VN30-2 evaluator notes (addendum, not part of the frozen protocol)

`docs/research/protocols/vn30-2.json` is frozen and is not edited. This note lists the choices that
`vn30_panel_eval.py` makes where the protocol text is silent, so a reviewer can check them against the code.
They were fixed in code before the VN30-2 holdout was opened (run `20261106T121249Z`); the later changes listed
at the end only add guards and do not change any reported number.

## Choices not stated in the protocol

| Topic | Choice in the evaluator | Protocol text it interprets |
|---|---|---|
| Label span | The label of a decision at session `s` spans sessions `[s, s + 21]` (entry at the open of `d+1`, exit at the open of `d+21`). | `decisionSchedule.exit`, `validation.purgeSessions = 21` |
| Purge + embargo | A training decision is kept only if `s + purge + embargo < first test decision` (`21 + 20`). A training block after the test block (not used in the forward-only walk) must start more than `purge + embargo` sessions after the last test decision. The embargo therefore acts as an extra gap before the test block, which is stricter than the minimal overlap purge. | `purgeSessions 21`, `embargoSessions 20` |
| Walk-forward | One fold per calendar year of test decisions, refit at that year's first decision on every earlier decision that survives purge + embargo (expanding). Years without training data are skipped, so the development comparison covers 2019-2023 for every model alike. | `validation.scheme` |
| Window end | `yearly_folds` takes the session calendar and finds `last` with `calendar[last] <= test_to < calendar[last + 1]`. A test decision whose label ends after it (`s + 21 > last`) is dropped, so a development score never reads a holdout return and a holdout score never reads a post-holdout return. | `developmentWindow`, `holdout` |
| History start | Decision dates before 2018-01-22 (the development window start, the first VN30 period) are refused unless `allow_pre_history` is passed explicitly. | `universe.history` |
| One guarded path | Every model, including the baselines B-mom_12_1, B-mom_6, B-rev_1, B-lowvol, B-composite and B-ew, is scored through folds sealed by `yearly_folds` (`walk_forward_scores`, `baseline_scores`). Rank IC for a protocol number comes from `window_rank_ic`, which scores only the test rows of those folds. Each fold is checked again before use: its label must end inside the window, training plus embargo must end before the fit, panel dates must match the fold's calendar, and holdout rows need a lock that verifies on disk. Hand-built or modified `Fold` objects are refused. A forward decision uses `forward_fold`, which requires the calendar to end at the decision session, so its label cannot exist yet. | `validation` |
| Label withholding | `fit_predict` never receives the label columns. A declared label column missing from the panel raises an error. An explicit feature allowlist (`feature_columns`) is optional. | (none) |
| Minimum names | A date's rank IC is NaN when fewer than `min_names = 10` members have both a score and a target (the runner uses `MIN_IC_NAMES = 10`). | `metrics.primary` |
| Rank IC | Spearman correlation with average ranks for ties. Mean and t use non-overlapping dates only (greedy, spacing of at least 20 sessions). The bootstrap is a circular moving-block bootstrap by date. | `metrics.primary` |
| V[SR] for the DSR | `max(cross-trial sample variance (ddof 1) of the per-period Sharpe ratios of the run's trials, mean over those trials of (1 - g3*SR + (g4 - 1)/4*SR^2)/(T - 1))`. The floor stops highly correlated trial series from shrinking V and inflating the DSR. N is the HOSE trial count (run + planned) from the registry at evaluation time. | `metrics.multipleTesting` |
| PBO gate statistic | CSCV with 16 equal blocks in time order (the oldest `T mod 16` rows are dropped), all 12870 splits, Sharpe ratio as the statistic, relative rank `w = rank / (N + 1)` with average ranks. `gatePbo = pbo` counts `logit(w) <= 0`, so a median rank counts as overfit (the conservative choice). `pboStrict` (`< 0`) is reported for information only. | `gates.holdout: PBO < 0.3` |
| Holdout lock | `HoldoutLock.acquire` runs before any holdout number is computed. It writes an exclusive sentinel and a per-run record in the protocol evidence root of the main checkout (`docs/research/evidence/vn30-2`, shared by every worktree). `verify_holdout_lock` accepts a lock only under that root, or under an `evidence_root` passed explicitly (for example a test directory). | `holdout.readOnce` |

## Provenance

Every report built on the evaluator records the sha256 of `vn30_panel_eval.py` next to the protocol sha256
(`evaluator_provenance()` / `evaluator_sha256()`). The fields are:

- VN30-2 run `results.json`: `protocol.evaluatorSha256`, and `evaluatorSha256` in `dev-results.json` and the dry-run output
- `multipleTesting.provenance`
- stops-1 and overlay-1 results: `evaluatorSha256`
- VN30-2 forward records: `evaluatorSha256`, next to the existing per-module hashes in `code`

Reports written before this change carry the evaluator hash only in their per-module code hashes: `code.modules`
in VN30-2 and `codeSha256` in overlay-1.

## Guard changes after the official run (no reported number changes)

These guards were added after the review of 2026-11-06:

- the baseline/window guard
- calendar-derived window end
- lock-root check
- missing-label check
- history-start check

The development window was recomputed with the new code from the same snapshot, without touching the holdout. The
IC summaries, folds, gate and proxy portfolio metrics are identical to the development section of run
`20261106T121249Z`. The holdout path runs the same folds and the same scores, so its numbers have the same
meaning, and it was not run again because the holdout is read once.
