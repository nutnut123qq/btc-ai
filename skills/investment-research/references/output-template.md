# Output template (`report.md`) — minimum sections, in this order

Write in the user's language (Vietnamese by default for this repo). Keep every section; write "none" /
"missing — <why>" instead of dropping one.

```markdown
# {SYMBOL} — research {forward|retrospective} as of {as-of, ICT} · horizon {N} sessions

## 1. As-of & data status
- Roles: subagents | run sequentially in-session (say which)
- Snapshot: {snapshotId} · snapshotSha256 {snapshotSha256[:12]} (content hash, not a manifest file hash) · cutoff {effectiveCutoffMs → ICT} · signal bar {openTimeMs → date}
- Readiness: technical {status} · fundamental {status} · quantitative {status} · portfolio-action {status} (reasons)
- PIT: {pit.reconstruction}; barsUpdatedAfterAsOf {per tf}; unavailable sections: {unavailable[]}; warnings: {conditions warnings}
- Fundamentals coverage: periods used {from..to}, stage per group (validated/persisted), gaps that limit claims; series fact count; derived metrics used
- Source documents used: {title · period · consolidation · audit status · documentId · availabilityBasis} or "none"

## 2. Thesis
One paragraph.

## 3. Evidence
- {claim} — [{evidenceIds}]

## 4. Counter-evidence & critic objections
- {objection} — [{evidenceIds}] → adopted / rebutted because …

## 5. Disagreements between analysts
| Topic | View A (evidence) | View B (evidence) | What would settle it | Resolution / unresolved |

## 6. Scenarios (horizon {N} sessions)
| Scenario | Path | Invalidation | Notes |

## 7. Price-direction view
{up|down|sideways|no-view}. Probability: {only with a declared reproducible method: value, method,
methodRef — else "none: no validated/reproducible method"}. Quant base rates: {status/reasons; "unvalidated"}.

## 8. Action ({position-aware|market-only|insufficient-input})
{buy-more|hold|reduce|sell|wait|none} — why; feasibility (T+2 sellable shares, ±7% band, 100-share
lots, declared budget vs shared cash). If inputs are insufficient: what is missing.

## 9. Risks & missing data
- …

## 10. Outcome check
How and when this will be checked (bar / sessions / comparison) — matches `outcomeCheck`.

## 11. References
- Export: {absolute path} (manifest sha256 …)
- Scripts: {scripts/… → outputs}
- Journal: noteId {…} (or "not recorded — <why>")
```
