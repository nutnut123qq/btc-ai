# Phase 1 evidence — Optional LLM

Date: 2026-08-31
Repository: `ai`
Pre-commit HEAD: `867e964` (`867e964e5daadff8d30a4f0cb3482de7df042cba`)
Status: ready for review; not committed or pushed

## Scope delivered

- Added `LLM_PROVIDER=none` as an explicit configuration.
- Added `GET /api/capabilities` with `mlInference`, `llmExplanation`, provider, and an unavailable reason.
- Kept `POST /api/predict` independent from LLM configuration and availability.
- Made `/api/analyze`, `/api/explain`, and `/api/explain/stream` return a structured `503` error envelope when LLM is disabled, unconfigured, or unavailable.
- Removed the fabricated `SIDEWAYS / 50` analysis fallback; provider failures no longer become a trading-style result.
- Sanitized provider initialization, invocation, graph, and streaming failures so raw provider exceptions and configuration details are not returned to callers.
- Prefetched the first stream item so an empty or initially failed stream returns HTTP `503`; failures after streaming begins use a structured SSE error envelope with the request ID.

## Changed files reviewed

- `main.py`
- `graph.py`
- `tests/unit/test_api_endpoints.py`
- `tests/unit/test_graph_parsers.py`

The diff is limited to the Phase 1 optional-LLM behavior and its tests. No environment, credential, model artifact, or unrelated source file is changed. A scan of added lines found no hard-coded API key, bearer token, password, private key, or secret-bearing filename.

## Verification

| Gate | Command | Result |
| --- | --- | --- |
| Unit tests | `venv\\Scripts\\python.exe -m pytest` | 28/28 passed |
| Syntax/import compilation | `venv\\Scripts\\python.exe -m py_compile main.py graph.py` | Passed |
| Dependency consistency | `venv\\Scripts\\python.exe -m pip check` | `No broken requirements found.` |
| Patch hygiene | `git diff --check` | Passed |

The endpoint tests cover provider `none`, capability reporting, prediction without LLM, structured disabled-provider responses, missing-key sanitization, unknown-provider sanitization, runtime failures, empty streams, and mid-stream failures.

## Known non-blocker

Pytest reports 36 `DeprecationWarning` messages from Joblib assigning NumPy array shapes under NumPy 2.5. They originate in the installed dependency while loading current model artifacts; they do not fail the suite and are not introduced by the Phase 1 behavior change.

## Gate conclusion

Phase 1 AI gate is green: quantitative inference remains available without LLM, explanation endpoints fail explicitly and safely, and streaming does not expose raw provider failures.
