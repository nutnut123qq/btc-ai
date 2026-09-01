# Phase 4 evidence — AI CI and artifact compatibility

Date: 2026-08-31

- Python 3.12 dependencies have an exact reviewed lock; Linux-only XGBoost NCCL is explicitly pinned with a platform marker.
- CI installs the lock, runs `pip check`, static compile/startup smoke with `LLM_PROVIDER=none`, and the hermetic unit suite. It never trains a model or requires DB/API credentials.
- Docker installs the lock, runs as non-root, starts Uvicorn without `--reload`, and excludes local environment/key files from its build context.
- Production inference now resolves only the exact registry `active` artifact. Filename/glob/legacy fallbacks are removed.
- Serving requires artifact SHA-256, feature-schema version/hash, class mapping, and exact runtime library versions in the sidecar manifest.

Verification:

- `pip check`: passed.
- Provider-none startup smoke: passed.
- Pytest: 37/37 passed (plus 4 parameterized subtests).
- All three current BTC/ETH/SOL binaries load under the pinned runtime and return a three-class `predict_proba` result.
- Current legacy sidecars lack the required immutable lineage fields, so all three are deliberately quarantined: `/api/capabilities` reports `mlInference=false`, `/api/predict/models` exposes no serveable artifact, and `/api/predict` returns structured `503 MODEL_ARTIFACT_INCOMPATIBLE` without a raw exception.

The NumPy 2.5/joblib load path emits known deprecation warnings for the old binaries. Phase 5 must rebuild manifests/artifacts rather than weakening this gate.
