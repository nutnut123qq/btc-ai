# AI service: pinned install and local production start

The service targets Python 3.12. `requirements.txt` describes direct dependency
ranges; CI and production builds install the fully resolved
`requirements.lock.txt` for repeatability.

```powershell
python -m pip install --requirement requirements.lock.txt
python -m pip check
$env:LLM_PROVIDER = "none"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Production-like runs must not use `--reload`. `LLM_PROVIDER=none` disables only
LLM explanations. Model inference is reported available only when an `active`
registry artifact has a complete manifest (artifact hash, feature-schema
version/hash, class mapping, and exact runtime library versions). The current
BTCUSDT artifact was rebuilt with that lineage and is active. ETHUSDT and
SOLUSDT remain quarantined because their local training provenance cannot be
proven. `/api/predict` and the paper trader fail closed for those unavailable
artifacts instead of guessing or loading a legacy fallback.

To update dependencies, create a clean Python 3.12 virtual environment, install
`requirements.txt`, run all CI commands, then regenerate `requirements.lock.txt`
with `python -m pip freeze`. Review the lock diff and artifact compatibility test
before accepting it. CI never invokes training or database scripts.
