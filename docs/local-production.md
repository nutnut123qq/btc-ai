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
version/hash, class mapping, and exact runtime library versions). The deployed
BTCUSDT artifact remains quarantined. The separate ML evaluator v2 produced a
hash-valid HGB research result that passed its predictive-only rolling-prior
gate on 13,680 OOS rows; it did not create or promote a deployable model and did
not establish post-cost, forward-paper, or live value. Its source database also
received one qualifying historical row immediately after the run, so exact
reruns require a persisted immutable dataset snapshot rather than a cutoff
alone. A model is servable only after its artifact manifest contains complete
dataset/label provenance and its own passing promotion evidence; `/api/predict`
and the paper trader otherwise fail closed instead of guessing or loading a
legacy fallback.

`paper_trader.py` defaults to `--mode forward-paper`. Each finite poll records at
most one idempotent `PaperObservations` row for the latest finalized BTCUSDT 4h
bar it observes; it does not backfill bars missed during downtime. The row stores
the decision or abstention, observation/availability timestamps, and a live
Binance book-ticker quote with receipt time. The production boundary accepts no
caller-supplied clock or quote, and PostgreSQL rejects backdated clocks. The
decision core is immutable; fill and outcome fields remain null and may mature
only once through separate observers with coherent timestamps. The recorder
never retroactively treats a stored future bar open as a live fill or writes PnL.
Historical simulation requires an explicit `--mode replay`; replay rows persist
`RunMode=replay` plus provenance declaring that they are not prospective paper
results. Scheduled tasks must use only `--mode forward-paper`.

To update dependencies, create a clean Python 3.12 virtual environment, install
`requirements.txt`, run all CI commands, then regenerate `requirements.lock.txt`
with `python -m pip freeze`. Review the lock diff and artifact compatibility test
before accepting it. CI never invokes training or database scripts.
