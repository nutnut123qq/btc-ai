import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from trading_config import DEFAULT_SYMBOL
import current_technical_conditions
import equity_base_rates
import equity_llm
import vietnam_market_calendar

load_dotenv()

# VN-only (VNONLY-1): the BTC routes (/api/predict*, /api/analyze, /api/explain*, /api/rl/*) are no longer
# served, so graph.py (LangGraph), prediction_service.py and rl_agent.py are not imported. Their code stays
# on disk; restore the routes from git history to bring them back.
RETIRED_BTC_REASON = "BTC ML inference and LLM explanation routes are retired (VN-only service)."

app = FastAPI(title="AI service for the VN equity lane (current conditions, base rates)")
logger = logging.getLogger(__name__)


def _provider_name() -> str:
    return (os.getenv("LLM_PROVIDER") or "ollama").strip().lower()


@app.get("/api/capabilities")
async def capabilities():
    # Shape kept for existing readers (backend AiChatController, ops status probe).
    return {
        "mlInference": False,
        "llmExplanation": False,
        "provider": _provider_name(),
        "reason": RETIRED_BTC_REASON,
        "mlReason": RETIRED_BTC_REASON,
    }


@app.get("/api/current-conditions")
def current_conditions(timeframe: str = "4h", symbol: str = DEFAULT_SYMBOL):
    try:
        return current_technical_conditions.get_current_conditions(timeframe, symbol=symbol)
    except (
        current_technical_conditions.UnknownSymbolError,
        current_technical_conditions.UnknownTimeframeError,
        current_technical_conditions.UnsupportedCapabilityError,
    ) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except current_technical_conditions.ConditionsUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/api/equity-base-rates/build")
def build_equity_base_rates(symbol: str):
    """Build the equity-base-rates-artifact-v1 for one decision equity from the DB.

    Writes ``{evidence root}/equity-base-rates/{SYMBOL}/base-rates.json`` atomically
    and returns a compact summary.  Non-decision/unknown symbols -> 400; no or too
    few finalized candles -> 422 with ``reason``; DB unreachable -> 503.
    """
    try:
        canonical = equity_base_rates.resolve_decision_symbol(symbol)
    except equity_base_rates.NonDecisionSymbolError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "UNSUPPORTED_CAPABILITY", "reason": "not-decision-symbol", "message": str(exc)},
        )
    out_root = os.getenv("EQUITY_BASE_RATES_EVIDENCE_ROOT") or str(equity_base_rates.DEFAULT_EVIDENCE_ROOT)
    try:
        rows = equity_base_rates.load_candles_db(canonical, equity_base_rates.now_ms())
    except Exception as exc:
        logger.warning("Equity base-rates candle load failed symbol=%s error=%s", canonical, type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail={"code": "CANDLES_UNAVAILABLE", "reason": "candle-load-failed", "message": type(exc).__name__},
        )
    try:
        result = equity_base_rates.build_and_write(canonical, rows=rows, out_root=out_root)
    except equity_base_rates.InsufficientDataError as exc:
        if exc.reason in (vietnam_market_calendar.CALENDAR_UNAVAILABLE, vietnam_market_calendar.CALENDAR_COVERAGE_EXPIRED):
            # Fail closed: no HOSE calendar (MarketCalendarDays empty/unreadable) is a
            # dependency outage, not bad input.
            raise HTTPException(
                status_code=503,
                detail={"code": exc.reason.upper().replace("-", "_"), "reason": exc.reason, "message": str(exc)},
            )
        raise HTTPException(
            status_code=422,
            detail={"code": "INSUFFICIENT_DATA", "reason": exc.reason, "message": str(exc)},
        )
    except equity_base_rates.BaseRatesError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_DATA", "reason": "invalid-candles", "message": str(exc)},
        )
    return equity_base_rates.summarize(result["artifact"])


class EquityLlmRunRequest(BaseModel):
    symbol: str
    dossier: dict
    configs: list[dict]


@app.post("/api/equity-llm/run")
def run_equity_llm_endpoint(request: EquityLlmRunRequest):
    """Researcher + judge LLM run for one decision equity (spec §7).

    200 for succeeded/failed (the run record is in the body), 503 ``LLM_UNAVAILABLE``
    when the provider/key is unavailable, 504 ``LLM_TIMEOUT`` on SDK timeout, 400 for
    non-decision symbols or malformed dossier/configs.
    """
    # Retired lane (EQX v2 B): refuse before building any client or touching the network.
    if not equity_llm.llm_enabled():
        return JSONResponse(status_code=503, content={
            "status": equity_llm.STATUS_UNAVAILABLE, "code": "LLM_DISABLED",
            "reason": "in-app LLM runs are disabled (EQUITY_LLM_ENABLED is not 1); research runs in an interactive session",
        })
    try:
        result = equity_llm.run_equity_llm(request.dossier, request.configs, symbol=request.symbol)
    except equity_llm.InvalidLlmInputError as exc:
        raise HTTPException(status_code=400, detail={"code": "INVALID_LLM_INPUT", "message": str(exc)})
    if result["status"] == equity_llm.STATUS_UNAVAILABLE:
        return JSONResponse(status_code=503, content={**result, "code": "LLM_UNAVAILABLE"})
    if result.get("errorCode") == equity_llm.ERROR_TIMEOUT:
        return JSONResponse(status_code=504, content={**result, "code": "LLM_TIMEOUT"})
    return result


@app.get("/api/equity-llm/status")
def equity_llm_status():
    return equity_llm.status()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
