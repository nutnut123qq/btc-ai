import json
import logging
import os
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

from graph import build_ta_graph
from prediction_service import ModelArtifactIncompatibleError, list_available_models, predict_from_vector

load_dotenv()

# LLM_PROVIDER: "none" | "ollama" (default) | "gemini" | "blackbox"
# Ollama: OLLAMA_MODEL (default qwen2.5:1.5b — fits ~3GB RAM), OLLAMA_BASE_URL, optional OLLAMA_NUM_CTX
# Gemini: GOOGLE_API_KEY, optional GEMINI_MODEL (default gemini-2.5-flash)
# Blackbox: BLACKBOX_API_KEY, optional BLACKBOX_BASE_URL (default https://api.blackbox.ai).
# BLACKBOX_MODEL: use an id from GET https://api.blackbox.ai/v1/models (e.g. blackboxai/openai/gpt-5.2).

app = FastAPI(title="Bitcoin AI Analyst (Ollama, Gemini, or Blackbox + RAG context from .NET backend)")
logger = logging.getLogger(__name__)


class LlmProviderError(Exception):
    def __init__(self, code: str, message: str, retryable: bool):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


def _error_envelope(error: LlmProviderError, request_id: str | None = None) -> dict:
    return {
        "code": error.code,
        "message": error.message,
        "retryable": error.retryable,
        "requestId": request_id or str(uuid4()),
    }


@app.exception_handler(LlmProviderError)
async def llm_provider_error_handler(request: Request, error: LlmProviderError):
    request_id = request.headers.get("x-request-id") or str(uuid4())
    return JSONResponse(status_code=503, content=_error_envelope(error, request_id))


@app.exception_handler(ModelArtifactIncompatibleError)
async def model_artifact_error_handler(request: Request, error: ModelArtifactIncompatibleError):
    request_id = request.headers.get("x-request-id") or str(uuid4())
    logger.warning("Model artifact unavailable code=MODEL_ARTIFACT_INCOMPATIBLE")
    return JSONResponse(status_code=503, content={
        "code": "MODEL_ARTIFACT_INCOMPATIBLE",
        "message": "Không có model artifact đủ bằng chứng tương thích để suy luận.",
        "retryable": False,
        "requestId": request_id,
    })


def _provider_name() -> str:
    return (os.getenv("LLM_PROVIDER") or "ollama").strip().lower()


def _not_configured_error() -> LlmProviderError:
    return LlmProviderError(
        "LLM_NOT_CONFIGURED",
        "Tính năng giải thích LLM chưa được cấu hình.",
        False,
    )


def _unavailable_error(retryable: bool = True) -> LlmProviderError:
    return LlmProviderError(
        "LLM_PROVIDER_UNAVAILABLE",
        "Nhà cung cấp LLM hiện không khả dụng.",
        retryable,
    )


def _log_llm_failure(endpoint: str) -> None:
    logger.warning(
        "LLM request unavailable endpoint=%s provider=%s code=LLM_PROVIDER_UNAVAILABLE",
        endpoint,
        _provider_name(),
    )


def _provider_capability() -> tuple[str, bool, str | None]:
    provider = _provider_name()
    if provider == "none":
        return provider, False, "LLM provider is disabled."
    if provider == "gemini" and not (os.getenv("GOOGLE_API_KEY") or "").strip():
        return provider, False, "Gemini API key is not configured."
    if provider == "blackbox" and not (os.getenv("BLACKBOX_API_KEY") or "").strip():
        return provider, False, "Blackbox API key is not configured."
    if provider not in {"ollama", "gemini", "blackbox"}:
        return provider, False, "Configured LLM provider is not supported."
    return provider, True, None


class AnalyzeRequest(BaseModel):
    symbol: str
    news_context: str = Field(default="", description="Retrieved news text from backend RAG")
    tech_context: str = Field(default="", description="Market/technical summary from backend (e.g. Binance)")


class PredictRequest(BaseModel):
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    window_size: int = 5
    horizon: str = "1h"
    feature_vector: list[float]
    model_name: str | None = None


def _build_llm():
    provider, available, _ = _provider_capability()
    if not available:
        if provider in {"none", "gemini", "blackbox"}:
            raise _not_configured_error()
        raise _unavailable_error(retryable=False)

    try:
        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                google_api_key=os.environ["GOOGLE_API_KEY"],
                temperature=0.7,
                max_retries=1,
                timeout=20,
            )

        if provider == "ollama":
            from langchain_ollama import ChatOllama

            timeout_s = float(os.getenv("OLLAMA_TIMEOUT", "600"))
            num_ctx = os.getenv("OLLAMA_NUM_CTX", "").strip()
            kwargs: dict = {
                "model": os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                "temperature": 0.7,
                "sync_client_kwargs": {"timeout": timeout_s},
                "async_client_kwargs": {"timeout": timeout_s},
            }
            if num_ctx.isdigit():
                kwargs["num_ctx"] = int(num_ctx)
            return ChatOllama(**kwargs)

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model_name=os.getenv("BLACKBOX_MODEL", "blackboxai/openai/gpt-5.2"),
            openai_api_key=os.environ["BLACKBOX_API_KEY"],
            openai_api_base=os.getenv("BLACKBOX_BASE_URL", "https://api.blackbox.ai").rstrip("/"),
            temperature=0.7,
            max_retries=1,
            request_timeout=float(os.getenv("BLACKBOX_TIMEOUT", "120")),
        )
    except Exception:
        _log_llm_failure("provider_initialization")
        raise _unavailable_error() from None


@app.get("/api/capabilities")
async def capabilities():
    provider, llm_explanation, reason = _provider_capability()
    ml_reason = None
    try:
        ml_inference = bool(list_available_models())
        if not ml_inference:
            ml_reason = "No manifest-compatible model artifact is available."
    except Exception:
        logger.warning("Model registry unavailable endpoint=capabilities code=MODEL_REGISTRY_UNAVAILABLE")
        ml_inference = False
        ml_reason = "Model registry is unavailable."
    return {
        "mlInference": ml_inference,
        # Configuration capability only; provider reachability is checked by each LLM request.
        "llmExplanation": llm_explanation,
        "provider": provider,
        "reason": reason,
        "mlReason": ml_reason,
    }


@app.post("/api/predict")
async def predict(request: PredictRequest):
    try:
        result = predict_from_vector(
            feature_vector=request.feature_vector,
            symbol=request.symbol,
            timeframe=request.timeframe,
            window_size=request.window_size,
            horizon=request.horizon,
            model_name=request.model_name,
        )
        return result
    except ModelArtifactIncompatibleError:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected prediction failure")
        raise ModelArtifactIncompatibleError("Unexpected prediction failure.") from e


@app.get("/api/predict/models")
async def get_available_models():
    return {"models": list_available_models()}


@app.post("/api/analyze")
async def analyze_crypto(request: AnalyzeRequest):
    symbol = request.symbol.upper()
    if symbol != "BTC":
        raise HTTPException(status_code=400, detail="Only BTC is supported in this version.")

    try:
        llm = _build_llm()
        backend_base_url = os.getenv("BACKEND_BASE_URL", "http://localhost:5197")

        graph = build_ta_graph(llm=llm, backend_base_url=backend_base_url)

        state = {
            "symbol": symbol,
            "news_context": request.news_context or "",
            "tech_context": request.tech_context or "",
        }

        return await graph.ainvoke(state)

    except LlmProviderError:
        raise
    except Exception:
        _log_llm_failure("analyze")
        raise _unavailable_error() from None


class ExplainRequest(BaseModel):
    prompt: str = ""
    market_context: dict = Field(default_factory=dict)


@app.post("/api/explain")
async def explain_strategy(request: ExplainRequest):
    prompt = (request.prompt or "").strip()
    ctx = request.market_context or {}

    if not prompt and not ctx:
        raise HTTPException(
            status_code=400,
            detail="Missing required 'prompt' or 'market_context' payload. Please provide query prompt or market context.",
        )

    if not prompt:
        prompt = "Giải thích tổng quan dự báo BTC và các yếu tố kỹ thuật hiện tại."

    # Extract key metadata for tagging
    evidence_tags = []
    if "masterEnsemblePrediction" in ctx or "multiLayerEnsemble" in ctx:
        evidence_tags.append("Multi-Layer Ensemble")
    if "archetype" in ctx or "momentum" in ctx or "markov" in ctx:
        evidence_tags.append("Price Momentum")
    if "regime" in ctx:
        evidence_tags.append("Market Regime")
    if "confluence" in ctx:
        evidence_tags.append("Multi-TF Confluence")
    if "smc" in ctx or "volumeProfile" in ctx or "smartMoney" in ctx:
        evidence_tags.append("VPVR & SMC")
    if "sentiment" in ctx:
        evidence_tags.append("Market Sentiment")
    if not evidence_tags:
        evidence_tags = ["Market Analysis", "Technical Strategy"]

    llm = _build_llm()
    try:
        system_prompt = (
            "Bạn là chuyên gia phân tích chiến lược AI Bitcoin (Explainable AI - XAI). "
            "Nhiệm vụ của bạn là giải thích rõ ràng, mạch lạc các luận điểm kỹ thuật và dữ liệu thị trường "
            "dựa trên ngữ cảnh cung cấp. Không bịa đặt số liệu ngoài context. Định dạng câu trả lời bằng Markdown tiếng Việt."
        )
        human_prompt = f"""Dữ liệu ngữ cảnh thị trường (Market Context):
{json.dumps(ctx, ensure_ascii=False, indent=2) if ctx else "Không có context chi tiết"}

Câu hỏi / Yêu cầu của người dùng:
{prompt}

Hãy phân tích và giải thích:
1. Đánh giá xu hướng và độ tin cậy từ dữ liệu hiện có.
2. Phân tích các bằng chứng kỹ thuật (mẫu nến, chế độ thị trường, hội tụ đa khung, cấu trúc thanh khoản SMC/VPVR nếu có).
3. Rủi ro cần lưu ý và khuyến nghị hành động ngắn gọn."""

        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        )
        answer_text = response.content
    except Exception:
        _log_llm_failure("explain")
        raise _unavailable_error() from None

    return {
        "prompt": prompt,
        "answer": answer_text,
        "evidence_tags": evidence_tags,
    }


@app.post("/api/explain/stream")
async def explain_strategy_stream(request: ExplainRequest, http_request: Request):
    prompt = (request.prompt or "").strip()
    ctx = request.market_context or {}

    if not prompt:
        prompt = "Giải thích tổng quan dự báo BTC và các yếu tố kỹ thuật hiện tại."

    evidence_tags = []
    if "masterEnsemblePrediction" in ctx or "multiLayerEnsemble" in ctx:
        evidence_tags.append("Multi-Layer Ensemble")
    if "archetype" in ctx or "momentum" in ctx or "markov" in ctx:
        evidence_tags.append("Price Momentum")
    if "regime" in ctx:
        evidence_tags.append("Market Regime")
    if "confluence" in ctx:
        evidence_tags.append("Multi-TF Confluence")
    if "smc" in ctx or "volumeProfile" in ctx or "smartMoney" in ctx:
        evidence_tags.append("VPVR & SMC")
    if "sentiment" in ctx:
        evidence_tags.append("Market Sentiment")
    if not evidence_tags:
        evidence_tags = ["Market Analysis", "Technical Strategy"]

    llm = _build_llm()
    request_id = http_request.headers.get("x-request-id") or str(uuid4())
    system_prompt = (
        "Bạn là chuyên gia phân tích chiến lược AI Bitcoin (Explainable AI - XAI). "
        "Nhiệm vụ của bạn là giải thích rõ ràng, mạch lạc các luận điểm kỹ thuật và dữ liệu thị trường "
        "dựa trên ngữ cảnh cung cấp. Không bịa đặt số liệu ngoài context. Định dạng câu trả lời bằng Markdown tiếng Việt."
    )
    human_prompt = f"""Dữ liệu ngữ cảnh thị trường (Market Context):
{json.dumps(ctx, ensure_ascii=False, indent=2) if ctx else "Không có context chi tiết"}

Câu hỏi / Yêu cầu của người dùng:
{prompt}

Hãy phân tích và giải thích:
1. Đánh giá xu hướng và độ tin cậy từ dữ liệu hiện có.
2. Phân tích các bằng chứng kỹ thuật (mẫu nến, chế độ thị trường, hội tụ đa khung, cấu trúc thanh khoản SMC/VPVR nếu có).
3. Rủi ro cần lưu ý và khuyến nghị hành động ngắn gọn."""
    stream = llm.astream([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]).__aiter__()
    try:
        first_chunk = await anext(stream)
    except StopAsyncIteration:
        _log_llm_failure("explain_stream_empty")
        raise _unavailable_error() from None
    except Exception:
        _log_llm_failure("explain_stream_initial")
        raise _unavailable_error() from None

    async def token_generator():
        try:
            if first_chunk is not None:
                content = first_chunk.content if hasattr(first_chunk, "content") else str(first_chunk)
                if content:
                    payload = json.dumps({"token": content, "done": False}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"

            async for chunk in stream:
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    payload = json.dumps({"token": content, "done": False}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"

            payload = json.dumps({"token": "", "done": True, "evidence_tags": evidence_tags}, ensure_ascii=False)
            yield f"data: {payload}\n\n"
        except Exception:
            _log_llm_failure("explain_stream_midstream")
            err_payload = json.dumps(
                {
                    "error": _error_envelope(_unavailable_error(), request_id),
                    "done": True,
                    "evidence_tags": evidence_tags,
                },
                ensure_ascii=False,
            )
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


class RlPredictRequest(BaseModel):
    state_features: list[float] = [0.5, 0.5, 0.5, 0.5, 0.5]


class RlTrainRequest(BaseModel):
    state_features: list[float]
    action_name: str
    reward: float
    next_state_features: list[float]


@app.post("/api/rl/predict")
async def rl_predict(req: RlPredictRequest):
    from rl_agent import global_rl_agent
    action_name, confidence = global_rl_agent.get_action(req.state_features)
    return {
        "action": action_name,
        "confidence": round(confidence, 3),
        "status": "success"
    }


@app.post("/api/rl/train")
async def rl_train(req: RlTrainRequest):
    from rl_agent import global_rl_agent
    global_rl_agent.update_q_value(req.state_features, req.action_name, req.reward, req.next_state_features)
    return {
        "message": "Q-value updated successfully",
        "status": "success"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

