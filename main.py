import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

from graph import build_ta_graph
from prediction_service import list_available_models, predict_from_vector

load_dotenv()

# LLM_PROVIDER: "ollama" (default) | "gemini" | "blackbox"
# Ollama: OLLAMA_MODEL (default qwen2.5:1.5b — fits ~3GB RAM), OLLAMA_BASE_URL, optional OLLAMA_NUM_CTX
# Gemini: GOOGLE_API_KEY, optional GEMINI_MODEL (default gemini-2.5-flash)
# Blackbox: BLACKBOX_API_KEY, optional BLACKBOX_BASE_URL (default https://api.blackbox.ai).
# BLACKBOX_MODEL: use an id from GET https://api.blackbox.ai/v1/models (e.g. blackboxai/openai/gpt-5.2).

app = FastAPI(title="Bitcoin AI Analyst (Ollama, Gemini, or Blackbox + RAG context from .NET backend)")


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
    provider = (os.getenv("LLM_PROVIDER") or "ollama").strip().lower()

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="GOOGLE_API_KEY not found in .env (required when LLM_PROVIDER=gemini).",
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.7,
            max_retries=1,
            timeout=20,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        timeout_s = float(os.getenv("OLLAMA_TIMEOUT", "600"))
        num_ctx = os.getenv("OLLAMA_NUM_CTX", "").strip()
        kwargs: dict = {
            "model": model,
            "base_url": base_url,
            "temperature": 0.7,
            "sync_client_kwargs": {"timeout": timeout_s},
            "async_client_kwargs": {"timeout": timeout_s},
        }
        if num_ctx.isdigit():
            kwargs["num_ctx"] = int(num_ctx)
        return ChatOllama(**kwargs)

    if provider == "blackbox":
        api_key = os.getenv("BLACKBOX_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="BLACKBOX_API_KEY not found in .env (required when LLM_PROVIDER=blackbox).",
            )
        from langchain_openai import ChatOpenAI

        base_url = os.getenv("BLACKBOX_BASE_URL", "https://api.blackbox.ai").rstrip("/")
        model = os.getenv("BLACKBOX_MODEL", "blackboxai/openai/gpt-5.2")
        timeout_s = float(os.getenv("BLACKBOX_TIMEOUT", "120"))
        return ChatOpenAI(
            model_name=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.7,
            max_retries=1,
            request_timeout=timeout_s,
        )

    raise HTTPException(
        status_code=500,
        detail=f"Unknown LLM_PROVIDER={provider!r}. Use 'ollama', 'gemini', or 'blackbox'.",
    )


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
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e!s}")


@app.get("/api/predict/models")
async def get_available_models():
    return {"models": list_available_models()}


@app.post("/api/analyze")
async def analyze_crypto(request: AnalyzeRequest):
    symbol = request.symbol.upper()
    if symbol != "BTC":
        raise HTTPException(status_code=400, detail="Only BTC is supported in this version.")

    llm = _build_llm()

    try:
        backend_base_url = os.getenv("BACKEND_BASE_URL", "http://localhost:5197")

        graph = build_ta_graph(llm=llm, backend_base_url=backend_base_url)

        state = {
            "symbol": symbol,
            "news_context": request.news_context or "",
            "tech_context": request.tech_context or "",
        }

        return await graph.ainvoke(state)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"TA graph analysis failed: {e!s}",
        ) from e


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
    if "masterEnsemblePrediction" in ctx:
        evidence_tags.append("Master Ensemble")
    if "archetype" in ctx or "markov" in ctx:
        evidence_tags.append("Markov Transitions")
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

    try:
        llm = _build_llm()
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
    except Exception as e:
        answer_text = f"Không thể tạo giải thích từ LLM: {e!s}"

    return {
        "prompt": prompt,
        "answer": answer_text,
        "evidence_tags": evidence_tags,
    }


@app.post("/api/explain/stream")
async def explain_strategy_stream(request: ExplainRequest):
    prompt = (request.prompt or "").strip()
    ctx = request.market_context or {}

    if not prompt:
        prompt = "Giải thích tổng quan dự báo BTC và các yếu tố kỹ thuật hiện tại."

    evidence_tags = []
    if "masterEnsemblePrediction" in ctx:
        evidence_tags.append("Master Ensemble")
    if "archetype" in ctx or "markov" in ctx:
        evidence_tags.append("Markov Transitions")
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

    async def token_generator():
        try:
            llm = _build_llm()
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

            async for chunk in llm.astream([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    payload = json.dumps({"token": content, "done": False}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"

            payload = json.dumps({"token": "", "done": True, "evidence_tags": evidence_tags}, ensure_ascii=False)
            yield f"data: {payload}\n\n"
        except Exception as e:
            err_payload = json.dumps({"token": f"\n\n[Lỗi kết nối LLM: {e!s}]", "done": True, "evidence_tags": evidence_tags}, ensure_ascii=False)
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

