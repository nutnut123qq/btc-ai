import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

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

        return graph.invoke(state)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"TA graph analysis failed: {e!s}",
        ) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
