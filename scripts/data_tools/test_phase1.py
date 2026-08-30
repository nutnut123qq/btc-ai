import asyncio
import sys
import httpx
from main import app

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

async def run_tests():
    print("=== STARTING AI SERVICE PHASE 1 VERIFICATION ===")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 1. Test /api/explain validation with empty body -> should return 400
        print("\n--- Test 1: /api/explain validation (Empty payload) ---")
        r_empty = await client.post("/api/explain", json={"prompt": "", "market_context": {}})
        print(f"Status Code: {r_empty.status_code}")
        print(f"Response: {r_empty.json()}")
        assert r_empty.status_code == 400, f"Expected 400, got {r_empty.status_code}"
        print("-> PASS: Empty payload rejected with 400")

        # 2. Test /api/explain with context
        print("\n--- Test 2: /api/explain dynamic response ---")
        sample_context = {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "currentPrice": 67500.0,
            "masterEnsemblePrediction": {"finalDirection": "Bullish", "ensembleConfidence": 0.85},
            "regime": {"regimeType": "TrendingUp"},
            "confluence": {"confluenceScore": 88},
            "smc": {"fvgActive": True}
        }
        r_explain = await client.post("/api/explain", json={"prompt": "Giải thích xu hướng BTC", "market_context": sample_context})
        print(f"Status Code: {r_explain.status_code}")
        data = r_explain.json()
        print(f"Evidence Tags: {data.get('evidence_tags')}")
        print(f"Answer Sample:\n{data.get('answer', '')[:250]}...")
        assert r_explain.status_code == 200, f"Expected 200, got {r_explain.status_code}"
        assert len(data.get("evidence_tags", [])) > 0, "Expected evidence tags"
        print("-> PASS: /api/explain generated dynamic explanation")

        # 3. Test /api/predict/models
        print("\n--- Test 3: /api/predict/models ---")
        r_models = await client.get("/api/predict/models")
        models = r_models.json().get("models", [])
        print(f"Models Available: {len(models)}")
        assert r_models.status_code == 200
        print("-> PASS: /api/predict/models returned model list")

        # 4. Test /api/analyze async pipeline
        print("\n--- Test 4: /api/analyze async LangGraph pipeline ---")
        r_analyze = await client.post("/api/analyze", json={
            "symbol": "BTC",
            "news_context": "Bitcoin sets new institutional inflow record with $1.2B weekly volume.",
            "tech_context": "BTC trading above EMA 200 on 1h with RSI at 62, Bullish Engulfing pattern detected."
        })
        print(f"Status Code: {r_analyze.status_code}")
        adata = r_analyze.json()
        print(f"Forecast: {adata.get('forecast')}")
        print(f"Confidence: {adata.get('confidence')}")
        print(f"Reasoning: {adata.get('reasoning')[:200]}...")
        assert r_analyze.status_code == 200, f"Expected 200, got {r_analyze.status_code}"
        print("-> PASS: /api/analyze executed async pipeline successfully")

    print("\n=== ALL PHASE 1 TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    asyncio.run(run_tests())
