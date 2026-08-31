import json
import re
from typing import Any, TypedDict, List, Dict, Optional

import httpx
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END


class TAState(TypedDict, total=False):
    symbol: str

    # Contexts (retrieved by nodes)
    news_context: str
    tech_context: str

    # Agent texts
    news_agent_text: str
    tech_agent_text: str
    bull_arguments_text: str
    bear_arguments_text: str
    research_manager_text: str
    trader_text: str
    aggressive_debator_text: str
    neutral_debator_text: str
    conservative_debator_text: str

    # Final contract to return to UI
    forecast: str
    confidence: int
    reasoning: str
    debate_summary: Dict[str, str]

    news_evidence: List[Dict[str, Any]]
    tech_evidence: Dict[str, Any]
    risk_conditions: List[Dict[str, Any]]


def _strip_json_fences(text: str) -> str:
    # Removes common ```json fences without trying to be too clever.
    text = text.strip()
    text = text.replace("```json", "").replace("```", "")
    return text.strip()


def _extract_first_json_object(text: str) -> str:
    # Fallback parser: attempt to extract the first {...} block.
    stripped = text.strip()
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    return match.group(0) if match else stripped


def _parse_json_strict(text: str) -> Dict[str, Any]:
    cleaned = _strip_json_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return json.loads(_extract_first_json_object(cleaned))


async def _retrieve_news_context(*, backend_base_url: str, query: str, top_k: int) -> str:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                f"{backend_base_url}/api/rag/news-context",
                params={"query": query, "topK": top_k},
            )
            r.raise_for_status()
            data = r.json()
            return data.get("news_context", "") or ""
    except Exception:
        return ""


async def _retrieve_tech_summary(*, backend_base_url: str, interval: str, limit: int) -> str:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                f"{backend_base_url}/api/market/btc/tech-summary",
                params={"interval": interval, "limit": limit},
            )
            r.raise_for_status()
            data = r.json()
            return data.get("tech_context", "") or ""
    except Exception:
        return ""


def build_ta_graph(*, llm: Any, backend_base_url: str):
    """
    Build an async multi-node TA pipeline inspired by TradingAgents.

    Note: We keep the final output schema aligned with the UI contract.
    """

    async def news_analyst_node(state: TAState) -> Dict[str, Any]:
        symbol = state["symbol"]
        query = f"{symbol} BTC cryptocurrency market news regulation ETF price"

        # Tool retrieval (News)
        news_context = state.get("news_context", "") or ""
        if not news_context:
            try:
                news_context = await _retrieve_news_context(
                    backend_base_url=backend_base_url, query=query, top_k=8
                )
            except Exception:
                news_context = ""

        # Agent reasoning (text only; JSON produced at RiskJudgeNode)
        system_prompt = (
            "Bạn là News Analyst (crypto/BTC). "
            "Bạn CHỈ được dùng NEWS_CONTEXT dưới đây. "
            "Nếu không có đủ tin, hãy nói rõ thiếu dữ liệu và không bịa."
        )
        human_prompt = (
            f"=== NEWS_CONTEXT ===\n{news_context}\n\n"
            "Tóm tắt sentiment và các tin/ý chính nổi bật ảnh hưởng tới 24h tới. "
            "Cuối cùng ghi ngắn gọn: Bull case và Bear case (mỗi dòng 1-2 câu)."
        )
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        )
        news_agent_text = response.content
        return {
            "news_context": news_context,
            "news_agent_text": news_agent_text,
        }

    async def tech_analyst_node(state: TAState) -> Dict[str, Any]:
        interval = "1h"
        limit = 48

        # Tool retrieval (Tech)
        tech_context = state.get("tech_context", "") or ""
        if not tech_context:
            try:
                tech_context = await _retrieve_tech_summary(
                    backend_base_url=backend_base_url, interval=interval, limit=limit
                )
            except Exception:
                tech_context = ""

        system_prompt = (
            "Bạn là Tech Analyst (crypto/BTC). "
            "Bạn CHỈ được dùng TECH_CONTEXT dưới đây. "
            "Nếu dữ liệu không đủ, hãy ghi rõ n/a và không bịa."
        )
        human_prompt = (
            f"=== TECH_CONTEXT ===\n{tech_context}\n\n"
            "Diễn giải xu hướng ngắn hạn (~24h): biến động/range, momentum và ý nghĩa RSI(14) nếu có. "
            "Nếu context có các mẫu nến (Hammer, Engulfing, Morning Star, v.v.) hoặc bất thường volume, hãy phân tích ý nghĩa của chúng. "
            "Cuối cùng ghi ngắn gọn: Bull case và Bear case (mỗi dòng 1-2 câu)."
        )
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        )
        tech_agent_text = response.content
        return {
            "tech_context": tech_context,
            "tech_agent_text": tech_agent_text,
        }

    async def bull_researcher_node(state: TAState) -> Dict[str, Any]:
        system_prompt = (
            "Bạn là Bull Researcher. "
            "Dựa vào news_agent_text và tech_agent_text, "
            "lập luận theo hướng tăng/UP trong ~24h (nếu đủ dữ liệu). "
            "Không bịa."
        )
        human_prompt = (
            f"NEWS AGENT (text):\n{state.get('news_agent_text','')}\n\n"
            f"TECH AGENT (text):\n{state.get('tech_agent_text','')}\n\n"
            "Trả về chuỗi lập luận bull (2-5 gạch đầu dòng) + 1 kết luận tóm tắt."
        )
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        )
        bull_text = response.content
        return {"bull_arguments_text": bull_text}

    async def bear_researcher_node(state: TAState) -> Dict[str, Any]:
        system_prompt = (
            "Bạn là Bear Researcher. "
            "Dựa vào news_agent_text và tech_agent_text, "
            "lập luận theo hướng giảm trong ~24h (hoặc yếu đi). "
            "Không bịa."
        )
        human_prompt = (
            f"NEWS AGENT (text):\n{state.get('news_agent_text','')}\n\n"
            f"TECH AGENT (text):\n{state.get('tech_agent_text','')}\n\n"
            "Trả về chuỗi lập luận bear (2-5 gạch đầu dòng) + 1 kết luận tóm tắt."
        )
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        )
        bear_text = response.content
        return {"bear_arguments_text": bear_text}

    async def research_manager_node(state: TAState) -> Dict[str, Any]:
        system_prompt = (
            "Bạn là Research Manager. "
            "Tổng hợp bull_arguments_text và bear_arguments_text. "
            "Chỉ ra điểm nào thuyết phục hơn và cần theo dõi thêm gì. "
            "Không bịa."
        )
        human_prompt = (
            f"BULL:\n{state.get('bull_arguments_text','')}\n\n"
            f"BEAR:\n{state.get('bear_arguments_text','')}\n\n"
            "Viết 1 đoạn tổng hợp + 3 bullet: điều kiện ủng hộ bull, điều kiện ủng hộ bear, và điểm mơ hồ."
        )
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        )
        manager_text = response.content
        return {"research_manager_text": manager_text}

    async def trader_node(state: TAState) -> Dict[str, Any]:
        system_prompt = (
            "Bạn là Trader. "
            "Dựa vào research_manager_text, news_agent_text và tech_agent_text, "
            "đưa ra hướng dự báo và mức tự tin cho ~24h. "
            "Không bịa."
        )
        human_prompt = (
            f"RESEARCH_MANAGER:\n{state.get('research_manager_text','')}\n\n"
            "Hãy viết trader summary gồm: (1) forecast gợi ý (UP/DOWN/SIDEWAYS/DOWN_SLIGHTLY), "
            "(2) confidence gợi ý (1-100), (3) final decision ngắn, "
            "(4) reasoning 3-6 câu."
        )
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        )
        trader_text = response.content
        return {"trader_text": trader_text}

    async def aggressive_debator_node(state: TAState) -> Dict[str, Any]:
        system_prompt = (
            "Bạn là Aggressive Debator (về rủi ro). "
            "Dựa vào news_context/tech_context và các phân tích trước, "
            "phản biện hướng dự báo và liệt kê rủi ro có thể đảo chiều nhanh. "
            "Không bịa."
        )
        human_prompt = (
            f"NEWS_CONTEXT:\n{state.get('news_context','')}\n\n"
            f"TECH_CONTEXT:\n{state.get('tech_context','')}\n\n"
            f"TRADER_SUMMARY:\n{state.get('trader_text','')}\n\n"
            "Trả về 3-6 bullet risk triggers (mỗi bullet nêu trigger + vì sao nguy hiểm)."
        )
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        )
        text = response.content
        return {"aggressive_debator_text": text}

    async def neutral_debator_node(state: TAState) -> Dict[str, Any]:
        system_prompt = (
            "Bạn là Neutral Debator (về rủi ro). "
            "Đánh giá cân bằng rủi pro vs cơ hội, "
            "liệt kê những điều kiện có thể khiến dự báo sai. "
            "Không bịa."
        )
        human_prompt = (
            f"RESEARCH_MANAGER:\n{state.get('research_manager_text','')}\n\n"
            f"TRADER_SUMMARY:\n{state.get('trader_text','')}\n\n"
            "Trả về 3-6 bullet risk uncertainty (trigger/điều kiện + mức độ tác động)."
        )
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        )
        text = response.content
        return {"neutral_debator_text": text}

    async def conservative_debator_node(state: TAState) -> Dict[str, Any]:
        system_prompt = (
            "Bạn là Conservative Debator (về rủi ro). "
            "Hãy nêu các kịch bản rủi ro theo hướng xấu nhất hợp lý dựa trên context, "
            "và gợi ý cách phòng ngừa. "
            "Không bịa."
        )
        human_prompt = (
            f"TECH_CONTEXT:\n{state.get('tech_context','')}\n\n"
            f"NEWS_CONTEXT:\n{state.get('news_context','')}\n\n"
            "Trả về 2-5 kịch bản xấu (worst-case scenarios) + mỗi kịch bản kèm 1-2 biện pháp theo dõi/giảm thiểu."
        )
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        )
        text = response.content
        return {"conservative_debator_text": text}

    async def risk_judge_node(state: TAState) -> Dict[str, Any]:
        symbol = state["symbol"]
        news_context = state.get("news_context", "") or ""
        tech_context = state.get("tech_context", "") or ""

        system_prompt = (
            f"Bạn là Risk Judge cho {symbol}. "
            "Bạn KHÔNG được bịa tin tức hay số liệu ngoài hai khối context đã cho. "
            "Trả về TUYỆT ĐỐI JSON hợp lệ, không markdown, không text rác ngoài JSON."
        )
        human_prompt = (
            f"=== NEWS_CONTEXT ===\n{news_context}\n\n"
            f"=== TECH_CONTEXT ===\n{tech_context}\n\n"
            "Các phân tích trước (để tham khảo lập luận):\n"
            f"- news_agent_text:\n{state.get('news_agent_text','')}\n\n"
            f"- tech_agent_text:\n{state.get('tech_agent_text','')}\n\n"
            f"- bull_arguments_text:\n{state.get('bull_arguments_text','')}\n\n"
            f"- bear_arguments_text:\n{state.get('bear_arguments_text','')}\n\n"
            f"- research_manager_text:\n{state.get('research_manager_text','')}\n\n"
            f"- trader_text:\n{state.get('trader_text','')}\n\n"
            f"- aggressive_debator_text:\n{state.get('aggressive_debator_text','')}\n\n"
            f"- neutral_debator_text:\n{state.get('neutral_debator_text','')}\n\n"
            f"- conservative_debator_text:\n{state.get('conservative_debator_text','')}\n\n"
            "Return JSON theo schema sau (bắt buộc đủ các field):\n"
            "{\n"
            '  "forecast": "UP"|"DOWN"|"SIDEWAYS"|"DOWN_SLIGHTLY",\n'
            '  "confidence": int (1-100),\n'
            '  "debate_summary": { "news_agent": string, "tech_agent": string, "final_decision": string },\n'
            '  "reasoning": string,\n'
            '  "news_evidence": [ { "title": string, "link": string, "snippet": string, "sentiment": string, "why_it_matters": string } ],\n'
            '  "tech_evidence": { "first_close": number|null, "last_close": number|null, "change_pct": number|null, "period_high": number|null, "period_low": number|null, "rsi": number|null },\n'
            '  "risk_conditions": [ { "trigger": string, "severity": "HIGH"|"MEDIUM"|"LOW", "what_to_watch": string, "mitigation_hint": string } ]\n'
            "}\n\n"
            "Quy tắc:\n"
            "- Nếu không rút được evidence từ context, trả về [] hoặc null tương ứng.\n"
            "- news_evidence tối đa 5 item.\n"
            "- risk_conditions tối đa 5 item.\n"
            "- sentiment/evidence phải trích từ context; không bịa.\n"
        )

        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        )
        try:
            parsed = _parse_json_strict(response.content)

            # Ensure defaults if model misses optional fields.
            parsed.setdefault("forecast", "SIDEWAYS")
            parsed.setdefault("confidence", 50)
            parsed.setdefault("debate_summary", {})
            parsed.setdefault("news_evidence", [])
            parsed.setdefault("tech_evidence", {})
            parsed.setdefault("risk_conditions", [])

            # Keep UI-compatible debate_summary keys.
            debate_summary = parsed.get("debate_summary") or {}
            parsed["debate_summary"] = {
                "news_agent": debate_summary.get(
                    "news_agent",
                    parsed.get("debate_summary", {}).get("news_agent", ""),
                ),
                "tech_agent": debate_summary.get(
                    "tech_agent",
                    parsed.get("debate_summary", {}).get("tech_agent", ""),
                ),
                "final_decision": debate_summary.get("final_decision", ""),
            }
            parsed["symbol"] = symbol
            return parsed
        except Exception:
            # If the model returns non-JSON output, degrade gracefully.
            return {
                "symbol": symbol,
                "forecast": "SIDEWAYS",
                "confidence": 50,
                "debate_summary": {
                    "news_agent": state.get("news_agent_text", ""),
                    "tech_agent": state.get("tech_agent_text", ""),
                    "final_decision": "Risk evaluation failed to produce valid JSON.",
                },
                "reasoning": "Model output was not valid JSON at Risk Judge stage. Falling back to safe defaults.",
                "news_evidence": [],
                "tech_evidence": {},
                "risk_conditions": [],
            }

    # Build graph
    workflow = StateGraph(TAState)
    workflow.add_node("news_analyst", news_analyst_node)
    workflow.add_node("tech_analyst", tech_analyst_node)
    workflow.add_node("bull_researcher", bull_researcher_node)
    workflow.add_node("bear_researcher", bear_researcher_node)
    workflow.add_node("research_manager", research_manager_node)
    workflow.add_node("trader", trader_node)
    workflow.add_node("aggressive_debator", aggressive_debator_node)
    workflow.add_node("neutral_debator", neutral_debator_node)
    workflow.add_node("conservative_debator", conservative_debator_node)
    workflow.add_node("risk_judge", risk_judge_node)

    workflow.add_edge(START, "news_analyst")
    workflow.add_edge("news_analyst", "tech_analyst")
    workflow.add_edge("tech_analyst", "bull_researcher")
    workflow.add_edge("bull_researcher", "bear_researcher")
    workflow.add_edge("bear_researcher", "research_manager")
    workflow.add_edge("research_manager", "trader")
    workflow.add_edge("trader", "aggressive_debator")
    workflow.add_edge("aggressive_debator", "neutral_debator")
    workflow.add_edge("neutral_debator", "conservative_debator")
    workflow.add_edge("conservative_debator", "risk_judge")
    workflow.add_edge("risk_judge", END)

    return workflow.compile()
