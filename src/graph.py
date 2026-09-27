from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from src.config import Settings, get_settings
from src.vector_store import KnowledgeBase


class RAGState(TypedDict, total=False):
    question: str
    matches: list[dict[str, Any]]
    answer: str
    sources: list[dict[str, Any]]


def build_rag_graph(kb: KnowledgeBase | None = None, settings: Settings | None = None):
    settings = settings or get_settings()
    kb = kb or KnowledgeBase(settings)
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for Gemini answer generation")
    model = ChatGoogleGenerativeAI(
        model=settings.chat_model,
        temperature=0,
        google_api_key=settings.google_api_key,
    )

    def retrieve(state: RAGState) -> RAGState:
        return {"matches": kb.search(state["question"])}

    async def answer(state: RAGState) -> RAGState:
        matches = state.get("matches", [])
        if not matches:
            return {
                "answer": "I couldn't find that information in the company knowledge base.",
                "sources": [],
            }
        context = "\n\n".join(
            f"[Source {i}: {m['metadata'].get('source', 'unknown')}, "
            f"page {m['metadata'].get('page', 'n/a')}]\n{m['text']}"
            for i, m in enumerate(matches, start=1)
        )
        response = await model.ainvoke(
            [
                SystemMessage(
                    content=(
                        f"You answer questions about {settings.company_name}. Use only the supplied "
                        "context. If it does not contain the answer, say you do not know. Do not invent "
                        "facts. Keep the response concise and natural for speech. Do not read citations "
                        "aloud unless asked."
                    )
                ),
                HumanMessage(content=f"Question: {state['question']}\n\nContext:\n{context}"),
            ]
        )
        sources = [
            {
                "source": match["metadata"].get("source", "unknown"),
                "page": match["metadata"].get("page"),
                "relevance": round(match["relevance"], 3),
            }
            for match in matches
        ]
        return {"answer": str(response.content), "sources": sources}

    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("answer", answer)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("answer", END)
    return workflow.compile()


class RAGService:
    def __init__(
        self,
        graph=None,
        kb: KnowledgeBase | None = None,
        settings: Settings | None = None,
    ):
        self.graph = graph or build_rag_graph(kb=kb, settings=settings)

    async def ask(self, question: str) -> RAGState:
        return await self.graph.ainvoke({"question": question})
