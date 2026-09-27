from __future__ import annotations

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    function_tool,
)
from livekit.plugins import google, silero

from src.config import get_settings
from src.graph import RAGService
from src.vector_store import KnowledgeBase

settings = get_settings()
server = AgentServer()


class CompanyAssistant(Agent):
    def __init__(self, rag: RAGService) -> None:
        self.rag = rag
        super().__init__(
            instructions=(
                f"You are the voice assistant for {settings.company_name}. For every factual company "
                "question, call answer_from_company_knowledge and relay its answer faithfully. Never "
                "answer company questions from memory. If no source is found, say so and offer to "
                "connect the caller to a person. Keep spoken replies short and natural."
            )
        )

    @function_tool()
    async def answer_from_company_knowledge(self, context: RunContext, question: str) -> str:
        """Answer a factual question using the indexed company documents."""
        result = await self.rag.ask(question)
        return result["answer"]


@server.rtc_session(agent_name="company-rag-agent")
async def entrypoint(ctx: JobContext) -> None:
    knowledge_base = KnowledgeBase(settings)
    knowledge_base.ensure_seed_documents(settings.root_documents_path)
    rag = RAGService(graph=None, kb=knowledge_base, settings=settings)
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=google.beta.GeminiSTT(
            model=settings.stt_model,
            api_key=settings.google_api_key,
        ),
        llm=google.LLM(model=settings.chat_model, api_key=settings.google_api_key),
        tts=google.beta.GeminiTTS(
            model=settings.tts_model,
            voice_name=settings.tts_voice,
            api_key=settings.google_api_key,
            instructions="Speak clearly in a friendly, professional tone.",
        ),
    )
    await session.start(room=ctx.room, agent=CompanyAssistant(rag))
    await session.generate_reply(
        instructions=f"Greet the caller as the {settings.company_name} assistant and ask how you can help."
    )


if __name__ == "__main__":
    cli.run_app(server)
