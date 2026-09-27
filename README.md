# Company RAG Voice Agent

A Python voice agent that answers from uploaded company documents. LiveKit handles the realtime audio session, Gemini provides speech and intelligence, LangGraph runs retrieval and grounded answer generation, and Chroma stores Gemini embeddings locally.

## Architecture

`caller -> LiveKit (STT) -> agent -> LangGraph -> Chroma -> grounded answer -> LiveKit (TTS)`

The agent is instructed to use the knowledge-base tool for every factual company question. The graph always retrieves first, then generates from retrieved context only.

## Setup

Requires Python 3.11-3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

Add the LiveKit and Google Gemini credentials to `.env`. This project also reads `api.txt` temporarily for compatibility, but `.env` with `KEY=value` lines is recommended. Both files are ignored by Git.

A `GOOGLE_API_KEY` from Google AI Studio is required for Gemini embeddings, answer generation, speech recognition, and speech synthesis. LiveKit credentials connect the worker to your realtime room; they do not replace the Gemini credential.

The defaults use `gemini-3-flash-preview` for chat, `gemini-embedding-001` for retrieval, `gemini-3.5-transcribe-live` for speech recognition, and `gemini-3.8-flash-tts` with the `Zephyr` voice. Each can be changed in `.env`.

## Add documents

Supported formats are PDF, DOCX, Markdown, and plain text.

```bash
python -m src.ingest /path/to/company-documents
```

Or start the API and upload documents:

```bash
uvicorn src.api:app --reload --port 8000
curl -F 'file=@handbook.pdf' http://localhost:8000/documents
curl -H 'Content-Type: application/json' -d '{"question":"What is our return policy?"}' http://localhost:8000/ask
```

## Run the LiveKit agent

Validate the worker locally:

```bash
python -m src.agent download-files
python -m src.agent console
```

Run it for LiveKit Cloud jobs:

```bash
python -m src.agent start
```

The registered agent name is `company-rag-agent`. Dispatch that name from your LiveKit frontend or configure automatic dispatch in LiveKit Cloud.

## Deploy to LiveKit Cloud

The container bundles files from `documents/` and indexes them with Gemini into Chroma when a new deployment starts with an empty database. Secrets are never copied into the image.

```bash
brew install livekit-cli
lk cloud auth
lk project set-default "YOUR_PROJECT_NAME"
lk agent create --secrets GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
```

After the first deployment, publish updates with:

```bash
lk agent deploy
lk agent status
lk agent logs
```

## Security and production notes

- Rotate credentials that have been pasted into chat, logs, or source files.
- Protect `/documents` with authentication before exposing it publicly.
- Add tenant metadata filters if documents belong to multiple customers or departments.
- Replace local Chroma with a managed vector database when running multiple worker replicas.
- Scan uploads and enforce a maximum size at the reverse proxy/API layer.
# voice_agent
