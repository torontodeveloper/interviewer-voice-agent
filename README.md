# AI Interview Coach — Voice Agent

A real-time AI interviewer that conducts mock technical interviews over phone, powered by a RAG pipeline built on the job description. Call in, get interviewed by Angelina.

## What It Does

- Calls your phone via Twilio and conducts a structured mock interview
- Ingests a Job Description PDF into Pinecone vector DB
- Retrieves JD context at startup via cosine similarity search
- Asks technical and behavioral questions tailored to the role
- Probes with follow-up questions after each answer
- Gives honest feedback at the end of the session

## Architecture

```
Phone call → Twilio → FastAPI WebSocket
                           ↓
                     Deepgram STT
                           ↓
                    LLM User Aggregator
                           ↓
              OpenAI GPT-4.1 (JD context in system prompt)
                           ↓
                     Cartesia TTS
                           ↓
              Twilio → Phone call
```

**RAG flow (startup only):**
```
ManulifeJD.pdf → NLTK chunking → Pinecone upsert (llama-text-embed-v2)
                                        ↓
             search_query("Senior GenAI Engineer LLM RAG Azure requirements")
                                        ↓
                          top-5 chunks → system prompt
```

## Stack

| Component | Service |
|---|---|
| Voice transport | Twilio + FastAPI WebSocket |
| Speech-to-text | Deepgram |
| Text-to-speech | Cartesia |
| LLM | OpenAI GPT-4.1 |
| Vector DB | Pinecone (llama-text-embed-v2 embeddings) |
| VAD | Silero |
| Orchestration | Pipecat |

## Setup

**1. Install dependencies**
```bash
uv sync
```

**2. Create `.env`**
```
OPENAI_API_KEY=
DEEPGRAM_API_KEY=
CARTESIA_API_KEY=
CARTESIA_VOICE_ID=
PINECONE_API_KEY=
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
```

**3. Ingest the JD into Pinecone (run once)**
```bash
uv run python rag_processor.py
```

**4. Start the bot**
```bash
uv run python bot.py -t twilio 2>&1 | tee /tmp/bot.log
```

**5. Configure Twilio**

Set your Twilio webhook URL to:
```
wss://<your-domain>/ws
```

## Interview Format

Angelina conducts the interview in this order:

**Technical questions:**
1. Walk me through a RAG pipeline you built end to end
2. How do you implement guardrails in a production LLM system?
3. How did you deploy your GenAI solution on Azure?
4. How do you evaluate LLM performance?

**Behavioral questions:**
1. Tell me about a time you worked with non-technical stakeholders
2. Describe a time you made a confident technical decision with incomplete information

## Key Design Decisions

**JD loaded once at startup, not per turn** — the RAG search runs once when the bot initializes and injects the JD context into the system prompt. This removes per-turn Pinecone latency and keeps the conversation smooth.

**Namespace isolation** — JD chunks are stored in the `"interviewer"` namespace, separate from any other Pinecone data.

**VAD threshold 0.7** — higher than default (0.5) to reduce false triggers from background noise during phone calls.

## Deployment

Deploy to Azure App Service for production quality audio (removes ngrok latency):

```bash
az webapp up --name <app-name> --resource-group <rg> --runtime "PYTHON:3.12"
```

Set all `.env` variables as Azure App Settings and configure startup command:
```
python bot.py -t twilio --host 0.0.0.0 --port 8000
```
