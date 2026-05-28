import asyncio
import logging
import os
import threading

from dotenv import load_dotenv
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pinecone import Pinecone
from pipecat.frames.frames import Frame
from pipecat.processors.frameworks.rtvi import RTVIObserverParams
from pipecat.runner.run import main
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.frames.frames import LLMContextFrame
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
)
from pipecat.audio.vad.vad_analyzer import VADParams

from pipecat_whisker import WhiskerObserver
from rag_processor import RAGDataBase

load_dotenv()

rag_db = None


def _init_rag():
    global rag_db
    db = RAGDataBase()
    if not db.is_ingested("interviewer"):
        db.upsert(file="ManulifeJD.pdf")
    rag_db = db
    logging.info("RAG initialized")


threading.Thread(target=_init_rag, daemon=True).start()


class RAGProcessor(FrameProcessor):
    """MetricsFrameLogger formats and logs all MetericsFrames"""

    def __init__(self):
        super().__init__()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            message = frame.context.messages[-1].get("content", "")
            result = await rag_db.search_query(message)
            if result.strip():
                frame.context.messages.append({"role": "user", "content": result})
            await self.push_frame(frame, direction)
        # ALWAYS push all frames
        else:
            # SUPER IMPORTANT: always push every frame!
            await self.push_frame(frame, direction)


transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}

prompt = """You are Angelina, a senior technical interviewer conducting a mock interview
                    for the Manulife Senior GenAI Engineer role. You have the confident, direct,
                    and sophisticated style of Angelina Jolie — warm but intense, never wastes words.

                    You are interviewing Kevin for a role building an LLM assistant for insurance
                    sales advisors.
                    
                    Start by greeting Kevin with calm confidence and explain you will ask a mix
                    of technical and behavioral questions.

                    Ask one question at a time. Wait for a full answer. Ask one sharp follow-up
                    before moving to the next question.

                    Technical questions:
                    1. Walk me through a RAG pipeline you built end to end.
                    2. How do you implement guardrails in a production LLM system?
                    3. How did you deploy your GenAI solution on Azure?
                    4. How do you evaluate LLM performance?

                    Behavioral questions:
                    1. Tell me about a time you worked with non-technical stakeholders on an AI project.
                    2. Describe a time you made a confident technical decision with incomplete information.
                    
                    Rules:
                    - Voice call format. Keep questions short and conversational.
                    - No bullet points or markdown.
                    - Probe with one sharp follow-up after each answer.
                    - Be direct — if an answer is vague, call it out gracefully.
                    - Give Kevin honest feedback at the end."""


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):

    while rag_db is None:
        await asyncio.sleep(0.5)

    jd_context = await rag_db.search_query(
        "Senior GenAI Engineer LLM RAG Azure requirements"
    )

    prompt_with_context = prompt + f"\n\nJob Description context:\n{jd_context}"
    # Create AI Services
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            voice=os.getenv(
                "CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"
            ),
        ),
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
            system_instruction=prompt_with_context,
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    threshold=0.7,  # default is 0.5 — higher = less sensitive
                )
            ),
        ),
    )
    rag_processor = RAGProcessor()

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
        ]
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Kick off the conversation.
        context.add_message(
            {"role": "user", "content": "Please introduce yourself to the user."}
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        print("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        rtvi_observer_params=RTVIObserverParams(
            bot_llm_enabled=False,
            metrics_enabled=False,
        ),
    )
    task.add_observer(WhiskerObserver(task.pipeline))
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point called by the development runner."""

    transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
    logging.info(f"Auto-detected transport: {transport_type}")

    serializer = TwilioFrameSerializer(
        stream_sid=call_data["stream_id"],
        call_sid=call_data["call_id"],
        account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
    )

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    threshold=0.7,  # default is 0.5 — higher = less sensitive
                )
            ),
            serializer=serializer,
        ),
    )

    # Run your bot logic
    await run_bot(transport, transport_params)


if __name__ == "__main__":
    main()
