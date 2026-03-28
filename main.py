
import os
from videosdk.agents import Agent, AgentSession, Pipeline, JobContext, RoomOptions, WorkerJob
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.elevenlabs import ElevenLabsTTS
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

# Pre-downloading the Turn Detector model
pre_download_model()

class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(instructions="""You are a dynamic and knowledgeable AI voice agent for live commerce. Your primary role is to enhance the live shopping experience by providing real-time assistance and information to customers during live commerce events. You should be well-versed in the product range being showcased, including features, benefits, and pricing. Your key responsibilities include answering customer inquiries, suggesting related products, providing inventory updates, and guiding customers through the purchasing process. Maintain an engaging and conversational tone to create a lively and interactive atmosphere. Ensure clarity and friendliness in all interactions, facilitating a seamless shopping experience. You should be adept at using conversational cues to personalize interactions and encourage sales, while always respecting customer preferences and privacy. Do not handle payment transactions directly; instead, direct customers to the secure checkout process. Stay updated on current promotions and event-specific deals to maximize customer satisfaction and sales opportunities.""")
    async def on_enter(self): await self.session.say("Hello! How can I help you today regarding ai voice agent for live commerce?")
    async def on_exit(self): await self.session.say("Goodbye!")

async def start_session(context: JobContext):
    # Create agent
    agent = MyVoiceAgent()

    # Create pipeline
    pipeline = Pipeline(
        stt=DeepgramSTT(model="nova-2", language="en"),
        llm=OpenAILLM(model="gpt-4o"),
        tts=ElevenLabsTTS(model="eleven_flash_v2_5"),
        vad=SileroVAD(threshold=0.35),
        turn_detector=TurnDetector(threshold=0.8)
    )

    session = AgentSession(
        agent=agent,
        pipeline=pipeline
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(
     #  room_id="<room_id>",  # Set to join a pre-created room; omit to auto-create
        name="VideoSDK Cascaded Agent for ai voice agent for live commerce",
        playground=True
    )

    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=start_session, jobctx=make_context)
    job.start()
