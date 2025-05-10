import speech_recognition as sr
import pocketsphinx as ps
from langgraph.checkpoint.mongodb import MongoDBSaver
from .graph import create_chat_graph
from dotenv import load_dotenv
import asyncio
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer
load_dotenv()

openai=AsyncOpenAI()

MONGODB_URI="mongodb://localhost:27017/"
config={"configurable": {"thread_id":"1"}}

def main():
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph=create_chat_graph(checkpointer=checkpointer)

        r=sr.Recognizer()

        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source) #for noise cancellation
            r.pause_threshold=4
            while True:
                print("Say something....")
                audio=r.listen(source)
                print("Processing audio .....")
                sst= r.recognize_google(audio)
                print("You said: ",sst)

                for event in graph.stream({"messages":[{"role":"user", "content":sst}]},config,stream_mode="values"):
                    if "messages" in event:
                        event["messages"][-1].pretty_print()

async def speak(text:str):
    async with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o.mini-tts",
        voice="coral",
        input=text,
        instructions="Speak in a cheerful and positive tone.",
        response_format="pcm",
    ) as response:
        await LocalAudioPlayer().play(response)

# main()

if __name__ == "__main__":
   asyncio.run(speak(text="this is a smaple"))