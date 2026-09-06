# chainlit_app/chainlit_app.py
import os
from dotenv import load_dotenv
import chainlit as cl
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import summarize_text, classify_text, generate_names

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.2)

@cl.on_message
async def main(message: cl.Message):
    """Simple router that recognizes user intent based on quick heuristics.
    The UI in Chainlit will be simple: user types, assistant asks which tool to use or auto-routes.
    """
    text = message.content
    print(text)
    # Very small heuristic router — in production use a proper intent classifier.
    lower = text.lower()
    if lower.startswith("summarize:"):
        content = text.split(":", 1)[1].strip()
        out = summarize_text(content, llm)
        await cl.Message(content=out).send()
        return
    if lower.startswith("classify:"):
        content = text.split(":", 1)[1].strip()
        out = classify_text(content, llm)
        await cl.Message(content=out).send()
        return
    if lower.startswith("names:"):
        brief = text.split(":", 1)[1].strip()
        out = generate_names(brief, llm)
        await cl.Message(content=out).send()
        return

    await cl.Message(content="I didn't understand. Prefix with `summarize:`, `classify:`, or `names:`").send()
