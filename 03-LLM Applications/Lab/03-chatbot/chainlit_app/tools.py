# chainlit_app/tools.py
from langchain_google_genai import ChatGoogleGenerativeAI

def summarize_text(text: str, llm: ChatGoogleGenerativeAI) -> str:
    prompt = f"Summarize the following text in 3 lines and list 3 key points:\n\n{text}"
    return llm.invoke(prompt).content

def classify_text(text: str, llm: ChatGoogleGenerativeAI) -> str:
    prompt = (
        "You are a classifier. Given the text, return ONLY a JSON object with keys: category, confidence.\n"
        f"Text: {text}\n\nRespond in JSON. Categories: [Support, Sales, Feedback, Other]"
    )
    return llm.invoke(prompt).content

def generate_names(brief: str, llm: ChatGoogleGenerativeAI) -> str:
    prompt = (
        f"You are a creative naming assistant. Given this brief: {brief}, produce 12 product name suggestions. "
        "For each name add a one-line rationale."
    )
    return llm.invoke(prompt).content
