# streamlit_app/utils.py
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pathlib import Path

# Prompt builders (now return PromptTemplate objects)
def build_recipe_prompt(ingredients: str, diet: str, time_limit: int) -> PromptTemplate:
    template = (
        "You are a helpful chef. Create a quick recipe using these ingredients: {ingredients}. "
        "Dietary restriction: {diet}. Total cook time should be under {time_limit} minutes. "
        "Include an ingredients list, step-by-step instructions, and a short tips section."
    )
    return PromptTemplate(template=template, input_variables=["ingredients", "diet", "time_limit"])


def build_study_prompt(topic: str, mode: str) -> PromptTemplate:
    if mode == "Summary":
        template = "You are a friendly study assistant. Produce a concise 150-word summary on the topic: {topic}."
        return PromptTemplate(template=template, input_variables=["topic"])
    if mode == "Quiz":
        template = "Create five multiple-choice questions (4 choices each) about {topic}. Mark the correct answer."
        return PromptTemplate(template=template, input_variables=["topic"])
    if mode == "Flashcards":
        template = "Produce 8 flashcards for {topic}. Each flashcard should have a question and brief answer. Only provide the Q&A pairs, nothing else, not even headers of flashcards."
        return PromptTemplate(template=template, input_variables=["topic"])
    template = "Provide helpful notes on {topic}."
    return PromptTemplate(template=template, input_variables=["topic"])


# Small helper to match imports from app.py (not currently used directly)
def build_history_prompt() -> PromptTemplate:
    template = "Answer the history question concisely and cite sources when available: {question}"
    return PromptTemplate(template=template, input_variables=["question"])

# RAG helper (simple illustration)
def run_rag_query(query: str, llm: ChatGoogleGenerativeAI) -> str:
    # NOTE: In a real app you should persist embeddings and documents. This is a minimal demo.
    # Assumes you have small docs list prepared in embeddings_data/ as plain text.
    docs = []
    try:
        p = Path(__file__).parent.parent / "embeddings_data"
        for f in p.glob("**/*.txt"):
            docs.append({"text": f.read_text(encoding="utf-8"), "metadata": {"source": str(f)}})
    except Exception:
        docs = []

    # If no docs found, fall back to a direct llm answer
    if not docs:
        prompt = f"Answer concisely (no hallucination): {query}"
        return llm.invoke(prompt)

    # Build embeddings + FAISS vector store
    emb = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    texts = [d["text"] for d in docs]
    metadatas = [d.get("metadata", {}) for d in docs]
    faiss_index = FAISS.from_texts(texts, emb, metadatas=metadatas)
    retriever = faiss_index.as_retriever(search_kwargs={"k": 3})

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa.invoke(query)["result"]
