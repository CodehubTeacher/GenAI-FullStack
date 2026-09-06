# streamlit_app/app.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from utils import build_recipe_prompt, build_study_prompt, build_history_prompt, run_rag_query

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="LLM Level2 — Streamlit Toy UI", layout="wide")
st.title("Level-2 LLM Apps — Streamlit Toy UI")

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.2)

app_tab = st.tabs(["Recipe Generator", "Study Buddy", "History Bot"])

# --- Recipe Generator ---
with app_tab[0]:
    st.header("Recipe Generator")
    col1, col2 = st.columns([0.2, 0.8])

    with col1:
        ingredients = st.text_input("Ingredients (comma separated)")
        prefs = st.selectbox("Diet", ["Any", "Vegetarian", "Vegan", "Gluten-free"]) 
        time_limit = st.slider("Max cook time (minutes)", 5, 120, 30)
        generate_recipe = st.button("Generate Recipe")
        with st.expander("Prompt"):
            prompt_t = build_recipe_prompt(ingredients, prefs, time_limit)
            inp = prompt_t.format(ingredients=ingredients, diet=prefs, time_limit=time_limit)  # Example formatting
            st.write(inp)
    with col2:
        if generate_recipe:
            if not ingredients.strip():
                st.warning("Please enter at least one ingredient.")
            else:
                prompt_t = build_recipe_prompt(ingredients, prefs, time_limit)
                recipe_chain = prompt_t | llm
                with st.spinner("Generating recipe..."):
                    resp = recipe_chain.invoke({"ingredients": ingredients, "diet": prefs, "time_limit": time_limit})
                    with st.container(border = True):
                        st.markdown(resp.content)

# --- Study Buddy ---
with app_tab[1]:
    st.header("Study Buddy")
    col1, col2 = st.columns([0.2, 0.8])

    with col1:
        topic = st.text_input("Topic")
        mode = st.selectbox("Mode", ["Summary", "Quiz", "Flashcards"])
        create_study_material = st.button("Create Study Material")
        with st.expander("Prompt"):
            prompt_t = build_study_prompt(topic, mode)
            inp = prompt_t.format(topic=topic, mode=mode)
            st.write(inp)
    with col2:
        if create_study_material:
            if not topic.strip():
                st.warning("Please enter a topic.")
            else:
                prompt_t = build_study_prompt(topic, mode)
                chain = prompt_t | llm
                with st.spinner("Generating study material..."):    
                    resp = chain.invoke({"topic": topic, "mode": mode})
                    if mode == "Flashcards":
                        # Assume resp.content is a string with flashcards separated by double newlines
                        flashcards = [fc.strip() for fc in resp.content.split("\n\n") if fc.strip()]
                        if flashcards:
                            st.subheader("Flashcards")
                            for i in range(0, len(flashcards), 4):
                                cols = st.columns(4)
                                for j, card in enumerate(flashcards[i:i+4]):
                                    with cols[j]:
                                        with st.container(border=True):
                                            st.markdown(card)
                        else:
                            st.info("No flashcards generated.")
                    else:
                        with st.container(border=True):
                            st.markdown(resp.content)

# --- History Bot (RAG-enabled) ---
with app_tab[2]:
    st.header("History Bot (RAG)")
    col1, col2 = st.columns([0.2, 0.8])

    with col1:
        query = st.text_input("Ask a history question")
        ask = st.button("Ask")
        with st.expander("Prompt"):
            prompt_t = build_history_prompt()
            try:
                inp = prompt_t.format(question=query)
            except Exception:
                inp = str(prompt_t)
            st.write(inp)

    with col2:
        if ask:
            if not query.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Retrieving and generating answer..."):
                    answer = run_rag_query(query, llm)
                    with st.container(border=True):
                        st.markdown(answer)
