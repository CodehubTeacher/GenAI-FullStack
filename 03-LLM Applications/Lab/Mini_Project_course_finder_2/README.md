Course Finder — LangChain + Gemini + SerpAPI

Simple Streamlit app that demonstrates using Gemini (via langchain-google-genai) with function/tool calling to query SerpAPI for course search results.

Files:

- `streamlit_app.py` — main Streamlit app
- `requirements.txt` — Python dependencies
- `.env.example` — environment variable example

Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create a `.env` file with your keys (see `.env.example`).

3. Run the app:

```bash
streamlit run streamlit_app.py
```

Notes and assumptions

- The example uses the `langchain-google-genai` integration and expects `GOOGLE_API_KEY` for Gemini.
- The app uses a simple decision prompt and permissive parsing. For production, validate and strictly parse JSON responses.
