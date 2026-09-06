import os
from dotenv import load_dotenv
import streamlit as st

# LangChain and Gemini imports
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import json


import requests
import re

load_dotenv()

# Environment variables required
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # for Gemini via langchain-google-genai
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

st.title("Course Finder — LangChain + Gemini + SerpAPI")
st.write("Enter a short query like: `course on llm under $50` and the app will use Gemini to decide whether to call SerpAPI, then show results.")

query = st.text_input("Search for courses", value="course on llm under $50")
if not query:
    st.stop()

if st.button("Find courses"):
    if not GOOGLE_API_KEY or not SERPAPI_API_KEY:
        st.error("Missing GOOGLE_API_KEY or SERPAPI_API_KEY in environment. Create a .env file with both keys.")
        st.stop()

    # Initialize Gemini model (langchain-google-genai wrapper)
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=GOOGLE_API_KEY)

    # Define a simple SerpAPI wrapper tool

    def search_tool(input_text: str) -> str:
        """Call SerpAPI and return a short formatted result."""
        params = {
            "q": input_text,
            "api_key": SERPAPI_API_KEY,
            "engine": "google",
        }
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        organic_results = data.get("organic_results", [])
        results = []
        for item in organic_results:
            title = item.get("title", "")
            link = item.get("link", "")
            if title and link:
                results.append(f"{title}: {link}")
        results = str(results)
        return results

    # Use structured prompting: ask Gemini to decide whether to call the tool and what query to pass.
    structured_prompt = f"You are an assistant that decides if a web search is needed and returns a JSON object with keys: call_tool (true/false), tool_query (string). User query: {query}"

    # Ask the LLM for a decision
    llm_output = llm.bind_tools(search_tool).invoke([HumanMessage(content=structured_prompt)])
    # Extract arguments from llm_output.additional_kwargs
    arguments_json = llm_output.additional_kwargs['function_call']['arguments']
    arguments = json.loads(arguments_json)

    # Call the search_tool with extracted arguments
    search_result = search_tool(arguments['input_text'])
    # Feed the search results to Gemini for formatting
    format_prompt = (
        f"Here are some raw course search results:\n{search_result}\n\n"
        "Please convert these into a beautiful, readable answer for the user, listing the courses with their titles and links."
    )
    formatted_output = llm.invoke([HumanMessage(content=format_prompt)])
    st.markdown(formatted_output.content)