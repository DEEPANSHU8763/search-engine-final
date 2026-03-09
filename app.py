import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchResults,
)
from langgraph.prebuilt import create_tool_calling_agent

# -----------------------
# Streamlit UI
# -----------------------

st.title("🔎 AI Research Assistant")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if not api_key:
    st.info("Please enter your Groq API key in the sidebar.")
    st.stop()

# -----------------------
# Tools
# -----------------------

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchResults()

tools = [search, arxiv, wiki]

# -----------------------
# Chat Memory
# -----------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I can search the web, Wikipedia, and Arxiv. Ask me anything.",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -----------------------
# User Input
# -----------------------

if prompt := st.chat_input("Ask something..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
    )

    # Tool-calling agent
    agent = create_tool_calling_agent(llm, tools)

    with st.chat_message("assistant"):

        with st.spinner("Searching..."):

            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]}
                )

                output = result["messages"][-1].content

            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        st.session_state.messages.append(
            {"role": "assistant", "content": output}
        )

        st.write(output)
