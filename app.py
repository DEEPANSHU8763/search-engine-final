import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchResults,
)
from langgraph.prebuilt import create_react_agent

# -------------------------
# Tools
# -------------------------

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchResults(
    name="Web Search",
    description="Search the internet for recent information",
)

tools = [search, arxiv, wiki]

# -------------------------
# Streamlit UI
# -------------------------

st.title("LangChain Search Assistant")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I can search the web, Wikipedia, and Arxiv. Ask me anything.",
        }
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -------------------------
# User Input
# -------------------------

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
    )

    # Create LangGraph Agent
    agent = create_react_agent(llm, tools)

    with st.chat_message("assistant"):

        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )

        # Last message from agent
        output = result["messages"][-1].content

        st.session_state.messages.append(
            {"role": "assistant", "content": output}
        )

        st.write(output)
