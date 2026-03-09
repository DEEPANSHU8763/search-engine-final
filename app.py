import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchResults,
)

# -------------------
# Streamlit UI
# -------------------

st.title("🔎 AI Research Assistant")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if not api_key:
    st.info("Please enter your Groq API key in the sidebar.")
    st.stop()

# -------------------
# LLM
# -------------------

llm = ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile",
)

# -------------------
# Tools
# -------------------

search = DuckDuckGoSearchResults()

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# -------------------
# Chat memory
# -------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I can search the web, Wikipedia, and Arxiv. Ask me anything.",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -------------------
# User Input
# -------------------

if prompt := st.chat_input("Ask something..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Searching..."):

            try:
                # Search the web
                web_result = search.run(prompt)

                # Wikipedia
                wiki_result = wiki.run(prompt)

                # Arxiv
                arxiv_result = arxiv.run(prompt)

                context = f"""
                Web Search Result:
                {web_result}

                Wikipedia Result:
                {wiki_result}

                Arxiv Result:
                {arxiv_result}

                User Question:
                {prompt}

                Using the information above, give a helpful answer.
                """

                response = llm.invoke(context)
                output = response.content

            except Exception as e:
                st.error(e)
                st.stop()

        st.session_state.messages.append(
            {"role": "assistant", "content": output}
        )

        st.write(output)
