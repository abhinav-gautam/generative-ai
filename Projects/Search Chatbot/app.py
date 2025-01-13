import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
)
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# Initialize tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

duckduckgo_tool = DuckDuckGoSearchRun(name="DuckDuckGo")

# Initialize model
llm = ChatGroq(model="Gemma2-9b-it", streaming=True)

# Initialize agent
tools = [arxiv_tool, wikipedia_tool, duckduckgo_tool]
agent = initialize_agent(
    tools, llm, AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True
)

st.title("Search Chatbot")
"""A chatbot that searches Wikipedia, Arxiv and DuckDuckGo."""

st.sidebar.title("Settings")
show_thinking = st.sidebar.checkbox("Show thinking")

# Initialize messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am a search chatbot. How can I help you today?",
        }
    ]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input(placeholder="Type a message...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        if show_thinking:
            streamlit_callback = StreamlitCallbackHandler(
                st.container(), expand_new_thoughts=False
            )
            response = agent.run(
                st.session_state.messages, callbacks=[streamlit_callback]
            )
        else:
            response = agent.run(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
