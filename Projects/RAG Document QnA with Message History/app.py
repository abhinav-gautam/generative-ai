import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from dotenv import load_dotenv
import os

load_dotenv()


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


def initialize():
    if "llm" not in st.session_state:
        st.session_state.llm = ChatGroq(model="Gemma2-9b-It")
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        st.session_state.store = {}
        print("[INFO] Initialized the models")


def load_docs(files):
    # Load the documents if not already loaded
    if "splitted_docs" not in st.session_state:
        docs = []

        # Create a local copy of the loaded document and load it
        for file in files:
            tempPDF = f"./temp{file.name}"
            with open(tempPDF, "wb") as f:
                f.write(file.getvalue())

            loaded_docs = PyPDFLoader(tempPDF).load()
            os.remove(tempPDF)
            docs.extend(loaded_docs)

        print("[INFO] Loaded the docs")

        st.session_state.splitted_docs = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(docs)
        print("[INFO] Splitted the docs")


def create_retriever():
    # Create a retriever
    if "retriever" not in st.session_state:
        vector_store = FAISS.from_documents(
            st.session_state.splitted_docs, st.session_state.embeddings
        )
        st.session_state.retriever = vector_store.as_retriever()
        print("[INFO] Created the retriever")


def create_prompts():
    # Create prompts
    if (
        "context_prompt" not in st.session_state
        or "history_prompt" not in st.session_state
    ):
        context_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        st.session_state.context_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", context_system_prompt),
                MessagesPlaceholder("history"),
                ("human", "{input}"),
            ]
        )
        print("[INFO] Created the context prompt")

        history_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        st.session_state.history_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", history_system_prompt),
                MessagesPlaceholder("history"),
                ("human", "{input}"),
            ]
        )
        print("[INFO] Created the history prompt")


def create_chains():
    # Create a document retrieval chain
    if "retrieval_chain" not in st.session_state:
        # Create a retriever
        create_retriever()

        # Create prompts
        create_prompts()

        # Create a history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            st.session_state.llm,
            st.session_state.retriever,
            st.session_state.history_prompt,
        )
        print("[INFO] Created the history-aware retriever")

        # Create a document chain
        document_chain = create_stuff_documents_chain(
            st.session_state.llm, st.session_state.context_prompt
        )
        print("[INFO] Created the document chain")

        # Create a retrieval chain
        st.session_state.retrieval_chain = create_retrieval_chain(
            history_aware_retriever, document_chain
        )
        print("[INFO] Created the retrieval chain")


def create_history_rag_chain():
    # Create a history RAG chain
    if "history_rag_chain" not in st.session_state:
        # Create a retrieval chain
        create_chains()

        # Create a history RAG chain
        st.session_state.history_rag_chain = RunnableWithMessageHistory(
            st.session_state.retrieval_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="answer",
        )
        print("[INFO] Created the history RAG chain")


def reset():
    # Delete the stored documents
    if "splitted_docs" in st.session_state:
        del st.session_state.splitted_docs
        del st.session_state.history_rag_chain
        del st.session_state.retrieval_chain
        del st.session_state.context_prompt
        del st.session_state.history_prompt
        del st.session_state.retriever
        print("[INFO] Deleted the existing docs")


def main():
    # Initialize the models
    initialize()

    st.title("RAG Document QnA with Message History")

    # Document Uploader
    uploaded_files = st.file_uploader(
        "Upload documents to ask questions from",
        type=["pdf"],
        accept_multiple_files=True,
        on_change=reset,
    )

    if uploaded_files:
        st.write("Documents uploaded successfully!")

        # Load the documents
        load_docs(uploaded_files)

        # Create a history RAG chain
        create_history_rag_chain()

        # Get session id
        session_id = st.text_input("Session ID", value="default_session")

        # Get user input
        user_input = st.text_input("Your question:")

        if user_input:
            st.session_state.history_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )

        for i in range(len(get_session_history(session_id).messages) - 1, -1, -1):
            if get_session_history(session_id).messages[i].type == "ai":
                st.subheader("AI")
            if get_session_history(session_id).messages[i].type == "human":
                st.subheader("You")
            st.write(get_session_history(session_id).messages[i].content)
            st.divider()


if __name__ == "__main__":
    main()
