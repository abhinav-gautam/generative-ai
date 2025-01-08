import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv

# Import environment variables
load_dotenv()


def create_chains():
    # Create new chains if they don't exist
    if "retrieval_chain" not in st.session_state:
        # Create document chain
        document_chain = create_stuff_documents_chain(
            st.session_state.llm, st.session_state.prompt
        )

        # Create retrieval chain
        st.session_state.retrieval_chain = create_retrieval_chain(
            st.session_state.retriever, document_chain
        )


def create_retriever():
    # Create new retriever if it doesn't exist
    if "retriever" not in st.session_state:
        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        # Split documents
        splitted_docs = splitter.split_documents(st.session_state.docs)

        # Create vector store
        vector_store = FAISS.from_documents(splitted_docs, st.session_state.embeddings)

        # Get retriever
        st.session_state.retriever = vector_store.as_retriever()


def initialize():
    if "docs" not in st.session_state:
        # Initialize llm model and embeddings
        st.session_state.llm = ChatGroq(model="Llama3-8b-8192")
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Create prompt template
        st.session_state.prompt = ChatPromptTemplate.from_template(
            """
                Answer the following questions based on the provided context only:
                Please provide the most accurate response based on the question and context.
                <context>
                {context}
                <context>                                              
                Question: {input}
            """
        )

        # Load documents
        st.session_state.docs = PyPDFDirectoryLoader("source/").load()


def main():
    # Initialize models and load data
    initialize()

    # Create retriever
    create_retriever()

    # Create chains
    create_chains()

    st.title("RAG Document QnA")

    # Get user query
    user_query = st.text_input("Enter your question:")

    if st.button("Submit"):

        # Get response
        response = st.session_state.retrieval_chain.invoke({"input": user_query})
        st.write(response["answer"])


if __name__ == "__main__":
    main()
