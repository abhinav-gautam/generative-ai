import streamlit as st
import validators
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Initialize model
llm = ChatGroq(model="Gemma2-9b-it")

# Initialize prompt template
prompt_template = """Provide the summary of the following content in 300 words.
Content: {text}
"""
prompt = PromptTemplate.from_template(prompt_template)

# Set page title
st.set_page_config(page_title="Summarize youtube video or website")
st.title("Summarize Youtube video or any website")

# Take input url
url = st.text_input("Enter a Youtube video url or any page url to summarize")


if st.button("Summarize URL"):
    # Validate url
    if not validators.url(url):
        st.error("Please provide a valid URL.")

    else:
        with st.spinner("Summarizing..."):
            # Create loader based on url type
            if "youtube" in url:
                loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
            else:
                loader = UnstructuredURLLoader(
                    urls=[url],
                    ssl_verify=False,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                    },
                )

            # Load documents
            docs = loader.load()

            # Create chain and invoke chain
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
            summary = chain.invoke(docs)

            # Write the output
            st.success(summary["output_text"])
