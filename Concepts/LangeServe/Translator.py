import uvicorn
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes

# Create Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [("system", "Translate this into {language}"), ("human", "{text}")]
)

# Create LLM mode
llm = ChatGroq(model="Gemma2-9b-It")

# Create output parser
parser = StrOutputParser()

# Create Chain
chain = prompt | llm | parser

# Create FastAPI app
app = FastAPI(
    title="Language Translator",
    version="1.0.0",
    description="This is a simple language translator using LLM deployed using LangServe",
)

# Add route for the chain
add_routes(app, chain, path="/chain")

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=4000)
