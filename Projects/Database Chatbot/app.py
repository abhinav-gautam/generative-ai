from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from dotenv import load_dotenv
from pathlib import Path
import streamlit as st
import sqlite3

load_dotenv()

st.set_page_config(page_title="Chat with SQL Database")
st.title("Chat with SQL Database")

LOCAL_DB = "USE_LOCALDB"
MYSQL_DB = "USE_MYSQL"

st.sidebar.title("Settings")
selection_options = [
    "Use SQLLite 3 Database - Student.db",
    "Connect to your MeSQL Database",
]
selected_db = st.sidebar.radio(
    label="Choose the database with which you want to chat",
    options=selection_options,
)

if selection_options.index(selected_db) == 1:
    db_uri = MYSQL_DB
    mysql_host = st.sidebar.text_input("MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database")
else:
    db_uri = LOCAL_DB

if not db_uri:
    st.info("Please enter database information and URI.")

# Initialize model
llm = ChatGroq(model="Gemma2-9b-it", streaming=True)


@st.cache_resource(ttl="2h")
def configure_db(
    db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None
):
    if db_uri == LOCAL_DB:
        dbfilepath = (Path(__file__).parent / "student.db").absolute()
        print(dbfilepath)
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri == MYSQL_DB:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        return SQLDatabase(
            create_engine(
                f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
            )
        )


if db_uri == MYSQL_DB:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_uri)

# Create toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Create agent
agent = create_sql_agent(
    llm=llm, toolkit=toolkit, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[callback])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
