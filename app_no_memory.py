# from typing_extensions import TypedDict
from dotenv import load_dotenv

load_dotenv()
import streamlit as st

from langchain_openai import ChatOpenAI
from typing_extensions import Annotated
from langchain_community.utilities import SQLDatabase
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from typing_extensions import TypedDict
import os

from database import (
    create_database_from_csv,
    ddl,
    description
)

llm = ChatOpenAI(model="gpt-4o-mini")

if not os.path.exists('db.db'):
    create_database_from_csv('remont.csv', 'db.db')

db = SQLDatabase.from_uri("sqlite:///db.db")


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

class QueryOutput(BaseModel):
    """Generated SQL query."""

    query: str = Field(description = "Syntactically valid SQL query.")    



prompt = """Please write a sql query from the input. Transform the question from the user and provide an answer with the 
sql query only. These are the tables in the database: {tables} and the columns are to be found in the following 
{table_info}. The description of database is here - {description}
User question: {input}.
"""

query_prompt_template = PromptTemplate.from_template(prompt)


def write_query(state: State):
    """Generate SQL query to fetch information."""
      
    structured_llm = llm.with_structured_output(QueryOutput)
    
    chain = query_prompt_template | structured_llm
        
    result = chain.invoke({
        "input": state["question"],
        "tables": ', '.join(db.get_usable_table_names()),
        "table_info": db.table_info,
        "description": description
    })
    return {"query": result.query}

# print(write_query(state={"question": "How many activities are there?", "query": "", "result": "", "answer": ""}))

def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

print(
    execute_query(state={"question": "", "query": "SELECT Min(Date) FROM remont", "result": "", "answer": ""})
)

def generate_answer(state: State):
    """Answer question using retrieved information as context. """
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question. \n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}


# build the agent with Langgraph

from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()


# for step in graph.stream(
#     {"question": "Колко сме дали за обзавеждане?"}, stream_mode="updates"
# ):
#     print(step)



st.title("Ремонт в Гео Милев - търсачка")

# Initialize session state if not exists
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# def clear_text():
#     st.session_state.inputs = st.session_state.user_input
#     st.session_state.user_input = ""

user_question = st.text_input(label = "Попитай ме нещо:", key="user_input")


if st.button("Отговор"):
    if user_question:

        result = graph.invoke({"question": user_question})
                        
        # Add to chat history
        st.session_state.chat_history.insert(0, {
            "question": user_question,
            "answer": result["answer"],
            "query": result["query"]
        })

      
# Display chat history
for chat in st.session_state.chat_history:
    st.write("Question:", chat["question"])
    with st.expander("Show SQL Query"):
        st.code(chat["query"], language="sql")
    st.write("Answer:", chat["answer"])
    st.divider()


