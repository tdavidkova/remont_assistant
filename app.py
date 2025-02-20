# from typing_extensions import TypedDict
from dotenv import load_dotenv

load_dotenv()
import streamlit as st

from langchain_openai import ChatOpenAI
from typing_extensions import Annotated
from langchain_community.utilities import SQLDatabase
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from typing_extensions import TypedDict, List
import os
from operator import add
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.graph import MermaidDrawMethod

checkpointer = MemorySaver()


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
    messages: List[BaseMessage]
    question: HumanMessage
    on_topic: str
    rephrased_question: str
    query: str
    result: str
    


def question_rewriter(state: State):
    print(f"Entering question_rewriter with following state: {state}\n")

    # Reset state variables except for 'question' and 'messages'
    state["on_topic"] = ""
    state["rephrased_question"] = ""
    state["query"] = ""
    state["result"] = ""


    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    if state["question"] not in state["messages"]:
        state["messages"].append(state["question"])

    if len(state["messages"]) > 1:
        conversation = state["messages"][:-1]
        current_question = state["question"].content
        messages = [
            SystemMessage(
                content="You are a helpful assistant that rephrases the user's question to be a standalone question optimized for transformation to a sql query."
            )
        ]
        messages.extend(conversation)
        messages.append(HumanMessage(content=current_question))
        rephrase_prompt = ChatPromptTemplate.from_messages(messages)
        
        prompt = rephrase_prompt.format()
        response = llm.invoke(prompt)
        better_question = response.content.strip()
        print(f"question_rewriter: Rephrased question: {better_question}\n")
        state["rephrased_question"] = better_question
    else:
        state["rephrased_question"] = state["question"].content
    return state

# Test the question_rewriter function
# test_state = {
#     "question": HumanMessage(content="Какво е врeмвто в София?"),
#     "on_topic": "",
#     "rephrased_question": "",
#     "query": "",
#     "result": "",
#     "messages": []
# }

# Invoke the function and print the output
# output_state = question_rewriter(test_state)
# print("Output State after question_rewriter:", output_state)

class GradeQuestion(BaseModel):
    score: str = Field(
        description="Question is about the specified topics? If yes -> 'Yes' if not -> 'No'"
    )

def question_classifier(state: State):
    """Classify question as on-topic or off-topic"""
    print("Entering question_classifier...")
    system_message = SystemMessage(
        content="""You are a classifier that determines whether a user's question is about one of the following topics:

1. Costs associated with reconstruction of an apartment, buying furniture, etc.
2. Questions regarding payments (quentities, payers, persons who received the payments, reasons for payments, etc.
4. Questions regarding time - when, how long, etc. related to reconstruction works

If the question IS about any of these topics, respond with 'Yes'. Otherwise, respond with 'No'."""
    )

    human_message = HumanMessage(
        content=f"User question: {state['rephrased_question']}"
    )
    grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    
    structured_llm = llm.with_structured_output(GradeQuestion)
    grader_llm = grade_prompt | structured_llm
    result = grader_llm.invoke({})
    state["on_topic"] = result.score.strip()
    print(f"question_classifier: on_topic = {state['on_topic']}\n")
    return state

# output_state2 = question_classifier(output_state)
# print("Output State after question_classifier:", output_state2)


class QueryOutput(BaseModel):
    """Generated SQL query."""

    query: str = Field(description = "Syntactically valid SQL query.")    


def write_query(state: State):
    """Generate SQL query to fetch information."""
    print("Entering write_query...")
    prompt = """Please write a sql query from the input. Transform the question from the user and provide an answer with the 
sql query only. These are the tables in the database: {tables} and the columns are to be found in the following 
{table_info}. The description of database is here - {description}
User question: {input}.
"""
    query_prompt_template = PromptTemplate.from_template(prompt)
      
    structured_llm = llm.with_structured_output(QueryOutput)
    
    chain = query_prompt_template | structured_llm
        
    result = chain.invoke({
        "input": state["rephrased_question"],
        "tables": ', '.join(db.get_usable_table_names()),
        "table_info": db.table_info,
        "description": description
    })

    state["query"] = result.query
    print(f"query: {state['query']}\n")
    return state
    # return state

# output_state3 = write_query(output_state2)
# print("Output State after write_query:", output_state3)


def execute_query(state: State):
    """Execute SQL query."""
    print("Entering execute_query...")
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    state["result"] = execute_query_tool.invoke(state["query"])
    print(f"result: {state['result']}\n")
    return state

# output_state4 = execute_query(output_state3)
# print("Output State after execute_query:", output_state3)



def generate_answer(state: State):
    """Answer question using the query results"""
    print("Entering generate_answer...")
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question. "
        "If on-topic is 'no' respond in the language in which the question was asked "
        "that you can't answer this question."
        "\n\n"
        f'Question: {state["rephrased_question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
        f'On-topic: {state["on_topic"]}'
    )
    response = llm.invoke(prompt)
    state["messages"].append(AIMessage(content=response.content))
    print(f"AI response: {response.content}\n")
    print(f"Final state: {state}\n")
    return state

# output_state32 = generate_answer(output_state2)
# print("Output State after generate_answer:", output_state32)

# output_state5 = generate_answer(output_state4)
# print("Output State after generate_query:", output_state5)
# build the agent with Langgraph

def on_topic_router(state: State):
    print("Entering on_topic_router...")
    on_topic = state.get("on_topic", "").strip().lower()
    if on_topic == "yes":
        print("Routing to write_query\n")
        return "yes"
    else:
        print("Routing to off_topic_response\n")
        return "no"

from langgraph.graph import START, END, StateGraph

workflow = StateGraph(State)
workflow.add_node("question_rewriter", question_rewriter)
workflow.add_node("question_classifier", question_classifier)
workflow.add_node("write_query", write_query)
workflow.add_node("execute_query", execute_query)
workflow.add_node("generate_answer", generate_answer)


workflow.set_entry_point("question_rewriter")
workflow.add_edge("question_rewriter", "question_classifier")
workflow.add_conditional_edges(
    "question_classifier",
    on_topic_router,
    {
        "yes": "write_query",
        "no": "generate_answer",
    },
)
workflow.add_edge("write_query", "execute_query")
workflow.add_edge("execute_query", "generate_answer")
workflow.add_edge("generate_answer", END)
graph = workflow.compile(checkpointer=checkpointer)


# for step in graph.stream(
#     {"question": HumanMessage(content="Колко сме дали за обзавеждане?")}, stream_mode="updates",
#     config={"configurable": {"thread_id": 1}}
# ):
#     print(step)

# from IPython.display import Image, display


graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

# result = graph.invoke(input = {"question":HumanMessage(content= "Кога е започнал ремонтът?")},
#              config={"configurable": {"thread_id": 1}})

# graph.invoke(input = {"question":HumanMessage(content= "А кога е зазвършил?")},
#              config={"configurable": {"thread_id": 1}})

#### First (working version of a chatbot)

# st.title("Ремонт в Гео Милев - търсачка")


# # Initialize session state if not exists
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []

# if 'graph' not in st.session_state:
#     st.session_state.graph= workflow.compile(checkpointer=checkpointer)
#     st.session_state.config = {"configurable": {"thread_id": "1"}}    

# if "user_question" not in st.session_state:
#     st.session_state.user_question = ''


# user_question = st.text_input(label = "Попитай ме нещо:", key="user_input")



# if st.button("Отговор"):
#     if user_question:

#         result = st.session_state.graph.invoke(input = {"question":HumanMessage(content= user_question)},
#              config=st.session_state.config)
                        
#         # Add to chat history
#         st.session_state.chat_history.insert(0, {
#             "question": user_question,
#             "answer": result["messages"][-1].content,
#             "query": result["query"]
#         })

        
      
# # Display chat history
# for chat in st.session_state.chat_history:
#     st.write("Question:", chat["question"])
#     with st.expander("Show SQL Query"):
#         st.code(chat["query"], language="sql")
#     st.write("Answer:", chat["answer"])
#     st.divider()

### Second verison


st.title("Ремонт в Гео Милев - търсачка")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'graph' not in st.session_state:
    st.session_state.graph= workflow.compile(checkpointer=checkpointer)
    st.session_state.config = {"configurable": {"thread_id": "1"}}    

if "user_question" not in st.session_state:
    st.session_state.user_question = ""

if "widget_input" not in st.session_state:
    st.session_state.widget_input = ""


def submit():
    st.session_state.user_question = st.session_state.widget_input
    st.session_state.widget_input = ""

st.text_input("You:", key="widget_input", placeholder="Type your question here.", on_change=submit)

if st.session_state.user_question:
    result = st.session_state.graph.invoke(input = {"question":HumanMessage(content= st.session_state.user_question)},
             config=st.session_state.config)
    
            # Add to chat history
    st.session_state.chat_history.insert(0, {
        "question": st.session_state.user_question,
        "answer": result["messages"][-1].content,
        "query": result["query"]
    })

    st.session_state.user_question = ""

# Display chat history
for chat in st.session_state.chat_history:
    st.write("Question:", chat["question"])
    with st.expander("Show SQL Query"):
        st.code(chat["query"], language="sql")
    st.write("Answer:", chat["answer"])
    st.divider()


