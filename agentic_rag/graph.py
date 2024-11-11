import os
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from main_v4 import tools


def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    class grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")

    model = AzureChatOpenAI(
        openai_api_version="2024-08-01-preview",
        azure_deployment=model_name,
        azure_endpoint=endpoint,
        max_tokens=800,
        streaming=True
    )
    llm_with_tool = model.with_structured_output(grade)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    chain = prompt | llm_with_tool

    messages = state["messages"]

    question = messages[0].content
    docs = messages[-1].content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"


def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """

    print("---CALL AGENT---")
    messages = state["messages"]

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")

    model = AzureChatOpenAI(
        openai_api_version="2024-08-01-preview",
        azure_deployment=model_name,
        azure_endpoint=endpoint,
        max_tokens=800,
        streaming=True
    )
    model = model.bind_tools(tools)

    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n
    Look at the input and try to reason about the underlying semantic intent / meaning. \n
    Here is the initial question:
    \n ------- \n
    {question}
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")

    model = AzureChatOpenAI(
        openai_api_version="2024-08-01-preview",
        azure_deployment=model_name,
        azure_endpoint=endpoint,
        max_tokens=800,
        streaming=True
    )
    response = model.invoke(msg)
    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    query = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    messages = [
        ("system", "You are an expert RAG model. Based on the retrieved documents ie. context, provide an answer to the following query"),
        ("user", "context: {context} \nquery: {query}")
    ]

    prompt_template = ChatPromptTemplate(messages=messages)

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")

    model = AzureChatOpenAI(
        openai_api_version="2024-08-01-preview",
        azure_deployment=model_name,
        azure_endpoint=endpoint,
        max_tokens=800,
        streaming=True
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = prompt_template | model | StrOutputParser()

    response = rag_chain.invoke({"context": docs, "query": query})

    return {"messages": [response]}
