import os
import atexit
import gradio as gr
from dotenv import load_dotenv
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential

load_dotenv()
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")

model = AzureChatOpenAI(
    openai_api_version="2024-08-01-preview",
    azure_deployment=model_name,
    azure_endpoint=endpoint,
    max_tokens=800
)

embedding_endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
embedding_model_name = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME")

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=embedding_model_name,
    openai_api_version="2023-05-15",
    azure_endpoint=embedding_endpoint,
)

azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
azure_search_key = os.getenv("AZURE_SEARCH_API_KEY")

vector_store = AzureSearch(
    azure_search_endpoint=azure_search_endpoint,
    azure_search_key=azure_search_key,
    index_name="azuresearch-index",
    embedding_function=embeddings.embed_query,
    additional_search_client_options={"retry_total": 4},
)


def index_exists(index_name):
    index_client = SearchIndexClient(endpoint=azure_search_endpoint,
                                     credential=AzureKeyCredential(azure_search_key))
    index_list = index_client.list_indexes()
    for index in index_list:
        if index.name == index_name:
            return True
    return False


def create_vector_store():
    print("Index does not exist")
    loader = PyPDFLoader("./data/2024-Wikipedia.pdf")
    pages = []
    for doc in loader.lazy_load():
        pages.append(doc)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=512, chunk_overlap=64)
    docs = text_splitter.split_documents(pages)
    vector_store.add_documents(documents=docs)


if index_exists("azuresearch-index") == False:
    create_vector_store()
    print("Index created")
else:
    print("Index exists already")


def format_documents(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def chatbot_response(query):
    retriever = AzureAISearchRetriever(service_name=azure_search_endpoint, api_key=azure_search_key, index_name="azuresearch-index", top_k=3)

    prompt_template = ChatPromptTemplate([
        ("system", "You are an expert RAG model. Based on the retrieved documents ie. context, provide an answer to the following query"),
        ("user", "context: {context} \nquery: {query}")
    ])

    llm_chain = (
        {"context": retriever | format_documents, "query": RunnablePassthrough()}
        | prompt_template
        | model
        | StrOutputParser()
    )

    response = llm_chain.invoke(query)

    return response


iface = gr.Interface(fn=chatbot_response, inputs="text", outputs="text", title="Azure AI Chatbot", description="Ask anything!")

if __name__ == "__main__":
    iface.launch()
