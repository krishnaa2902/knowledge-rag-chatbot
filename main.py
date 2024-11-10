import asyncio
import os
import atexit
import time
from dotenv import load_dotenv
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
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

embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=embedding_model_name,
    openai_api_version="2023-05-15",
    azure_endpoint=embedding_endpoint,
)

azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
azure_search_key = os.getenv("AZURE_SEARCH_API_KEY")

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=azure_search_endpoint,
    azure_search_key=azure_search_key,
    index_name="azuresearch-index",
    embedding_function=embeddings.embed_query,
    # Configure max retries for the Azure client
    additional_search_client_options={"retry_total": 4},
)

index_client = SearchIndexClient(endpoint=azure_search_endpoint,
                                 credential=AzureKeyCredential(azure_search_key))


def cleanup():
    try:
        vector_store.client.close()
        index_client.close()
        print("Cleanup complete.")
    except Exception as e:
        print(f"Error during cleanup: {e}")
# Function to check if an index exists


atexit.register(cleanup)


def index_exists(index_name):
    index_list = index_client.list_indexes()
    # for index in index_list:
    #     print(f"index name = {index.name}")

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

    # pages = ""
    # for doc in loader.lazy_load():
    #     pages += "page_content: " + doc.page_content + "\npage_metadata: " + doc.metadata.get("Source", "Unknown") + "\n\n"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    docs = text_splitter.split_documents(pages)

    vector_store.add_documents(documents=docs)


def main_run():
    if index_exists("azuresearch-index") == False:
        create_vector_store()
        print("Index created")
    else:
        print("Index exists already")

    user_query = "What is the most important day of the year?"

    docs_and_scores = vector_store.similarity_search_with_relevance_scores(
        query=user_query,
        k=3,
        score_threshold=0.60,
    )

    # print(f"score = {docs_and_scores[0][1]}")
    # docs = vector_store.similarity_search(
    #     query="What did the president say about Ketanji Brown Jackson",
    #     k=3,
    #     search_type="hybrid",
    # )

    retrieved_doc_string = ""
    for doc in docs_and_scores:
        retrieved_doc_string += doc[0].page_content + "\n" + doc[0].metadata.get("Source", "unknown") + "\n" + doc[0].metadata.get("Page", "unknown") + "\n\n"

    prompt_template = ChatPromptTemplate([
        ("system", "You are an expert RAG model. Based on the retrieved documents provide an answer to the following query"),
        ("user", "retrirved_docs: {retrieved_docs} \nquery: {query}")
    ])

    llm_chain = prompt_template | model | StrOutputParser()

    response = llm_chain.invoke({"retrieved_docs": retrieved_doc_string,
                                "query": user_query})

    with open("response.md", "w") as f:
        f.write(response)


if __name__ == "__main__":
    main_run()
