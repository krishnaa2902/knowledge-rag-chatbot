"""
AGENTIC RAG
"""


import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain_community.retrievers import AzureAISearchRetriever
import pprint


load_dotenv()


endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")

model = AzureChatOpenAI(
    openai_api_version="2024-08-01-preview",
    azure_deployment=model_name,
    azure_endpoint=endpoint,
    max_tokens=800
)

azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
azure_search_key = os.getenv("AZURE_SEARCH_API_KEY")

retriever = AzureAISearchRetriever(service_name=azure_search_endpoint, api_key=azure_search_key, index_name="azuresearch-index", top_k=3)

retrieval_tool = create_retriever_tool(
    retriever=retriever,
    name="Azure_AI_Search_Retriever",
    description="This tools retrieves documents from Azure AI Search based on relevance to a query."
)

tools = [retrieval_tool]

if __name__ == "__main__":

    inputs = {
        "messages": [
            ("user", "What is the latest news of the year 2024? On which date did the event happen?"),
        ]
    }
    from agentic_rag.workflow import rag_graph
    for output in rag_graph.stream(inputs):
        for key, value in output.items():
            pprint.pprint(f"Output from node '{key}':")
            pprint.pprint("---")
            pprint.pprint(value, indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")

    from agentic_rag.view_graph import display_graph
    display_graph()
