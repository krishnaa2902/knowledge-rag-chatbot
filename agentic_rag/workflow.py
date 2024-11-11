from main_v4 import retrieval_tool
from agentic_rag.graph import agent, generate, grade_documents, rewrite
from langgraph.prebuilt import tools_condition
from agentic_rag.agent_state import AgentState
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph, START
import sys
sys.path.append('D:/MIT/0_general folders/Barclays/work/copilot poc/Gen AI')


# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode([retrieval_tool])
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node(
    "generate", generate
)

# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)

workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile
rag_graph = workflow.compile()
