from agentic_rag.workflow import rag_graph
from IPython.display import Image, display


def display_graph():
    try:
        display(Image(rag_graph.get_graph(xray=True).draw_mermaid_png(output_file_path="graph.png")))
    except Exception:
        # This requires some extra dependencies and is optional
        pass
