from typing import Any, Optional, TypedDict

from langgraph.graph import StateGraph

from fetch_node import fetch_node
from retrieve_node import retrieve_node


class GraphState(TypedDict):
    file_path: str
    documents: list
    vector_store: Optional[Any]


def fetch(state: GraphState):
    documents = fetch_node(state["file_path"])
    return {"documents": documents}


def retrieve(state: GraphState):
    vector_store = retrieve_node(state["documents"])
    return {"vector_store": vector_store}


graph_builder = StateGraph(GraphState)
graph_builder.add_node("fetch", fetch)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_edge("fetch", "retrieve")
graph_builder.set_entry_point("fetch")
graph = graph_builder.compile()


if __name__ == "__main__":
    result = graph.invoke({"file_path": "data/engro_annual_report.pdf"})
    if result.get("vector_store") is not None:
        print("Vector store created")