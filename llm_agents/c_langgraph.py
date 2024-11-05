from langgraph.graph import StateGraph, END
from typing import TypedDict

class BasicState(TypedDict):
    count: int

def create_simple_graph():
    graph = StateGraph(BasicState)
    
    def increment_node(state: BasicState):
        return {"count": state["count"] + 1}
    
    graph.add_node("increment", increment_node)
    graph.set_entry_point("increment")
    graph.add_edge("increment", END)
    
    return graph.compile()

agent = create_simple_graph()
agent.invoke(input={"count":0})