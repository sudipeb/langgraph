import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph, MessagesState
from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON="agent_reason"
ACT="act"
LAST=-1


def _message_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "\n".join(part for part in text_parts if part).strip()

    return str(content)

def should_continue(state:MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT


flow =StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)
flow.add_conditional_edges(AGENT_REASON, should_continue,{
    END:END,
    ACT:ACT
})
flow.add_edge(ACT, AGENT_REASON)
app=flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")
# flow.add_edge(START,AGENT_REASON)
if __name__== "__main__":
    print('Hello ReAct Langgraph with function calling')
    res = app.invoke({"messages":[HumanMessage(content="What is the weather in  Kathmandu? List it and triple all factors")]})
    print(_message_text(res["messages"][LAST].content))
