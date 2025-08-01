import os
from dotenv import load_dotenv
from typing import Annotated,Sequence,TypedDict
from langchain_core.messages import BaseMessage,ToolMessage,SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

@tool
def add_tool(a : int,b : int):
    """ This tools helps to add two numbers """
    return a + b

tools = [add_tool]

# --- Gemini Model Setup ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key = os.getenv("GEMINI_API_KEY"),
    temperature=0.4
).bind_tools(tools)

def modal_call(state: AgentState) -> AgentState:
    system_message = SystemMessage(content="You are an AI assistant Bob.")
    response = llm.invoke([system_message] + state["messages"])
    return {"messages":[response]}

def should_continue(state : AgentState):
    lastMessage = state["messages"][-1]
    if not lastMessage.tool_calls:
        return "end"
    else:
        return "continue" 

graph = StateGraph(AgentState)

graph.add_node("our_agent",modal_call)
tool_node = ToolNode(tools=tools)
graph.add_node("our_tools",tool_node)

graph.add_edge(START,"our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "end":END,
        "continue":"our_tools",
    })
graph.add_edge("our_tools","our_agent")

agent = graph.compile()
            
initial_state = {"messages":[("user","add 45+54 then add 1 to it")]}
for step in agent.stream(initial_state, stream_mode="values"):
    step["messages"][-1].pretty_print()
    # print(step)
