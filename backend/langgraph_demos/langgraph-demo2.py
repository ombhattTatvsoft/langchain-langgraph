import os
from dotenv import load_dotenv
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

document_content = ""

@tool
def update_doc(content: str) -> str:
    """This tool helps update the document content."""
    global document_content
    document_content = content
    return f"Document updated with content: {content}"

@tool
def save_doc(filename: str) -> str:
    """This tool saves the document as a text file."""
    global document_content
    try:
        with open(filename, "w") as f:
            f.write(document_content)
        return f"Document saved with filename: {filename}"
    except Exception as e:
        return f"Document not saved. Error: {str(e)}"

tools = [update_doc, save_doc]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.4
).bind_tools(tools)

def modal_call(state: AgentState) -> AgentState:
    system_message = SystemMessage(content=f"""
You are an AI assistant named Bob who helps update(provide content) and save content in a file if {document_content} is not empty. 
Always show the current document state after modifications.
Current document content: {document_content}""")

    if not state["messages"]:
        user_input = input("Let's create a document: ")
    else:
        user_input = input("Would you like to update or save the document? ")

    user_message = HumanMessage(content=user_input)
    response = llm.invoke([system_message] + state["messages"] + [user_message])

    return {"messages": state["messages"] + [user_message, response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage) and last_message.name == "save_doc":
        return "end"
    return "continue"

graph = StateGraph(AgentState)

graph.add_node("our_agent", modal_call)
graph.add_node("our_tools", ToolNode(tools=tools))

graph.add_edge(START, "our_agent")
graph.add_edge("our_agent", "our_tools")
graph.add_conditional_edges("our_tools", should_continue, {
    "end": END,
    "continue": "our_agent",
})

agent = graph.compile()

for step in agent.stream({"messages": []}, stream_mode="values"):
    if(len(step["messages"]) != 0):
        step["messages"][-1].pretty_print()
