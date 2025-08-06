import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langgraph.graph.message import add_messages
from langchain_community.vectorstores import FAISS

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# load pdf
pdf_path = os.path.join(os.path.dirname(__file__), "demo3.pdf")
pdf_loader = PyPDFLoader(pdf_path)
pages = pdf_loader.load()

# Path to local FAISS directory
FAISS_STORE_DIR = "faiss_store_2"
# If the FAISS store directory exists, load it
if os.path.exists(FAISS_STORE_DIR) and os.path.exists(os.path.join(FAISS_STORE_DIR, "index.faiss")):
    vector_store = vector_store = FAISS.load_local(
        FAISS_STORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_STORE_DIR)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

@tool
def retriever_tool(query:str) -> str:
    """This tool helps to search the document and return information."""
    docs = retriever.invoke(query)
    if not docs:
        return "I found no relevant information"
    
    results = []
    for i,doc in enumerate(docs):
        results.append(f"Document {i+1}: {doc.page_content}")
    return "".join(results)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.4
)

tools = [retriever_tool]
llm = llm.bind_tools(tools)

system_prompt = """
You are an intelligent AI assistant who answers about e-commerce system database schema based on the loaded knowledgebase.
Use the retriever tool to answer question about e-commerce system database schema otherwise use your knowledge to give answer, you can make multiple calls if needed.
If you need to look up some information before asking a follow-up question, you are allowed to do that.
Please always cite the specific parts of documents in your answers.
"""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

tools_dict = {our_tool.name : our_tool for our_tool in tools}

def should_continue(state : AgentState):
    """Check if last message was a tool call"""
    result = state["messages"][-1]
    return hasattr(result,'tool_calls') and len(result.tool_calls)>0

def model_call(state: AgentState) -> AgentState:
    """Function to call LLM"""
    messages = list(state["messages"])
    system_message = SystemMessage(content=system_prompt)
    response = llm.invoke([system_message] + messages)
    return {"messages":[response]}

# Retriever Agent
def take_action(state: AgentState):
    """Execute tool calls from the LLM's response."""
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict:  # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
        
        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}

graph = StateGraph(AgentState)

graph.add_node("model_call",model_call)
graph.add_node("retreiver_agent",take_action)

graph.add_conditional_edges("model_call",should_continue,
    {True:"retreiver_agent",False:END}
)
graph.add_edge("retreiver_agent","model_call")
graph.set_entry_point("model_call")

agent = graph.compile()

def run_agent():
    while True:
        user_input = input("User : ")
        if user_input == "EXIT":
            break
        messages = HumanMessage(content=user_input)
        response = agent.invoke({"messages":messages})
        print("\n\n"+response["messages"][-1].content)
        
run_agent()