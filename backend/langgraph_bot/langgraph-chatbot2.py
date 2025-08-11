from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from datetime import date
import requests
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

file_path = os.path.join(os.path.dirname(__file__), "restaurant_info.json")
Web_Url = "https://en.wikipedia.org/wiki/Main_Page"
pdf_path = os.path.join(os.path.dirname(__file__), "restaurant_details.pdf")

try:
    with open(file_path, "r") as f:
        knowledge = json.load(f)
except Exception as e:
    knowledge = {}
    print("Failed to load knowledge base files:", e)
    
app = Flask(__name__)
CORS(app)

today_str = date.today().strftime("%B %d, %Y")

# --- Knowledge Base Setup ---
def json_to_documents(json_data: dict) -> list[Document]:
    documents = []
    for section, content in json_data.items():
        documents.append(Document(
            page_content=json.dumps({section: content}, indent=2),
            metadata={"source": "restaurant_json", "section": section}
        ))
    return documents

def url_to_document(url: str) -> list[Document]:
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents[0]

def pdf_to_document(pdf_path: str) -> list[Document]:
    pdf_loader = PyPDFLoader(pdf_path)
    pages = pdf_loader.load()
    return pages

FAISS_STORE_DIR = os.path.join(os.path.dirname(__file__), "faiss_store")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

if os.path.exists(FAISS_STORE_DIR) and os.path.exists(os.path.join(FAISS_STORE_DIR, "index.faiss")):
    vector_store = FAISS.load_local(
        FAISS_STORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:  
    documents = json_to_documents(knowledge)
    documents.append(url_to_document(Web_Url))
    documents.extend(pdf_to_document(pdf_path))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_STORE_DIR)
    
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# --- TOOL 1: RAG Retrieval ---
@tool
def retriever_tool(query: str) -> str:
    """This tool helps to search the restaurant knowledge base and return information."""
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs]) if docs else "No relevant information found."

# --- TOOL 2: Add Slot ---
class AddSlotInput(BaseModel):
    booking_name: str = Field(..., description="Name of the user, e.g., John, Suman")
    booking_date: str = Field(..., description=f"Date in YYYY-MM-DD format, > {today_str}")
    no_of_people: int = Field(..., gt=0, description="Number of people, > 0")
    booking_time: str = Field(..., description="Time in HH:MM:SS format, must be between restaurant opening hours")
    contact_number: str = Field(..., min_length=10, max_length=13, description="valid contact number")

@tool(args_schema=AddSlotInput)
def add_slot_tool(booking_name: str, booking_date: str, no_of_people: int, booking_time: str, contact_number: str):
    """Books a 1-hour slot if available."""
    try:
        url = "http://localhost:5162/Slot"
        data = {
            "id": 0,
            "bookingName": booking_name,
            "bookingDate": booking_date,
            "noOfPeople": no_of_people,
            "bookingTime": booking_time,
            "contactNumber": contact_number,
            "isActive": True
        }
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return f"<div><p>Slot booked successfully for {booking_name} on {booking_date} at {booking_time} for {no_of_people} people.</p></div>"
        else:
            return f"<div><p>❌ Failed with status {response.status_code}: {response.text}</p></div>"
    except Exception as e:
        return f"<div><p>🔥 Exception occurred: {str(e)}</p></div>"

# --- TOOL 3: Get My Slots ---
class GetMySlotsInput(BaseModel):
    contact_number: str = Field(..., min_length=10, max_length=13, description="valid contact number")
    
@tool(args_schema=GetMySlotsInput)
def get_my_slots_tool(contact_number: str):
    """Gets all slots booked by user using contact number."""
    try:
        url = "http://localhost:5162/Slot/GetSlotByContact"
        data = {
            "contactNumber": contact_number,
        }
        response = requests.get(url, params=data)
        if response.status_code == 200:
            return response.json()
        else:
            return f"<div><p>❌ Failed with status {response.status_code}: {response.text}</p></div>"
    except Exception as e:
        return f"<div><p>🔥 Exception occurred: {str(e)}</p></div>"

# --- TOOL 4: Cancel Slot ---
class CancelSlotInput(BaseModel):
    slot_id: int = Field(..., description="ID of a slot")
    
@tool(args_schema=CancelSlotInput)
def cancel_slot_tool(slot_id: int):
    """Cancels a slot by ID."""
    try:
        url = "http://localhost:5162/Slot"
        data = {
            "slotId": slot_id,
        }
        response = requests.patch(url, params=data)
        if response.status_code == 200:
            return response.json()
        else:
            return f"<div><p>❌ Failed with status {response.status_code}: {response.text}</p></div>"
    except Exception as e:
        return f"<div><p>🔥 Exception occurred: {str(e)}</p></div>"

# --- TOOL 5: Get Slots by Date ---
class GetSlotsInput(BaseModel):
    booking_date: str = Field(..., description="Date in YYYY-MM-DD format")

@tool(args_schema=GetSlotsInput)
def get_slots_tool(booking_date: str):
    """Checks available booking slots on a specific date."""
    try:
        url = "http://localhost:5162/Slot"
        data = {
            "date": booking_date,
        }
        response = requests.get(url, params=data)
        if response.status_code == 200:
            return response.json()
        else:
            return f"<div><p>❌ Failed with status {response.status_code}: {response.text}</p></div>"
    except Exception as e:
        return f"<div><p>🔥 Exception occurred: {str(e)}</p></div>"

# --- LLM Setup ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.4
)

tools = [retriever_tool, add_slot_tool, get_my_slots_tool, cancel_slot_tool, get_slots_tool]
llm = llm.bind_tools(tools)

# --- Agent State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# --- System Prompt ---
system_message = f"""
You are a helpful assistant for ABC Restaurant. Today is {today_str}. Your responses must always be wrapped in valid HTML tags (<div>, <p>, etc.) without markdown code blocks(```html). Follow these steps:
    **Restaurant Queries** : For questions about ABC Restaurant (e.g., menu, hours), use retriever_tool to fetch information.
    **General Queries** : For non-restaurant questions without tools, respond conversationally.
    **Booking Process**:
    - When a user requests to book a table (e.g., contains "add slot"), 
    - On confirmation, call get_slots_tool to check booked slots for the given date.
    - Compare the requested time with existing slots (each slot is 1 hour). If no overlap, call add_slot_tool with the parsed details. If overlap, inform the user and stop.
    - Do not ask for confirmation more than once for the same booking request.
    **Tool Usage** : For other tool requests, confirm inputs and execute after user confirmation.
    **Context Awareness** : Use the full conversation history in messages to maintain context and avoid redundant questions.
    If the user speaks in another language translate it in english to use for tools and knowledge base then reconvert the response to user's input language.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# --- Agent Node ---
def agent_node(state: AgentState) -> AgentState:
    user_input = state["messages"][-1].content
    knowledge_context = retriever_tool.invoke(user_input)
    final_input = f"{user_input}\n\nContext:\n{knowledge_context}"
    
    # Prepare agent scratchpad
    agent_scratchpad = [msg for msg in state["messages"] if isinstance(msg, ToolMessage)]
    
    messages = state["messages"][:-1] + [HumanMessage(content=final_input)]
    
    response = llm.invoke(prompt.format_prompt(
        messages=messages,
        agent_scratchpad=agent_scratchpad
    ).to_messages())
    
    tool_calls = getattr(response, 'tool_calls', [])
    messages = state["messages"] + [AIMessage(content=response.content, tool_calls=tool_calls)]
    
    return {"messages": messages}

tools_dict = {tool.name: tool for tool in tools}

def take_action(state: AgentState) -> AgentState:
    tool_calls = getattr(state["messages"][-1], 'tool_calls', [])
    messages = state["messages"]
    
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with args: {t['args']}")
        if t['name'] not in tools_dict:
            result = f"<div><p>Incorrect Tool Name, Please Retry and Select tool from List of Available tools.</p></div>"
        else:
            result = tools_dict[t['name']].invoke(t['args'])
        messages.append(ToolMessage(content=str(result), tool_call_id=t["id"], name=t["name"]))
    
    return {"messages": messages}

# --- Conditional Edge ---
def should_continue(state: AgentState) -> bool:
    return len(getattr(state["messages"][-1], 'tool_calls', [])) > 0

# --- Build Graph ---
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", take_action)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {True: "tools", False: END})
graph.add_edge("tools", "agent")
agent = graph.compile(checkpointer=MemorySaver())

# --- Flask Route ---
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "")
        if not user_input:
            return jsonify({"error": "<div><p>No message provided</p></div>"}), 400
        
        # Load previous state or initialize new
        config = {"configurable": {"thread_id": "default"}}
        previous_state = agent.get_state(config).values if agent.get_state(config) else {}
        state = {
            "messages": previous_state.get("messages", []) + [HumanMessage(content=user_input)]
        }
        print("State messages:", [msg.content for msg in state["messages"]])
        
        # Invoke the agent
        response = agent.invoke(state, config=config)
        
        # Get the final response
        final_response = response["messages"][-1].content
        
        return jsonify({"response": final_response})
    except Exception as e:
        return jsonify({"error": f"<div><p>Error: {str(e)}</p></div>"}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)