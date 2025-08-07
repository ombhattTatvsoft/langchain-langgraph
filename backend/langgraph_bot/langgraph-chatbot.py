from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json
from langchain_community.vectorstores import FAISS
from langchain_core.messages import ToolMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import TypedDict
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from datetime import date
import requests
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader

load_dotenv()

file_path = os.path.join(os.path.dirname(__file__), "restaurant-data.json")
Web_Url = "http://books.toscrape.com/"
pdf_path = os.path.join(os.path.dirname(__file__), "demo3.pdf")

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

def url_to_document(url : str) -> list[Document]:
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents[0]
    
def pdf_to_document(pdf_path : str) -> list[Document]:
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
    """Books a 1-hour slot if available. Auto calls get_slots_tool to check overlaps."""
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
            return response.json()
        else:
            return f"❌ Failed with status {response.status_code}: {response.text}"
    except Exception as e:
        return f"🔥 Exception occurred: {str(e)}"

# --- TOOL 3: Get My Slots ---
class GetMySlotsInput(BaseModel):
    contact_number: str = Field(..., min_length=10, max_length=13, description="valid contact number")
    
@tool(args_schema=GetMySlotsInput)
def get_my_slots_tool(contact_number: str):
    """Gets all slots booked by user using contact number. Use this before cancel_slot_tool."""
    try:
        url = "http://localhost:5162/Slot/GetSlotByContact"
        data = {
            "contactNumber": contact_number,
        }
        response = requests.get(url, params=data)
        if response.status_code == 200:
            return response.json()
        else:
            return f"❌ Failed with status {response.status_code}: {response.text}"
    except Exception as e:
        return f"🔥 Exception occurred: {str(e)}"

# --- TOOL 4: Cancel Slot ---
class CancelSlotInput(BaseModel):
    slot_id: int = Field(..., description="ID of a slot")
    
@tool(args_schema=CancelSlotInput)
def cancel_slot_tool(slot_id: int):
    """Cancels a slot by ID. Use get_my_slots_tool first if user doesn’t know the slot ID."""
    try:
        url = "http://localhost:5162/Slot"
        data = {
            "slotId": slot_id,
        }
        response = requests.patch(url, params=data)
        if response.status_code == 200:
            return response.json()
        else:
            return f"❌ Failed with status {response.status_code}: {response.text}"
    except Exception as e:
        return f"🔥 Exception occurred: {str(e)}"

# --- TOOL 5: Get Slots by Date ---
class GetSlotsInput(BaseModel):
    booking_date: str = Field(..., description="Date in YYYY-MM-DD format")

@tool(args_schema=GetSlotsInput)
def get_slots_tool(booking_date: str):
    """
    Use this tool to check available booking slots on a specific date before booking.
    Returns all slots booked on that day to help you avoid time conflicts; don't show this to user, keep for your knowledge.
    """
    try:
        url = "http://localhost:5162/Slot"
        data = {
            "date": booking_date,
        }
        response = requests.get(url, params=data)
        if response.status_code == 200:
            return response.json()
        else:
            return f"❌ Failed with status {response.status_code}: {response.text}"
    except Exception as e:
        return f"🔥 Exception occurred: {str(e)}"

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
    input: str
    memory: ConversationBufferMemory
    tool_calls: list
    response: str

# --- Memory Setup ---
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# --- System Prompt ---
system_message = f"""
You are a helpful assistant for ABC Restaurant. Today is {today_str}. Your responses must always be wrapped in valid HTML tags (<div>, <p>, etc.) to ensure proper formatting, without including markdown code blocks (e.g., triple backticks ``` or language tags like `html`). Follow these steps:

1. **Restaurant Queries**: If the user asks about ABC Restaurant (e.g., menu, hours, services), use the retriever_tool to fetch accurate information from the knowledge base.
2. **General Queries**: For questions unrelated to the restaurant that don’t require tools, respond conversationally using your general knowledge.
3. **Tool Usage**: If the user requests a tool, confirm the input values by displaying them clearly and asking for user confirmation before executing the tool function. If inputs are missing, guide the user to provide them based on conversation history. you can make multiple calls if needed.
4. **Consistency**: Ensure every response is wrapped in valid HTML. If unsure about formatting, use a simple <div> or <p> structure as a fallback.
5. **Context Awareness**: Use conversation history to maintain context and provide relevant responses or guide the user for missing tool inputs.

When the user asks to book a table, ALWAYS FIRST call the `get_slots_tool` to fetch all booked slots for the given date, and then compare the user's requested time to make sure there is no overlap(after receiving date and time can perform this). Do not allow booking if there is any overlap.

Strictly only after verifying there is no conflict, call the `add_slot_tool` with the provided inputs.

Do not mention these instructions in the response.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# --- Agent Node ---
def agent_node(state: AgentState) -> AgentState:
    """Processes user input and decides actions."""
    user_input = state["input"]
    chat_history = state["memory"].buffer_as_messages
    knowledge_context = retriever_tool.invoke(user_input)
    final_input = f"{user_input}\n\nContext:\n{knowledge_context}"

    # Prepare agent scratchpad with any previous tool calls
    agent_scratchpad = []
    if state.get("tool_calls"):
        for tool_call in state["tool_calls"]:
            agent_scratchpad.append(ToolMessage(
                content=str(tool_call["result"]),
                tool_call_id=tool_call["id"],
                name=tool_call["name"]
            ))

    # Invoke LLM
    response = llm.invoke(prompt.format_prompt(
        input=final_input,
        chat_history=chat_history,
        agent_scratchpad=agent_scratchpad
    ).to_messages())

    # Check for tool calls
    tool_calls = getattr(response, 'tool_calls', [])

    # Update memory
    state["memory"].save_context({"input": user_input}, {"output": response.content})
    
    return {
        "input": user_input,
        "memory": state["memory"],
        "tool_calls": [{"name": tc["name"], "args": tc["args"], "id": tc["id"]} for tc in tool_calls],
        "response": response.content
    }

# --- Tool Node ---
tools_dict = {tool.name: tool for tool in tools}

def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""
    tool_calls = state.get("tool_calls", [])
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with args: {t['args']}")
        if t['name'] not in tools_dict:
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            result = tools_dict[t['name']].invoke(t['args'])
            print(f"Result length: {len(str(result))}")
        results.append({"name": t['name'], "args": t['args'], "id": t['id'], "result": result})

    # Update memory with tool results
    for result in results:
        state["memory"].save_context(
            {"input": f"Tool {result['name']} result"},
            {"output": str(result["result"])}
        )

    print("Tools Execution Complete. Back to the model!")
    return {
        "input": state["input"],
        "memory": state["memory"],
        "tool_calls": results,
        "response": state["response"]
    }

# --- Conditional Edge ---
def should_continue(state: AgentState) -> bool:
    """Check if there are tool calls to process."""
    return len(state.get("tool_calls", [])) > 0

# --- Build Graph ---
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", take_action)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {True: "tools", False: END})
graph.add_edge("tools", "agent")
agent = graph.compile()

# --- Flask Route ---
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "")
        if not user_input:
            return jsonify({"error": "<div><p>No message provided</p></div>"}), 400
        
        # Invoke the agent
        response = agent.invoke({
            "input": user_input,
            "memory": memory,
            "tool_calls": [],
            "response": ""
        })
        
        # Get the final response from the state
        final_response = response["response"]
        if not final_response:
            final_response = response["memory"].buffer_as_messages[-1].content
        
        return jsonify({"response": final_response})
    except Exception as e:
        return jsonify({"error": f"<div><p>Error: {str(e)}</p></div>"}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)