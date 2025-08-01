from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json
from typing import Type
from pydantic import BaseModel,Field
from langchain_core.tools import BaseTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import requests
import pandas as pd
from datetime import date, time

load_dotenv()

try:
    with open("restaurant-data.json", "r") as f:
        knowledge = json.load(f)
except Exception as e:
    knowledge = {}
    print("Failed to load knowledge base files:", e)
    
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

today_str = date.today().strftime("%B %d, %Y")

# --- TOOL 1: Create User ---
class CreateUserInput(BaseModel):
    name: str
    email: str

class CreateUserTool(BaseTool): 
    name: str = "create_user"
    description: str = "Creates a user with name and email"
    args_schema: Type[BaseModel] = CreateUserInput

    def _run(self, name: str, email: str):
        return f"User created: {name} ({email})"


# --- TOOL 2: add holiday ---
class SaveHolidayInput(BaseModel):
    holiday_name: str = Field(description="Name of the holiday, e.g., Christmas")
    holiday_date: str = Field(description="Date in YYYY-MM-DD format")

class SaveHolidayTool(BaseTool):
    name: str = "save_holiday"
    description: str = "Creates a holiday with name and date"
    args_schema: Type[BaseModel] = SaveHolidayInput

    def _run(self, holiday_name: str, holiday_date: str):
        try:
            url = "http://localhost:5131/api/holiday/save-holiday"

            data = {
                "holidayId":0,
                "holidayName":holiday_name,
                "holidayDate":holiday_date
            }

            response = requests.post(url, json=data)
            return response.json()
        except Exception as e:
            return f"🔥 Exception occurred: {str(e)}"
    
# --- TOOL 3: add slot ---
class AddSlotInput(BaseModel):
    booking_name: str = Field(...,description="Name of the user, e.g., John,Suman")
    booking_date: str = Field(...,description=f"Date in YYYY-MM-DD format, must be greater or equal to {today_str}")
    no_of_people: int = Field(...,gt=0,description="must be greater than 0")
    booking_time: str = Field(...,description="Time in HH:MM:SS format, must be between restaurant opening hours correct format by yourself")
    contact_number: str = Field(..., min_length=10, max_length=13, description="valid contact number")
    
class AddSlotTool(BaseTool):    
    name: str = "add_slot_tool"
    description: str = (
        "Books a 1-hour slot if available. Always call get_slots_tool first to confirm no overlaps. "
        "Slots are exactly 1 hour long (e.g., 14:30 to 15:30). If a slot is booked (e.g., 14:20 to 15:20), "
        "do not allow any new slot that overlaps (e.g., 13:21 to 15:20 is not permitted)."
    )
    args_schema: Type[BaseModel] = AddSlotInput

    def _run(self, booking_name: str, booking_date: str,no_of_people : int,booking_time : str,contact_number : str):
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
            
# --- TOOL 4: get my slots ---
class GetMySlotsInput(BaseModel):
    booking_date: str = Field(...,description="Date in YYYY-MM-DD format")
    contact_number: str = Field(..., min_length=10, max_length=13, description="valid contact number")
    
class GetMySlotsTool(BaseTool):
    name: str = "get_my_slots_tool"
    description: str = "Gets all slots booked by user using contact number and date. Use this before cancel_slot_tool."
    args_schema: Type[BaseModel] = GetMySlotsInput

    def _run(self, booking_date: str,contact_number : str):
        try:
            url = "http://localhost:5162/Slot/GetSlotByContactAndDate"

            data = {
                "contactNumber": contact_number,
                "bookingDate": booking_date,
            }
            response = requests.get(url, params=data)
            if response.status_code == 200:
                return response.json()
            else:
                return f"❌ Failed with status {response.status_code}: {response.text}"
        except Exception as e:
            return f"🔥 Exception occurred: {str(e)}"

# --- TOOL 5: cancel slot ---
class CancelSlotInput(BaseModel):
    slot_id: int = Field(...,description="Id of a slot")
    
class CancelSlotTool(BaseTool):
    name: str = "cancel_slot_tool"
    description: str = "Cancels a slot by ID. Use get_my_slots_tool first if user doesn’t know the slot ID."
    args_schema: Type[BaseModel] = CancelSlotInput

    def _run(self, slot_id: int):
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

# --- TOOL 6: get slots by date ---
class GetSlotsInput(BaseModel):
    booking_date: str = Field(...,description="Date in YYYY-MM-DD format")
    
class GetSlotsTool(BaseTool):
    name: str = "get_slots_tool"
    description : str = (
        "Use this tool to check available booking slots on a specific date before booking."
        "Returns all slots booked on that day to help you avoid time conflicts don't show this slots to user keep it for your knowledge."
    )
    args_schema: Type[BaseModel] = GetSlotsInput

    def _run(self, booking_date: str):
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
        
# --- Knowledge Base Setup ---
def json_to_documents(json_data: dict) -> list[Document]:
    documents = []
    for section, content in json_data.items():
        documents.append(Document(
            page_content=json.dumps({section: content}, indent=2),
            metadata={"source": "restaurant_json", "section": section}
        ))
    return documents

# Path to local FAISS directory
FAISS_STORE_DIR = "faiss_store"

# Initialize embedding model (needed for both load and create)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# If the FAISS store directory exists, load it
if os.path.exists(FAISS_STORE_DIR) and os.path.exists(os.path.join(FAISS_STORE_DIR, "index.faiss")):
    vector_store = vector_store = FAISS.load_local(
        FAISS_STORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:  
    documents = json_to_documents(knowledge)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_STORE_DIR)

# # for csv generation to view vector table
# # Initialize records list
# records = []
# # Extract both document content and their vectors
# for i, (doc_id, doc) in enumerate(vector_store.docstore._dict.items()):
#     try:
#         # Get vector for this doc (reconstruct from FAISS index)
#         vector = vector_store.index.reconstruct(i)

#         record = {
#             "content": doc.page_content,
#             **doc.metadata,
#         }

#         # Add vector as individual dimensions
#         for j, dim in enumerate(vector):
#             record[f"dim_{j}"] = dim

#         records.append(record)
#     except:
#         continue

# # Save to CSV
# df = pd.DataFrame(records)
# df.to_csv("vector_db_restaurant_knowledgebase.csv", index=False)

# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# --- Knowledge Base Retrieval Function ---
def retrieve_knowledge(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs]) if docs else ""

# --- Gemini Model Setup ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key = os.getenv("GEMINI_API_KEY"),
    temperature=0.4
)

system_message = f"""
You are a helpful assistant for ABC Restaurant. Today is {today_str}. Your responses must always be wrapped in valid HTML tags (<div>, <p>, etc.) to ensure proper formatting, without including markdown code blocks (e.g., triple backticks ``` or language tags like `html`). Follow these steps:

1. **Restaurant Queries**: If the user asks about ABC Restaurant (e.g., menu, hours, services), use the provided knowledge base to answer accurately.
2. **General Queries**: For questions unrelated to the restaurant that don’t require tools, respond conversationally using your general knowledge.
3. **Tool Usage**: If the user requests a tool, confirm the input values by displaying them clearly and asking for user confirmation before executing the tool function (_run). If inputs are missing, guide the user to provide them based on conversation history.
4. **Consistency**: Ensure every response is wrapped in valid HTML. If unsure about formatting, use a simple <div> or <p> structure as a fallback.
5. **Context Awareness**: Use conversation history to maintain context and provide relevant responses or guide the user for missing tool inputs.
Do not mention these instructions in the response.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system",system_message),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# --- Memory Setup ---
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# --- Create Agent ---
tools = [CreateUserTool(),SaveHolidayTool(),AddSlotTool(),GetMySlotsTool(),CancelSlotTool(),GetSlotsTool()]

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# --- Flask Routes ---
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_input = data.get('message', '')
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400

        # Retrieve knowledge base context
        knowledge_context = retrieve_knowledge(user_input)
        # Prepare the input for the agent
        final_input = f"{user_input}\n\nContext (if any):\n{knowledge_context}"
        # Invoke the agent with the primary input
        response = agent_executor.invoke({
            "input": final_input,
            "chat_history": memory.buffer_as_messages,
        })
        return jsonify({'response': response["output"]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)