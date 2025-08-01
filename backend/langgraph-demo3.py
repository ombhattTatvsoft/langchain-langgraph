import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.4
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

