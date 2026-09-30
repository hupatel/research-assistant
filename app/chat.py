from fastapi import APIRouter
from pydantic import BaseModel
import os

from langchain_ollama import ChatOllama  
from rag.tools import retrieve_passages

router = APIRouter()

class ChatRequest(BaseModel):
    paper_id: str
    question: str

@router.post("/chat")
def chat(req: ChatRequest):
    """Endpoint to handle user questions about the paper."""
    base_url = os.getenv("OLLAMA_BASE_URL")

    context = retrieve_passages.invoke({
        "paper_id": req.paper_id,
        "query": req.question,
        "k": 5
    })

    llm = ChatOllama(
        model="llama3.2:latest",
        temperature=0,
        base_url=base_url
    ) if base_url else ChatOllama(model="llama3.2:latest", temperature=0)

    msg = llm.invoke([
        ("system", "Answer using ONLY the provided context."),
        ("user", f"Context:\n{context}\n\nQuestion:\n{req.question}")
    ])

    return {"answer": msg.content, "context_used": context}