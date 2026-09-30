from typing import Dict
import os
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from schemas.pydantic_file import ReviewOut
from rag.tools import retrieve_passages
from src.prompts import reviewer_prompt
from src.state import GraphState

def reviewer_node(state: GraphState) -> Dict:
    paper_id = state["paper_id"]
    outputs = state.get("outputs", {})

    base_url = os.getenv("OLLAMA_BASE_URL")

    model = ChatOllama(
        model="llama3.1:8b",
        temperature=0,
        base_url=base_url
    ) if base_url else ChatOllama(
        model="llama3.1:8b",
        temperature=0
    )

    context = retrieve_passages.invoke({
        "paper_id": paper_id,
        "query": "Verify claims and evidence in extracted outputs",
        "k": 10
    })

    structured_llm = model.with_structured_output(ReviewOut)

    prompt = f"""
You are a strict scientific reviewer.

Verify whether the extracted outputs are grounded in the paper.

CONTEXT:
{context}

EXTRACTED OUTPUTS:
{outputs}

Return ONLY a valid ReviewOut JSON.
"""

    review_obj = structured_llm.invoke(prompt)

    return {
        "review": review_obj.model_dump(),
        "logs": ["Reviewer completed"]
    }