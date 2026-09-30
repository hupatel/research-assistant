from typing import Dict, Any, Type
import os

from langchain_ollama import ChatOllama 
from rag.tools import retrieve_passages
from src.prompts import worker_prompt
from src.state import GraphState

from schemas.pydantic_file import (
    SummaryOut, FindingsOut, ResultsOut, LimitationsOut,
    ExperimentalSetupOut, ReplicationChecklistOut, PeerReviewOut, AlgorithmExtractionOut
)

SCHEMA_REGISTRY: Dict[str, Type] = {
    "SummaryOut": SummaryOut,
    "FindingsOut": FindingsOut,
    "ResultsOut": ResultsOut,
    "LimitationsOut": LimitationsOut,
    "ExperimentalSetupOut": ExperimentalSetupOut,
    "ReplicationChecklistOut": ReplicationChecklistOut,
    "PeerReviewOut": PeerReviewOut,
    "AlgorithmExtractionOut": AlgorithmExtractionOut,
}

def worker_node(state: GraphState) -> Dict:
    """Worker node that performs a specific analysis task based on the provided schema_id."""
    paper_id = state["paper_id"]
    task: Dict[str, Any] = state["task"]
    schema_id = task["schema_id"]

    schema_cls = SCHEMA_REGISTRY.get(schema_id, SummaryOut)
    base_url = os.getenv("OLLAMA_BASE_URL")

    query = f"{task.get('task_name','')} | {task.get('task_goal','')} | sections: {task.get('section_focus',[])}"
    passages = retrieve_passages.invoke({"paper_id": paper_id, "query": query, "k": 6})

    llm = ChatOllama(model="llama3.2:latest", temperature=0, base_url=base_url) if base_url else ChatOllama(model="llama3.2:latest", temperature=0) 
    structured_llm = llm.with_structured_output(schema_cls) 

    prompt = worker_prompt(
        paper_id=paper_id,
        task=task,
        existing_outputs=state.get("outputs", {}),
    )

    messages = [
        ("system", "Use ONLY the retrieved passages as evidence. If missing, leave fields empty; do not guess."),
        ("user", f"{prompt}\n\n=== RETRIEVED PASSAGES (EVIDENCE) ===\n{passages}")
    ]

    obj = structured_llm.invoke(messages)

    key = task.get("task_name") or f"{schema_id}_{task.get('task_id','')}"
    return {"outputs": {key: obj.model_dump()}, "logs": [f"Worker finished: {key}"]}