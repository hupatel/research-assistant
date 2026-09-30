from typing import Dict, Any, List
import uuid
import os
from langchain_ollama import ChatOllama
from schemas.pydantic_file import PlanTasks
from rag.retriever import build_or_load_vectorstore
from src.prompts import planner_prompt
from src.state import GraphState

ALLOWED_SCHEMA_IDS = [
    "SummaryOut",
    "FindingsOut",
    "ResultsOut",
    "LimitationsOut",
    "ExperimentalSetupOut",
    "ReplicationChecklistOut",
    "PeerReviewOut",
    "AlgorithmExtractionOut",
]

def analyzer_node(state: GraphState) -> Dict:
    paper_id = state["paper_id"]
    raw_text = state["raw_text"]
    metadata = state.get("metadata", {})
    section_names = state.get("section_names", [])
    user_goal = state.get("user_goal", "Analyze the paper.")

    base_url = os.getenv("OLLAMA_BASE_URL")

    build_or_load_vectorstore(
        paper_id=paper_id,
        raw_text=raw_text,
        base_url=base_url
    )

    llm = ChatOllama(
        model="llama3.2:latest",
        temperature=0,
        base_url=base_url
    ) if base_url else ChatOllama(
        model="llama3.2:latest",
        temperature=0
    )

    planner = llm.with_structured_output(PlanTasks)

    prompt = planner_prompt(
        user_goal=user_goal,
        metadata=metadata,
        section_names=section_names,
        allowed_schema_ids=ALLOWED_SCHEMA_IDS,
    )

    plan_obj: PlanTasks = planner.invoke(prompt)

    tasks: List[Dict[str, Any]] = []

    for t in plan_obj.tasks:
        td = t.model_dump()

        if not td.get("task_id"):
            td["task_id"] = str(uuid.uuid4())

        if td["schema_id"] not in ALLOWED_SCHEMA_IDS:
            td["schema_id"] = "SummaryOut"

        tasks.append(td)

    return {
        "plan": tasks,
        "rag_ready": True,
        "logs": [f"Planner created {len(tasks)} tasks"]
    }