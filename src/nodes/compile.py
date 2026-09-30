from typing import Dict
from schemas.pydantic_file import FinalReport, ReviewOut
from src.state import GraphState


def compile_node(state: GraphState) -> Dict:
    outputs = state.get("outputs", {})
    metadata = state.get("metadata", {})
    review = state.get("review", {"grounded": False, "issues": []})

    final = FinalReport(
        outputs=outputs,
        review=ReviewOut(**review),
        metadata=metadata,
    )
    return {"final_report": final.model_dump(), "logs": ["Compiled final report"]}