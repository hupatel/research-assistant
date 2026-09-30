from typing_extensions import TypedDict
from typing import Annotated, Dict, Any, List
import operator


def merge_dict(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(left) if left else {}
    for k, v in (right or {}).items():
        out[k] = v
    return out


class GraphState(TypedDict, total=False):
    paper_id: str
    pdf_path: str
    user_goal: str

    raw_text: str
    metadata: Dict[str, str]
    section_names: List[str]

    plan: List[Dict[str, Any]]
    rag_ready: bool

    outputs: Annotated[Dict[str, Any], merge_dict]

    logs: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]

    review: Dict[str, Any]
    final_report: Dict[str, Any]