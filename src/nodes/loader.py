from typing import Dict, List
import re
from langchain_community.document_loaders import PyPDFLoader  
from src.state import GraphState


SECTION_HEADINGS = {
    "abstract", "introduction", "related work", "background", "methods", "methodology",
    "experiments", "results", "discussion", "conclusion", "limitations", "references"
}
HEADING_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 \-]{2,60})\s*$")


def loader_node(state: GraphState) -> Dict:
    pdf_path = state["pdf_path"]
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    raw_text = "\n\n".join(d.page_content for d in docs if d.page_content)
    md = docs[0].metadata if docs else {}
    metadata = {str(k): str(v) for k, v in (md or {}).items()}

    section_names: List[str] = []
    for line in raw_text.splitlines():
        m = HEADING_RE.match(line.strip())
        if not m:
            continue
        cand = m.group(1).strip()
        if cand.lower() in SECTION_HEADINGS and cand not in section_names:
            section_names.append(cand)

    return {
        "raw_text": raw_text,
        "metadata": metadata,
        "section_names": section_names,
        "logs": [f"Loaded PDF pages={len(docs)}"],
    }