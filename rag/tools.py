from langchain.tools import tool
from rag.retriever import get_retriever


@tool
def retrieve_passages(paper_id: str, query: str, k: int = 5) -> str:
    """
    Retrieve relevant passages from the uploaded paper for grounding.
    """
    retriever = get_retriever(paper_id=paper_id, k=k)
    docs = retriever.invoke(query)

    parts = []
    for i, d in enumerate(docs, start=1):
        parts.append(f"[Passage {i}]\n{d.page_content}")
    return "\n\n".join(parts)
