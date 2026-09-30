import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings 

def build_or_load_vectorstore(
    paper_id: str,
    raw_text: str,
    persist_root: str = "storage/vectordb",
    embed_model: str = "nomic-embed-text",
    base_url: str | None = None,
) -> Chroma:
    """Build or load a vectorstore for the specified paper_id."""
    os.makedirs(persist_root, exist_ok=True)
    persist_dir = os.path.join(persist_root, paper_id)

    embeddings = OllamaEmbeddings(model=embed_model, base_url=base_url) if base_url else OllamaEmbeddings(model=embed_model)

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=f"paper_{paper_id}",
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    docs = splitter.create_documents([raw_text])

    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=f"paper_{paper_id}",
    )

def get_retriever(
    paper_id: str,
    persist_root: str = "storage/vectordb",
    k: int = 5,
    embed_model: str = "nomic-embed-text",
    base_url: str | None = None,
):
    """Get a retriever for the specified paper_id."""
    persist_dir = os.path.join(persist_root, paper_id)
    embeddings = OllamaEmbeddings(model=embed_model, base_url=base_url) if base_url else OllamaEmbeddings(model=embed_model)

    vs = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=f"paper_{paper_id}",
    )
    return vs.as_retriever(search_kwargs={"k": k})