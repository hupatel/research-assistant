# Agentic Research Paper Analyzer

A production-style workflow that turns research PDFs into **structured, evidence-backed outputs** (summary, findings, results, limitations, replication notes, critique) using **LangGraph** for orchestration and **LangChain** for structured output + tool calling. Runs locally with **Ollama** and persists a paper-specific retrieval index for Q&A.

## Why this exists (real-world problem)
Teams waste hours on repetitive “paper triage” work: extracting core claims, checking evidence, capturing experimental setup, and writing review notes. This project automates that workflow with **repeatable structure** and **traceability**, so the output is usable for: literature reviews, reproducibility checks, technical due diligence, and internal knowledge capture.

## What it delivers
- **Goal-driven analysis**: You provide an “analysis goal” (e.g., *replication checklist + critique*) and the system plans tasks accordingly.   
- **Structured outputs**: Each task returns a validated schema (Pydantic) instead of unstructured text, reducing post-processing and brittle parsing.   
- **Evidence grounding via retrieval (RAG)**: Tasks are executed using retrieved passages from the paper, enabling citations/traceability in outputs.   
- **Local model execution**: Uses `ChatOllama` for local inference and supports tool calling + structured output capabilities.

## Architecture (How it Works)

This system is implemented as a **stateful agent workflow** using LangGraph.

The pipeline separates planning, execution, and validation into explicit stages:

1. **Loader** — parses the PDF and extracts raw text and metadata  
2. **Planner (Analyzer)** — uses structured output to decompose the user’s goal into concrete analysis tasks  
3. **Workers (fan‑out)** — execute tasks in parallel using retrieval‑augmented generation (RAG)  
4. **Reviewer** — validates that extracted claims are grounded in evidence  
5. **Compiler** — aggregates validated outputs into a final structured report

LangGraph’s fan‑out / fan‑in execution model enables parallel task execution with deterministic state merging, which closely mirrors how research analysis is performed.

**Pipeline**
1. **Loader**: reads PDF → text + metadata
2. **Planner (Analyzer)**: generates a task plan using structured output
3. **Workers (fan-out)**: execute each task in parallel, retrieving evidence from the indexed paper (fan-out/fan-in) 
4. **Reviewer**: checks grounding and flags issues
5. **Compiler**: assembles a final JSON report

## Tech stack
- **LangGraph** (`StateGraph`) for stateful orchestration and fan-out/fan-in patterns  
- **LangChain** for structured outputs and retrieval integration   
- **Ollama** via `langchain-ollama` (`ChatOllama`) for local LLM inference  
- **Chroma** for persisted vector index per paper
- **FastAPI** backend + **Streamlit** UI

## Quickstart
```bash
# 1) Start Ollama and pull models
ollama serve
ollama pull llama3.2
ollama pull nomic-embed-text

# 2) Install deps
pip install -r requirements.txt

# 3) Run backend
uvicorn app.main:app --reload --port 8000

# 4) Run UI
streamlit run app/streamlit_app.py