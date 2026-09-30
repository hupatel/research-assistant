import os
import uuid
import json
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv

from src.graph import build_graph

load_dotenv()

app = FastAPI(title="Automated Research Agent")
graph = build_graph()

UPLOAD_DIR = "storage/uploads"
OUTPUT_DIR = "storage/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


class AnalyzeRequest(BaseModel):
    user_goal: str


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    paper_id = str(uuid.uuid4())
    pdf_path = os.path.join(UPLOAD_DIR, f"{paper_id}.pdf")

    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    return {"paper_id": paper_id}


@app.post("/analyze/{paper_id}")
async def analyze(paper_id: str, req: AnalyzeRequest):
    pdf_path = os.path.join(UPLOAD_DIR, f"{paper_id}.pdf")
    if not os.path.exists(pdf_path):
        return {"error": "PDF not found"}

    result = graph.invoke({
        "paper_id": paper_id,
        "pdf_path": pdf_path,
        "user_goal": req.user_goal,
        "outputs": {},
        "logs": [],
        "errors": [],
    })

    out_path = os.path.join(OUTPUT_DIR, f"{paper_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result["final_report"], f, ensure_ascii=False, indent=2)

    return {
        "paper_id": paper_id,
        "final_report": result["final_report"],
        "logs": result.get("logs", []),
        "errors": result.get("errors", []),
        "output_path": out_path,
    }