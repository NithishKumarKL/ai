# app/fastapi_app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scripts.retriever_and_llm import RAGAssistant

app = FastAPI(title="Insurance FAQ RAG Assistant")
rag = RAGAssistant()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query(request: QueryRequest):
    if not request.question or len(request.question.strip()) < 3:
        raise HTTPException(status_code=400, detail="Question too short")
    res = rag.answer(request.question.strip())
    return {
        "answer": res["answer"],
        "sources": res["sources"],
        "hits": res["hits"]
    }

# Run with: uvicorn app.fastapi_app:app --reload --host 0.0.0.0 --port 8000
