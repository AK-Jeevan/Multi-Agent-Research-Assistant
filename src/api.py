from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.research.workflow import run_research


class ResearchRequest(BaseModel):
    question: str


app = FastAPI(title="Multi-Agent Research Assistant API")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/research")
def research(request: ResearchRequest) -> dict[str, object]:
    result = run_research(request.question)

    return {
        "question": result.get("question", request.question),
        "summary": result.get("summary", ""),
        "needs_approval": bool(result.get("needs_approval", False)),
        "security": result.get("security", {}),
        "approval": result.get("approval", {}),
        "trace_id": result.get("trace_id"),
        "trace_summary": result.get("trace_summary", {}),
    }
