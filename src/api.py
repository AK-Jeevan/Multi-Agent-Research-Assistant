from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.research.workflow import run_research


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)


app = FastAPI(title="Multi-Agent Research Assistant API")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_PATH = PROJECT_ROOT / "static" / "index.html"


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    """Serve the browser UI without shadowing the API routes."""
    return FileResponse(FRONTEND_PATH)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/research")
def research(request: ResearchRequest) -> dict[str, object]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be blank.")
    result = run_research(question)

    return {
        "question": result.get("question", question),
        "summary": result.get("summary", ""),
        "needs_approval": bool(result.get("needs_approval", False)),
        "security": result.get("security", {}),
        "approval": result.get("approval", {}),
        "trace_id": result.get("trace_id"),
        "trace_summary": result.get("trace_summary", {}),
    }
