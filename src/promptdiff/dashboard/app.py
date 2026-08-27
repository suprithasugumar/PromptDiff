"""FastAPI web server for the PromptDiff dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from promptdiff.db import DEFAULT_DB_PATH, get_run_detail, get_runs_for_suite, get_suites

app = FastAPI(
    title="PromptDiff Dashboard",
    description="Local web dashboard for inspecting LLM test suite regression trends and diffs.",
    version="0.1.0",
)

# Enable CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/suites")
def list_suites(db: str = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    return get_suites(db_path=db)


@app.get("/api/runs")
def list_runs(
    suite: str | None = Query(None, description="Optional filter by suite name"),
    limit: int = Query(50, ge=1, le=500),
    db: str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    return get_runs_for_suite(suite_name=suite, limit=limit, db_path=db)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, db: str = DEFAULT_DB_PATH) -> dict[str, Any]:
    detail = get_run_detail(run_id=run_id, db_path=db)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return detail


@app.get("/", response_class=HTMLResponse)
def index():
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    return HTMLResponse("<h1>PromptDiff Dashboard static files not found</h1>", status_code=404)
