"""Unit tests for FastAPI dashboard endpoints."""

from pathlib import Path
from fastapi.testclient import TestClient
from promptdiff.dashboard.app import app
from promptdiff.db import record_run
from promptdiff.models import RunOutput, TargetConfig, TestCaseResult

client = TestClient(app)


def test_dashboard_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_dashboard_suites_and_runs_api(tmp_path: Path):
    db_path = tmp_path / "dashboard_test.db"
    run_output = RunOutput(
        run_id="run_dash_1",
        timestamp="2026-08-27T11:00:00Z",
        suite_name="dash-suite",
        target=TargetConfig(provider="gemini", model="gemini-3.6-flash"),
        results=[
            TestCaseResult(
                test_case_id="tc_dash",
                input="Hello",
                output="World",
            )
        ],
    )
    record_run(run_output, is_baseline=True, db_path=db_path)

    # Query suites with custom db param
    suites_res = client.get(f"/api/suites?db={str(db_path)}")
    assert suites_res.status_code == 200
    suites = suites_res.json()
    assert len(suites) == 1
    assert suites[0]["suite_name"] == "dash-suite"

    # Query runs with custom db param
    runs_res = client.get(f"/api/runs?suite=dash-suite&db={str(db_path)}")
    assert runs_res.status_code == 200
    runs = runs_res.json()
    assert len(runs) == 1
    assert runs[0]["id"] == "run_dash_1"

    # Query single run detail
    detail_res = client.get(f"/api/runs/run_dash_1?db={str(db_path)}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["id"] == "run_dash_1"
    assert len(detail["cases"]) == 1


def test_dashboard_index_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "PromptDiff" in response.text
