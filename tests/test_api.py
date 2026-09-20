from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_research_endpoint_returns_summary_and_metadata():
    response = client.post(
        "/research",
        json={"question": "What is hybrid retrieval?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["question"] == "What is hybrid retrieval?"
    assert "summary" in payload
    assert "needs_approval" in payload
    assert payload["needs_approval"] is True
