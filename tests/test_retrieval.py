import json
from pathlib import Path

import pytest

from src.rag.retrieve import retrieve
from src.rag.store import ingest_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((PROJECT_ROOT / "data" / "eval" / "retrieval_cases.json").read_text())


@pytest.fixture(scope="session", autouse=True)
def indexed_documents():
    ingest_documents()


@pytest.mark.parametrize("case", CASES)
def test_retrieval_returns_expected_evidence(case):
    results = retrieve(case["query"], top_k=2)
    assert any(
        result["source"] == case["expected_document"]
        and case["expected_text"] in result["text"]
        for result in results
    )
