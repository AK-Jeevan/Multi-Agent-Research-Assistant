from src.research.config import get_retrieval_settings, resolve_search


def test_get_retrieval_settings_reads_env_values(monkeypatch):
    monkeypatch.setenv("RESEARCH_RAG_TOP_K", "4")
    monkeypatch.setenv("RESEARCH_MEMORY_TOP_K", "3")
    monkeypatch.setenv("RESEARCH_WEB_MAX_RESULTS", "7")

    settings = get_retrieval_settings()

    assert settings == {"rag_top_k": 4, "memory_top_k": 3, "web_max_results": 7}


def test_resolve_search_defaults_to_existing_callable(monkeypatch):
    monkeypatch.delenv("RESEARCH_WEB_PROVIDER", raising=False)
    search = lambda query: [{"title": query, "href": "https://example.com", "body": "value"}]

    resolved = resolve_search(search)

    assert resolved is search


def test_resolve_search_disables_web_search_when_configured(monkeypatch):
    monkeypatch.setenv("RESEARCH_WEB_PROVIDER", "none")
    search = lambda query: [{"title": "fallback", "href": "https://example.com", "body": "value"}]

    resolved = resolve_search(search)

    assert resolved("What is hybrid retrieval?") == []
