import os
from typing import Callable, Sequence, TypeAlias

SearchCallable: TypeAlias = Callable[[str], Sequence[object]]


def get_retrieval_settings() -> dict[str, int]:
    """Return retrieval settings from environment variables with explicit defaults."""
    return {
        "rag_top_k": int(os.getenv("RESEARCH_RAG_TOP_K", "2")),
        "memory_top_k": int(os.getenv("RESEARCH_MEMORY_TOP_K", "2")),
        "web_max_results": int(os.getenv("RESEARCH_WEB_MAX_RESULTS", "5")),
    }


def resolve_search(search: SearchCallable | None = None) -> SearchCallable:
    """Return a safe search callable, honoring explicit provider configuration.

    The default behavior preserves the existing workflow. A provider named
    `none` disables external web lookups and returns an empty list.
    """
    provider = (os.getenv("RESEARCH_WEB_PROVIDER") or "default").strip().lower()

    if provider == "none":
        def _disabled(query: str):
            return []

        return _disabled

    if search is None:
        def _default(query: str):
            return []

        return _default

    return search
