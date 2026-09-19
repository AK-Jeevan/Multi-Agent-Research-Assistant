from datetime import datetime, timezone

from src.rag.store import get_embedding_function
import chromadb

from src.rag.store import CHROMA_DIR


MEMORY_COLLECTION_NAME = "research-assistant-memory"


def get_memory_collection():
    """Open a separate Chroma collection so remembered notes cannot alter source documents."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=MEMORY_COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def remember_research(question: str, sources: list[str]) -> str:
    """Persist only a small, provenance-bearing note after a supported run."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    memory_id = f"research-{timestamp}"
    note = f"Research was completed for: {question}. Sources used: {', '.join(sources)}."
    get_memory_collection().add(
        ids=[memory_id],
        documents=[note],
        metadatas=[
            {
                "kind": "research_run",
                "question": question,
                "sources": ", ".join(sources),
                "created_at": timestamp,
            }
        ],
    )
    return memory_id


def recall_memory(query: str, top_k: int = 2) -> list[dict[str, str]]:
    """Retrieve prior notes while preserving their provenance metadata."""
    collection = get_memory_collection()
    if collection.count() == 0:
        return []
    results = collection.query(query_texts=[query], n_results=top_k)
    return [
        {
            "text": text,
            "source": metadata["sources"],
            "created_at": metadata["created_at"],
        }
        for text, metadata in zip(results["documents"][0], results["metadatas"][0])
    ]
