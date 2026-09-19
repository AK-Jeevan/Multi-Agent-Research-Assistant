from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from chromadb.utils import embedding_functions


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "research-assistant-documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embedding_function():
    """Create the local embedding function used by both indexing and querying."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def get_collection(reset: bool = False):
    """Open the persistent Chroma collection, optionally replacing its contents."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except (NotFoundError, ValueError):
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def chunk_document(text: str, chunk_size: int = 700) -> list[str]:
    """Split a document into paragraph-aware chunks without cutting words."""
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip()
        if current and len(candidate) > chunk_size:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def ingest_documents() -> int:
    """Read Markdown files, chunk them, and persist their embeddings in Chroma."""
    collection = get_collection(reset=True)
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str]] = []

    for path in sorted(DOCUMENTS_DIR.glob("*.md")):
        for index, chunk in enumerate(chunk_document(path.read_text(encoding="utf-8"))):
            ids.append(f"{path.stem}-{index}")
            documents.append(chunk)
            metadatas.append({"source": path.name, "chunk_index": str(index)})

    if not documents:
        raise RuntimeError(f"No Markdown documents found in {DOCUMENTS_DIR}")
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(documents)
