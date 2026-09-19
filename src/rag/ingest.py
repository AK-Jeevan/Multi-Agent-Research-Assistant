from src.rag.store import ingest_documents


if __name__ == "__main__":
    print(f"Indexed {ingest_documents()} document chunks.")
