import argparse

from src.rag.store import get_collection


def retrieve(query: str, top_k: int = 3) -> list[dict[str, str]]:
    """Return the closest indexed chunks with their source and distance."""
    collection = get_collection()
    if collection.count() == 0:
        raise RuntimeError("The index is empty. Run `python -m src.rag.ingest` first.")
    results = collection.query(query_texts=[query], n_results=top_k)
    return [
        {
            "text": text,
            "source": metadata["source"],
            "distance": str(distance),
        }
        for text, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve relevant research documents.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    for result in retrieve(args.query, args.top_k):
        print(f"[{result['source']}, distance={result['distance']}]\n{result['text']}\n")


if __name__ == "__main__":
    main()
