# Retrieval guide

Hybrid retrieval combines lexical search with semantic vector search. Lexical search is useful when an exact term, identifier, or phrase matters. Semantic search is useful when the question and source use different wording but express a similar idea.

Reranking is a second-stage operation. A retriever first gathers a candidate set, then a reranker scores those candidates more precisely so the most useful passages appear first.
