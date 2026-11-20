"""Retrieve FDA excerpts with FAISS, BM25, and reciprocal rank fusion."""

from functools import lru_cache

import numpy as np

from ..ingestion.embed_chunks import load_chunk_embedding_model, load_chunk_indexes


DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_COUNT = 20
DEFAULT_RRF_CONSTANT = 60
RRF_SCORE_DECIMALS = 4


_get_chunk_indexes = lru_cache(maxsize=1)(load_chunk_indexes)
_get_embedding_model = lru_cache(maxsize=1)(load_chunk_embedding_model)


def retrieve(
    query: str,
    k: int = DEFAULT_TOP_K,
    top_n: int = DEFAULT_CANDIDATE_COUNT,
    c: int = DEFAULT_RRF_CONSTANT,
    *,
    query_embedding: np.ndarray | None = None,
) -> list[dict[str, int | str | float]]:
    """Return the top-k FDA chunks, optionally using a precomputed query vector."""
    chunks, index, bm25 = _get_chunk_indexes()

    if query_embedding is None:
        query_embedding = _get_embedding_model().encode(
            [query], normalize_embeddings=True
        ).astype("float32")
    else:
        query_embedding = query_embedding.reshape(1, -1).astype("float32")
    _, semantic_ids = index.search(query_embedding, top_n)
    semantic_ids = semantic_ids[0]
    assert (semantic_ids >= 0).all(), "semantic arm returned empty results"

    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_top = np.argsort(-bm25_scores)[:top_n]

    rrf_scores = {}
    for ranked_ids in (semantic_ids, bm25_top):
        for rank, chunk_id in enumerate(ranked_ids):
            chunk_id = int(chunk_id)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (c + rank + 1)

    ranked = sorted(rrf_scores.items(), key=lambda item: -item[1])[:k]
    results = []
    for chunk_id, score in ranked:
        row = chunks.iloc[chunk_id]
        results.append(
            {
                "chunk_id": chunk_id,
                "text": row["text"],
                "source": row["source"],
                "page": int(row["page"]),
                "section": row["section"],
                "rrf_score": round(score, RRF_SCORE_DECIMALS),
            }
        )
    return results
