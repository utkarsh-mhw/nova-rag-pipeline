"""Evaluate FDA excerpt retrieval against labeled additive queries."""

from collections.abc import Collection, Mapping

from src.retreival.retrieve import DEFAULT_TOP_K, retrieve


def evaluate_retrieval(
    labeled_queries: Mapping[str, Collection[int]], *, k: int = DEFAULT_TOP_K
) -> dict[str, float]:
    """Compute macro-averaged hit rate, reciprocal rank, and recall at k.

    Each query maps to its relevant chunk IDs from the current chunk artifact.
    """
    if k < 1:
        raise ValueError("k must be positive.")
    if not labeled_queries or any(not ids for ids in labeled_queries.values()):
        raise ValueError("Provide at least one query and relevant chunk IDs for each query.")

    hits = 0
    reciprocal_rank = 0.0
    recall = 0.0
    for query, relevant_ids in labeled_queries.items():
        relevant = set(relevant_ids)
        matches = set()
        for rank, chunk in enumerate(retrieve(query, k=k), start=1):
            chunk_id = chunk["chunk_id"]
            if chunk_id in relevant:
                if not matches:
                    hits += 1
                    reciprocal_rank += 1 / rank
                matches.add(chunk_id)
        recall += len(matches) / len(relevant)

    query_count = len(labeled_queries)
    return {
        f"hit_rate@{k}": hits / query_count,
        f"mrr@{k}": reciprocal_rank / query_count,
        f"recall@{k}": recall / query_count,
    }
