"""Recommend less-processed products from the local product catalog."""

from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PRODUCT_INDEX_PATH = PROCESSED_DIR / "full_index.parquet"
PRODUCT_EMBEDDINGS_PATH = PROCESSED_DIR / "emb_full.npy"
PRODUCT_CATALOG_PATH = PROCESSED_DIR / "product_catalog.csv"
# CLEANED_PRODUCTS_PATH = PROJECT_ROOT / "data" / "interim" / "only_english.parquet"
# CLEANED_PRODUCTS_PATH = PROJECT_ROOT / "data" / "processed" / "full_index.parquet"
CLEANED_PRODUCTS_PATH = PROJECT_ROOT / "data" / "processed" / "only_english.parquet"
NOVA_GROUPS = range(1, 5)
DEFAULT_TOP_K = 5
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 100
HNSW_EF_SEARCH = 256
CATEGORY_NEIGHBORS_PER_GROUP = 10
CATEGORY_VOTE_POOL = 20
CATEGORY_LIMIT = 3
RECOMMENDATION_NEIGHBORS_PER_GROUP = 200
SIMILARITY_DECIMALS = 3
MATCH_PERCENT_DECIMALS = 1


def _to_tag_set(value):
    if value is None or isinstance(value, float):
        return frozenset()
    if isinstance(value, str):
        return frozenset(tag.strip() for tag in value.split(",") if tag.strip())
    return frozenset(value)


@lru_cache(maxsize=1)
def _load_catalog_tags():
    catalog = pd.read_csv(
        PRODUCT_CATALOG_PATH,
        usecols=["code", "categories_tags"],
        dtype={"code": str},
    )
    return catalog.drop_duplicates("code").set_index("code")["categories_tags"]


def get_product_tags(code: str | int) -> frozenset[str]:
    """Look up category tags by barcode, preserving leading zeros in strings."""
    return _to_tag_set(_load_catalog_tags().get(str(code)))


@lru_cache(maxsize=1)
def _load_catalog():
    embeddings = np.load(PRODUCT_EMBEDDINGS_PATH).astype("float32")
    products = pd.read_parquet(PRODUCT_INDEX_PATH).reset_index(drop=True)
    columns = ["code", "product_name"]
    if "categories_tags" not in products.columns:
        columns.append("categories_tags")
    details = pd.read_parquet(CLEANED_PRODUCTS_PATH, columns=columns)
    details["code"] = details["code"].astype(str)
    products["code"] = products["code"].astype(str)
    products = products.merge(details.drop_duplicates("code"), on="code", how="left")
    assert len(products) == len(embeddings)

    nova = products["nova_group"].astype(int).values
    categories = [_to_tag_set(value) for value in products["categories_tags"]]
    tag_frequency = Counter(tag for tags in categories for tag in tags)
    return products, embeddings, nova, categories, tag_frequency


@lru_cache(maxsize=1)
def _load_group_indexes():
    _, embeddings, nova, _, _ = _load_catalog()
    indexes = {}
    for group in NOVA_GROUPS:
        rows = np.where(nova == group)[0]
        index = faiss.IndexHNSWFlat(
            embeddings.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT
        )
        index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        index.add(embeddings[rows])
        index.hnsw.efSearch = HNSW_EF_SEARCH
        indexes[group] = index, rows
    return indexes


def _top_categories(vector, exclude, indexes, categories, tag_frequency):
    hits = []
    for group in NOVA_GROUPS:
        index, rows = indexes[group]
        similarities, neighbors = index.search(vector, CATEGORY_NEIGHBORS_PER_GROUP)
        hits.extend(
            (score, rows[neighbor])
            for score, neighbor in zip(similarities[0], neighbors[0])
        )
    hits.sort(reverse=True)
    category_rows = [row for _, row in hits if row != exclude][:CATEGORY_VOTE_POOL]
    votes = Counter(tag for row in category_rows for tag in categories[row])
    tags = [tag for tag, count in votes.items() if count >= CATEGORY_VOTE_POOL / 2]
    tags.sort(key=lambda tag: tag_frequency[tag])
    return tags[:CATEGORY_LIMIT]


def recommend(
    vec: np.ndarray, query_group: int, exclude: int | None = None, k: int = DEFAULT_TOP_K
):
    """Infer a category and return similar products from lower NOVA groups."""
    products, _, nova, categories, tag_frequency = _load_catalog()
    indexes = _load_group_indexes()
    vector = vec.reshape(1, -1).astype("float32")
    tags = _top_categories(vector, exclude, indexes, categories, tag_frequency)
    candidates = []
    for group in range(1, query_group):
        index, rows = indexes[group]
        similarities, neighbors = index.search(
            vector, RECOMMENDATION_NEIGHBORS_PER_GROUP
        )
        candidates.extend(
            (score, rows[neighbor])
            for score, neighbor in zip(similarities[0], neighbors[0])
        )
    candidates.sort(reverse=True)

    for tag in tags:
        results, seen = [], set()
        for score, row in candidates:
            if tag not in categories[row]:
                continue
            text = products.loc[row, "ingredients_text"]
            if text in seen:
                continue
            seen.add(text)
            results.append(
                (
                    round(float(score), SIMILARITY_DECIMALS),
                    int(nova[row]),
                    products.loc[row, "product_name"],
                )
            )
            if len(results) == k:
                break
        if results:
            return tag, results
    return (tags[0] if tags else None), []


@lru_cache(maxsize=1)
def _load_tag_rows():
    products, _, _, categories, _ = _load_catalog()
    tag_rows = defaultdict(list)
    for row, tags in enumerate(categories):
        for tag in tags:
            tag_rows[tag].append(row)
    return tag_rows, products["ingredients_text"].astype(str).values


def recommend_with_tags(
    vec: np.ndarray, tags: Iterable[str], query_group: int, k: int = DEFAULT_TOP_K
):
    """Rank products separately within each lower NOVA group using known tags."""
    products, embeddings, nova, _, tag_frequency = _load_catalog()
    tag_rows, ingredients = _load_tag_rows()
    tags = sorted((tag for tag in tags if tag in tag_rows), key=tag_frequency.__getitem__)
    if not tags:
        return []
    results = []
    for group in range(query_group - 1, 0, -1):
        best = []
        for tag in tags:
            rows = np.array(tag_rows[tag])
            rows = rows[nova[rows] == group]
            if len(rows) == 0:
                continue
            similarities = embeddings[rows] @ vec
            matches, seen = [], set()
            for offset in np.argsort(-similarities):
                row = rows[offset]
                text = ingredients[row]
                if text in seen:
                    continue
                seen.add(text)
                score = float(similarities[offset])
                matches.append(
                    {
                        "product": products.loc[row, "product_name"],
                        "nova": group,
                        "similarity": round(score, SIMILARITY_DECIMALS),
                        "match_pct": round(score * 100, MATCH_PERCENT_DECIMALS),
                        "category": tag,
                    }
                )
                if len(matches) == k:
                    break
            if len(matches) >= k:
                best = matches
                break
            if len(matches) > len(best):
                best = matches
        results.extend(best)
    return results
