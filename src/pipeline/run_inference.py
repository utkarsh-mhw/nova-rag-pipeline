"""Analyze a product using the NutriLens pipeline."""

import os
from functools import lru_cache

import numpy as np
import pandas as pd
from huggingface_hub import InferenceClient

from src.classification.predict import load_classifier, predict_nova_group
from src.data_prep.data_cleaning import (
    clean_ingredient_text,
    stem_and_remove_stopwords_ingredients,
)
from src.explanation.explain import explain
from src.recommendation.recommend import get_product_tags, recommend_with_tags
from src.retreival.retrieve import retrieve


MODEL_ID = "NeuML/pubmedbert-base-embeddings"
_get_classifier = lru_cache(maxsize=1)(load_classifier)


@lru_cache(maxsize=1)
def _load_embedding_client() -> InferenceClient:
    return InferenceClient(token=os.environ["HF_TOKEN"])


def _embed_text(text: str) -> np.ndarray:
    response = _load_embedding_client().feature_extraction(text, model=MODEL_ID)
    embedding = np.array(response, dtype=np.float32)
    if embedding.ndim == 3:
        embedding = embedding[0]
    if embedding.ndim == 2 and embedding.shape[0] > 1:
        embedding = embedding.mean(axis=0)
    embedding = embedding.reshape(-1)
    norm = np.linalg.norm(embedding)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Hugging Face returned an invalid embedding.")
    return embedding / norm


def run_pipeline(code: str | int, product_name: str, ingredients_text: str) -> dict:
    """Analyze one product and return a JSON-serializable result."""
    # ingredient_row = pd.DataFrame({"ingredients_text": [ingredients_text]})
    # cleaned = clean_ingredient_text(ingredient_row)
    # processed = stem_and_remove_stopwords_ingredients(cleaned)
    embedding = _embed_text(ingredients_text)
    nova_group = predict_nova_group(embedding, _get_classifier())

    tags = get_product_tags(code)
    alternatives = recommend_with_tags(embedding, tags, nova_group) if tags else []
    chunks = retrieve(ingredients_text, query_embedding=embedding)
    explanation = explain(ingredients_text, nova_group, chunks)

    return {
        "code": code,
        "product_name": product_name,
        "ingredients_text": ingredients_text,
        "nova_group": nova_group,
        "categories_tags": sorted(tags),
        "recommendations": [
            {
                "similarity": alternative["similarity"],
                "match_pct": alternative["match_pct"],
                "nova_group": alternative["nova"],
                "product_name": (
                    None if pd.isna(alternative["product"]) else alternative["product"]
                ),
                "category": alternative["category"],
            }
            for alternative in alternatives
        ],
        "retrieved_chunks": chunks,
        "explanation": explanation,
    }
