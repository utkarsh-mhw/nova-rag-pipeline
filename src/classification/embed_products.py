"""Encode product ingredients with the notebook's full-dataset embedding model."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_PRODUCTS_PATH = PROJECT_ROOT / "data" / "interim" / "only_english.parquet"
MODEL_ID = "NeuML/pubmedbert-base-embeddings"
EMBEDDING_DEVICE = "cuda"
BATCH_SIZE = 128
CHUNK_SIZE = 50_000
ID_COLUMN = "code"
TEXT_COLUMN = "ingredients_text"
LABEL_COLUMN = "nova_group"


def load_embedding_model() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_ID, device=EMBEDDING_DEVICE).half()


def encode_products(
    texts: list[str],
    model: SentenceTransformer,
    *,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """Encode ingredient lists and L2-normalize each vector in float32."""
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    ).astype(np.float32)
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def build_product_embeddings(
    output_dir: Path,
    *,
    data_path: Path = CLEANED_PRODUCTS_PATH,
) -> None:
    """Write the aligned product index and embeddings to a caller-selected directory."""
    output_dir = Path(output_dir)
    data_path = Path(data_path)
    chunk_dir = output_dir / "full_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    if data_path.suffix == ".parquet":
        products = pd.read_parquet(data_path)
    else:
        products = pd.read_csv(data_path)
    products = (
        products.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
        .drop_duplicates(subset=[TEXT_COLUMN])
        .reset_index(drop=True)
    )
    products[[ID_COLUMN, TEXT_COLUMN, LABEL_COLUMN]].to_parquet(
        output_dir / "full_index.parquet", index=False
    )
    print("rows:", len(products))
    print(products[LABEL_COLUMN].value_counts())

    model = load_embedding_model()
    texts = products[TEXT_COLUMN].astype(str).tolist()
    for start in range(0, len(texts), CHUNK_SIZE):
        chunk_path = chunk_dir / f"emb_{start:07d}.npy"
        if chunk_path.exists():
            print("skip", chunk_path)
            continue
        embeddings = encode_products(
            texts[start : start + CHUNK_SIZE], model, show_progress_bar=True
        )
        np.save(chunk_path, embeddings)
        print("saved", chunk_path, embeddings.shape)

    chunk_paths = sorted(chunk_dir.glob("*.npy"))
    embeddings = np.concatenate([np.load(path) for path in chunk_paths])
    assert len(embeddings) == len(products)
    np.save(output_dir / "emb_full.npy", embeddings)
    print("done", embeddings.shape)
