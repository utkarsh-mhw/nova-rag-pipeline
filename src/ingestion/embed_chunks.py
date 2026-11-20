"""Encode FDA chunks and build the in-memory indexes used for retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "rag_chunks.pkl"
EMBEDDINGS_PATH = PROCESSED_DIR / "rag_emb.npy"
MODEL_ID = "NeuML/pubmedbert-base-embeddings"
BATCH_SIZE = 64
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 100
HNSW_EF_SEARCH = 50


def load_chunk_embedding_model() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_ID)


def embed_chunks(output_dir: Path, *, chunks_path: Path = CHUNKS_PATH) -> np.ndarray:
    """Encode chunks in pickle row order and save rag_emb.npy."""
    chunks = pd.read_pickle(chunks_path)
    model = load_chunk_embedding_model()
    embeddings = model.encode(
        chunks["text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "rag_emb.npy", embeddings)
    return embeddings


def build_chunk_indexes(
    chunk_texts: list[str], embeddings: np.ndarray
) -> tuple[faiss.IndexHNSWFlat, BM25Okapi]:
    """Build FAISS and BM25 indexes with positions matching the chunk rows."""
    index = faiss.IndexHNSWFlat(
        embeddings.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT
    )
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    index.add(embeddings)

    bm25 = BM25Okapi([text.lower().split() for text in chunk_texts])
    print(f"FAISS: {index.ntotal} vectors, dim {embeddings.shape[1]}")
    print(f"BM25:  {len(chunk_texts)} documents")
    return index, bm25


def load_chunk_indexes(
    *,
    chunks_path: Path = CHUNKS_PATH,
    embeddings_path: Path = EMBEDDINGS_PATH,
) -> tuple[pd.DataFrame, faiss.IndexHNSWFlat, BM25Okapi]:
    """Build runtime indexes from the pre-built local chunk artifacts."""
    chunks = pd.read_pickle(chunks_path)
    embeddings = np.load(embeddings_path)
    index, bm25 = build_chunk_indexes(chunks["text"].tolist(), embeddings)
    return chunks, index, bm25
