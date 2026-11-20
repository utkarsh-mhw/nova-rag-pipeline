# NutriLens

Food processing classifier + RAG explainer. Predicts NOVA group from ingredients, retrieves FDA regulatory context, generates plain-language explanations, and recommends less-processed alternatives.

## Architecture

```
User input (product name / ingredients)
  → Classification: XGBoost on PubMedBERT embeddings → NOVA 1-4
  → Retrieval: Hybrid FAISS HNSW + BM25 + RRF over FDA 21 CFR chunks → top-5
  → Explanation: Llama 3.1 8B Instruct via HF Inference API → paragraph with citations
  → Recommendation: Per-NOVA FAISS + category tag similarity → alternatives
```

## Project Structure

```
data/raw/           → source files, never modified (us_nova_rows.csv, fda_pdfs/)
data/interim/       → intermediate cleaning outputs (only_english.csv, .parquet)
data/processed/     → model artifacts loaded at runtime (embeddings, indexes, model weights, chunks)
src/config.py       → single source of truth for all paths, model IDs, hyperparams
src/data_prep/      → raw → interim → processed pipeline
src/classification/  → product embedding + XGBoost prediction
src/ingestion/      → PDF chunking + chunk embedding + index building
src/retrieval/      → hybrid search (FAISS + BM25 + RRF)
src/explanation/    → prompt template + HF Inference API call
src/recommendation/ → alternative product recommender
app/                → Streamlit frontend + backend orchestrator
eval/               → retrieval and generation evaluation scripts
experiments/        → Colab notebooks (reference only, not imported by app)
archive/            → superseded artifacts (old embeddings, earlier models)
```

## Key Technical Details

- **Embedding model**: `NeuML/pubmedbert-base-embeddings` (768-d), L2-normalized
- **Classifier**: XGBoost trained on PubMedBERT embeddings, 279K products, acc 0.95, macro-F1 0.78
- **FAISS index**: HNSW (M=32, efConstruction=100, efSearch=50). MUST pass `faiss.METRIC_INNER_PRODUCT` in constructor — setting metric_type after construction silently breaks HNSW and returns -1 ids
- **BM25**: rank_bm25.BM25Okapi on whitespace-tokenized chunk texts
- **RRF fusion**: c=60, top_n=20 from each arm, return top-k (k=5)
- **LLM**: `meta-llama/Llama-3.1-8B-Instruct` via HF InferenceClient, temperature=0.2, frequency_penalty=0.6, max_tokens=250
- **HF_TOKEN**: loaded from environment variable, never hardcoded

## Coding Rules

- All magic numbers (paths, model IDs, FAISS params, RRF params, LLM settings) live in `src/config.py`. No hardcoded values in module files.
- Raw ingredient text goes to the LLM at inference time. The `ingredients_text` column in full_index.parquet is stemmed — do NOT feed it to the LLM.
- After every `index.search()` call, assert `(ids >= 0).all()` to catch silent FAISS failures.
- If the HF provider rejects `frequency_penalty`, drop it silently — the prompt handles repetition.
- If `finish_reason == 'length'`, trim the response to the last full sentence (rfind `.`).
- Explanations are single paragraphs, no bullets, no lists, no repeated sentences. This is enforced in the system prompt.
- The system prompt lives in `src/explanation/prompt.py` as a constant. Do not inline it elsewhere.

## Data Files (do not regenerate)

These are pre-built and loaded at runtime. Do not overwrite or regenerate unless explicitly asked:

| File | Location | What it is |
|------|----------|------------|
| `xgb_full.json` | `data/processed/` | Trained XGBoost model |
| `emb_full.npy` | `data/processed/` | Product embeddings (279K × 768) |
| `full_index.parquet` | `data/processed/` | code, ingredients_text, nova_group |
| `split_full.npz` | `data/processed/` | Train/test split indices |
| `rag_chunks.pkl` | `data/processed/` | FDA chunk DataFrame (chunk_id, source, page, section, text) |
| `rag_emb.npy` | `data/processed/` | Chunk embeddings for FAISS |

## Deployment

Target: HF Spaces or Streamlit Cloud. Entry point is `app/app.py`. All processed data files must be available at the paths specified in `src/config.py`.

## Testing

- Retrieval: `eval/retrieval_eval.py` — labeled additive queries, measures hit rate@k, MRR, recall
- Generation: `eval/generation_eval.py` — faithfulness scoring on sampled outputs
- Quick sanity check: run `retrieve("sodium nitrite")` and confirm §172.177 appears in top-5
