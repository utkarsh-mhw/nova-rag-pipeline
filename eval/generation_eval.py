"""Run the notebook's product-level explanation evaluation."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import sleep

import pandas as pd

from src.explanation.explain import explain
from src.retreival.retrieve import DEFAULT_TOP_K, retrieve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCT_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "full_index.parquet"
CLEANED_PRODUCTS_PATH = PROJECT_ROOT / "data" / "interim" / "only_english.parquet"
SAMPLE_SIZE = 50
RANDOM_STATE = 42
MIN_RRF_SCORE = 0.005
INGREDIENT_PREVIEW_LENGTH = 200
PROGRESS_INTERVAL = 10
REQUEST_PAUSE_SECONDS = 1
INSUFFICIENT_INFORMATION = (
    "Insufficient information found in FDA documents to explain this product's classification."
)


def evaluate_generation(
    *,
    output_path: Path | None = None,
    faithfulness_scorer: Callable[[str, Sequence[Mapping[str, object]]], float]
    | None = None,
) -> pd.DataFrame:
    """Evaluate 50 products and optionally score explanations against their excerpts."""
    products = pd.read_parquet(PRODUCT_INDEX_PATH)
    if "product_name" not in products.columns:
        names = pd.read_parquet(CLEANED_PRODUCTS_PATH, columns=["code", "product_name"])
        names["code"] = names["code"].astype(str)
        products["code"] = products["code"].astype(str)
        products = products.merge(names.drop_duplicates("code"), on="code", how="left")

    sample = products.sample(SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    results = []
    for index, row in sample.iterrows():
        ingredients = row["ingredients_text"]
        nova_group = int(row["nova_group"])
        retrieved = retrieve(ingredients, k=DEFAULT_TOP_K)
        top_rrf = retrieved[0]["rrf_score"] if retrieved else 0

        faithfulness_score = None
        if top_rrf < MIN_RRF_SCORE:
            explanation = INSUFFICIENT_INFORMATION
        else:
            try:
                explanation = explain(ingredients, nova_group, retrieved)
            except Exception as error:
                explanation = f"[ERROR] {error}"
            else:
                if faithfulness_scorer is not None:
                    faithfulness_score = faithfulness_scorer(explanation, retrieved)

        result = {
            "product": row["product_name"],
            "nova_group": nova_group,
            "ingredients": ingredients[:INGREDIENT_PREVIEW_LENGTH],
            "top_rrf_score": top_rrf,
            "n_chunks_retrieved": len(retrieved),
            "explanation": explanation,
        }
        if faithfulness_scorer is not None:
            result["faithfulness_score"] = faithfulness_score
        results.append(result)

        if (index + 1) % PROGRESS_INTERVAL == 0:
            print(f"{index + 1}/{SAMPLE_SIZE} done")
        sleep(REQUEST_PAUSE_SECONDS)

    evaluation = pd.DataFrame(results)
    if output_path is not None:
        evaluation.to_csv(output_path, index=False)

    explanations = evaluation["explanation"]
    print(f"\nDone. {len(evaluation)} products evaluated.")
    print(f"Skipped (low confidence): {explanations.str.startswith('Insufficient').sum()}")
    print(f"Errors: {explanations.str.startswith('[ERROR]').sum()}")
    print("\nNOVA distribution in sample:")
    print(evaluation["nova_group"].value_counts().sort_index())
    return evaluation
