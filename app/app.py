"""Streamlit demo for the NutriLens inference pipeline."""

import csv
import logging
import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data/processed/product_catalog.csv"
GITHUB_URL = "https://github.com/utkarsh-mhw"
MAX_ALTERNATIVES = 5
logger = logging.getLogger(__name__)

# Streamlit puts the script directory first, where app.py can shadow the package.
if sys.path[0] != str(PROJECT_ROOT):
    sys.path.insert(0, str(PROJECT_ROOT))


@st.cache_data
def load_products() -> dict[str, dict[str, str]]:
    with CATALOG_PATH.open(encoding="utf-8", newline="") as catalog:
        return {row["product_name"]: row for row in csv.DictReader(catalog)}


def main() -> None:
    st.set_page_config(page_title="NutriLens")
    title, github = st.columns([4, 1], vertical_alignment="center")
    with title:
        st.title("NutriLens")
    with github:
        st.link_button("GitHub", GITHUB_URL)

    products = load_products()
    selected = st.selectbox("Product", sorted(products))

    if st.button("Analyze", type="primary", disabled=selected is None):
        st.session_state.pop("analysis_result", None)
        product = products[selected]
        try:
            with st.spinner("Analyzing product…"):
                from app.backend import run_pipeline

                result = run_pipeline(
                    product["code"], product["product_name"], product["ingredients_text"]
                )
        except Exception:
            logger.exception("NutriLens analysis failed")
            st.error("Analysis failed. Check the Streamlit terminal for details.")
        else:
            st.session_state["analysis_result"] = result

    result = st.session_state.get("analysis_result")
    if result is None or result["product_name"] != selected:
        return

    st.subheader("NOVA classification")
    st.write(f"NOVA {result['nova_group']}")

    st.subheader("Explanation")
    st.write(result["explanation"])

    st.subheader("Recommended alternatives")
    alternatives = result["recommendations"][:MAX_ALTERNATIVES]
    if alternatives:
        for alternative in alternatives:
            name = alternative["product_name"] or "Unnamed product"
            st.write(f"{name} · NOVA {alternative['nova_group']}")
    else:
        st.write("No alternatives found.")

    with st.expander("Retrieved FDA chunks"):
        if not result["retrieved_chunks"]:
            st.write("No excerpts found.")
        for chunk in result["retrieved_chunks"]:
            st.caption(f"{chunk['source']} · Page {chunk['page']} · {chunk['section']}")
            st.write(chunk["text"])


if __name__ == "__main__":
    main()
