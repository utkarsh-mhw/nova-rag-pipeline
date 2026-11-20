"""Expose NutriLens inference to Streamlit and the terminal."""

import argparse
import json
import sys
from dotenv import load_dotenv
load_dotenv()
from contextlib import redirect_stdout

from src.pipeline.run_inference import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one product with NutriLens.")
    parser.add_argument("--code", required=True)
    parser.add_argument("--product-name", required=True)
    parser.add_argument("--ingredients-text", required=True)
    args = parser.parse_args()

    with redirect_stdout(sys.stderr):
        result = run_pipeline(args.code, args.product_name, args.ingredients_text)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
