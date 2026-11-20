"""Generate a NOVA explanation through Hugging Face Inference."""

import os
from collections.abc import Iterable, Mapping
from functools import lru_cache

from huggingface_hub import InferenceClient

from .prompt import SYSTEM_PROMPT, build_user_message


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MAX_TOKENS = 250
TEMPERATURE = 0.2
FREQUENCY_PENALTY = 0.6


@lru_cache(maxsize=1)
def _load_client() -> InferenceClient:
    return InferenceClient(model=MODEL_ID, token=os.environ["HF_TOKEN"])


def explain(
    ingredients: str,
    nova_group: int,
    retrieved_chunks: Iterable[Mapping[str, object]],
) -> str:
    """Explain the classification using the supplied FDA excerpts."""
    user_message = build_user_message(ingredients, nova_group, retrieved_chunks)
    response = _load_client().chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        frequency_penalty=FREQUENCY_PENALTY,
    )
    choice = response.choices[0]
    text = choice.message.content.strip()
    if choice.finish_reason == "length" and "." in text:
        text = text[: text.rfind(".") + 1]
    return text
