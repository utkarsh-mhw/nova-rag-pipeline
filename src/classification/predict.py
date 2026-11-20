"""Load the trained XGBoost classifier and predict a NOVA group."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_PATH = PROJECT_ROOT / "data" / "processed" / "xgb_full.json"


def load_classifier(model_path: Path = CLASSIFIER_PATH) -> XGBClassifier:
    """Load the pre-built classifier for CPU inference."""
    try:
        from xgboost import XGBClassifier
    except ModuleNotFoundError as error:
        if error.name != "xgboost":
            raise
        raise RuntimeError(
            "NOVA prediction from xgb_full.json requires xgboost. "
            "Install it in the active environment with: python -m pip install xgboost"
        ) from None

    classifier = XGBClassifier()
    classifier.load_model(str(model_path))
    classifier.set_params(device="cpu")
    return classifier


def predict_nova_group(embedding: np.ndarray, classifier: XGBClassifier) -> int:
    """Map the classifier's zero-based label to a NOVA group from 1 to 4."""
    label = classifier.predict(embedding.reshape(1, -1))[0]
    return int(label) + 1
