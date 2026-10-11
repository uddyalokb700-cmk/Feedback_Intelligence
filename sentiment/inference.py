"""
Inference engine for customer feedback sentiment classification.
Implements batched inference and persistent model caching per worker process.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from django.conf import settings

from sentiment.preprocessing import clean_text

logger = logging.getLogger("sentiment")

_MODEL_CACHE = {}


def get_artifacts_dir() -> Path:
    """Retrieve the designated path for ML model artifacts."""
    if hasattr(settings, "ML_MODELS_DIR"):
        return Path(settings.ML_MODELS_DIR)
    return Path(__file__).resolve().parent.parent / "ml_artifacts"


def load_model(version: str = None):
    """
    Load and cache the Scikit-learn sentiment pipeline for the process.
    Loads once per worker process to ensure low inference latency.
    """
    target_version = version or getattr(settings, "ACTIVE_MODEL_VERSION", "v1.0-tfidf-logreg")

    if target_version in _MODEL_CACHE:
        return _MODEL_CACHE[target_version]

    artifacts_dir = get_artifacts_dir()
    model_filename = f"sentiment_model_{target_version}.joblib"
    model_path = artifacts_dir / model_filename

    if not model_path.exists():
        logger.warning("Sentiment model %s not found on disk. Training baseline...", target_version)
        from sentiment.train import train_and_evaluate
        pipeline, _ = train_and_evaluate(artifacts_dir=artifacts_dir)
        _MODEL_CACHE[target_version] = pipeline
        return pipeline

    try:
        pipeline = joblib.load(model_path)
        _MODEL_CACHE[target_version] = pipeline
        logger.info("Loaded sentiment model pipeline: %s", target_version)
        return pipeline
    except Exception as e:
        logger.exception("Failed to load sentiment model from %s: %s", model_path, e)
        raise RuntimeError(f"Unable to load sentiment model {target_version}") from e


def get_model_metadata(version: str = None) -> dict:
    """Read the evaluation metrics and configuration metadata for the active model."""
    target_version = version or getattr(settings, "ACTIVE_MODEL_VERSION", "v1.0-tfidf-logreg")
    artifacts_dir = get_artifacts_dir()
    meta_path = artifacts_dir / f"model_metadata_{target_version}.json"

    if not meta_path.exists():
        # Ensure model is trained
        load_model(target_version)

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "model_version": target_version,
        "status": "pending_evaluation",
        "evaluation_metrics": None,
        "known_limitations": ["Evaluation metrics pending."],
    }


def predict_batch(texts: list[str], model_version: str = None) -> list[dict]:
    """
    Perform high-throughput batched sentiment classification.
    Returns structured predictions with label and calibrated confidence scores.
    """
    if not texts:
        return []

    pipeline = load_model(model_version)
    target_version = model_version or getattr(settings, "ACTIVE_MODEL_VERSION", "v1.0-tfidf-logreg")
    classes = list(pipeline.classes_)

    # Clean and sanitize input texts
    cleaned_texts = [clean_text(t) for t in texts]

    # Handle empty/whitespace reviews gracefully
    valid_indices = []
    valid_texts = []
    for idx, ct in enumerate(cleaned_texts):
        if ct:
            valid_indices.append(idx)
            valid_texts.append(ct)

    results = [None] * len(texts)

    if valid_texts:
        # Vectorize and predict batch
        predictions = pipeline.predict(valid_texts)
        probabilities = pipeline.predict_proba(valid_texts)

        for i, idx in enumerate(valid_indices):
            pred_label = str(predictions[i])
            probs = probabilities[i]
            prob_dict = {cls_name: float(probs[c_idx]) for c_idx, cls_name in enumerate(classes)}

            confidence = round(float(np.max(probs)), 4)

            results[idx] = {
                "label": pred_label,
                "confidence": confidence,
                "positive_prob": round(float(prob_dict.get("POSITIVE", 0.0)), 4),
                "neutral_prob": round(float(prob_dict.get("NEUTRAL", 0.0)), 4),
                "negative_prob": round(float(prob_dict.get("NEGATIVE", 0.0)), 4),
                "model_version": target_version,
            }

    # Fill default neutral result for empty or whitespace-only inputs
    for idx in range(len(texts)):
        if results[idx] is None:
            results[idx] = {
                "label": "NEUTRAL",
                "confidence": 0.3333,
                "positive_prob": 0.3333,
                "neutral_prob": 0.3334,
                "negative_prob": 0.3333,
                "model_version": target_version,
            }

    return results
