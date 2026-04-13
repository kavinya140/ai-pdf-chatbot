"""
embeddings.py
=============
FREE embeddings using HuggingFace sentence-transformers (local model).
No API key needed! Uses all-MiniLM-L6-v2 (384 dimensions, ~90MB download once).
"""

from sentence_transformers import SentenceTransformer
from typing import List

_model = None  # cached model instance

def _get_model():
    """Load the model once and reuse it."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

EMBEDDING_DIM = 384  # output dimension of all-MiniLM-L6-v2


def get_embedding(text: str, api_key: str = "") -> List[float]:
    """Generate a single embedding vector. api_key ignored (local model)."""
    model = _get_model()
    cleaned = text.strip().replace("\n", " ")
    if not cleaned:
        raise ValueError("Cannot embed an empty string.")
    return model.encode(cleaned, normalize_embeddings=True).tolist()


def get_embeddings_batch(texts: List[str], api_key: str = "") -> List[List[float]]:
    """Generate embeddings for a list of chunks in one fast batch."""
    if not texts:
        return []
    model = _get_model()
    cleaned = [t.strip().replace("\n", " ") for t in texts if t.strip()]
    embeddings = model.encode(cleaned, normalize_embeddings=True, show_progress_bar=False)
    return [e.tolist() for e in embeddings]
