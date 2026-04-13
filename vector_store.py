"""
vector_store.py
===============
Endee vector database integration.
Handles index creation, vector upsert, and similarity search.

Endee exposes a REST API on http://localhost:8080 by default.
We use the official `endee` Python SDK which wraps these calls.
"""

from typing import List, Optional
from endee import Endee, Precision


class VectorStore:
    """
    Thin wrapper around the Endee Python SDK for PDF chunk storage/retrieval.
    """

    def __init__(self, base_url: str = "http://localhost:8080", auth_token: str = ""):
        """
        Initialise the Endee client.

        Args:
            base_url:   URL of your running Endee instance.
            auth_token: Optional bearer token (leave blank for unauthenticated mode).
        """
        if auth_token and auth_token.strip():
            # Authenticated mode: pass token, then set the custom base URL
            self.client = Endee(auth_token.strip())
            self.client.set_base_url(f"{base_url.rstrip('/')}/api/v1")
        else:
            # Open mode (no authentication)
            self.client = Endee()
            self.client.set_base_url(f"{base_url.rstrip('/')}/api/v1")

    # ---- Index management --------------------------------------------------

    def create_or_reset_index(self, name: str, dimension: int) -> None:
        """
        Create a new Endee index. If an index with the same name already
        exists (from a previous session), delete it first so we start clean.

        Args:
            name:      Unique index name (alphanumeric + underscores, max 48 chars).
            dimension: Embedding dimensionality (e.g. 1536 for OpenAI small model).
        """
        # Try to delete any existing index with this name
        try:
            existing = self.client.get_index(name=name)
            if existing:
                self.delete_index(name)
        except Exception:
            pass  # Index didn't exist -- that's fine

        # Create a cosine-similarity index with INT8 quantisation
        # INT8 cuts memory usage while keeping strong recall quality
        self.client.create_index(
            name=name,
            dimension=dimension,
            space_type="cosine",   # cosine similarity is best for text embeddings
            precision=Precision.INT8,
        )

    def delete_index(self, name: str) -> None:
        """Delete an Endee index by name (best-effort)."""
        try:
            index = self.client.get_index(name=name)
            index.delete()
        except Exception:
            pass

    # ---- Upsert vectors ----------------------------------------------------

    def upsert_chunks(
        self,
        index_name: str,
        chunks: List[str],
        embeddings: List[List[float]],
    ) -> None:
        """
        Store text chunks and their embeddings in Endee.

        Each vector record contains:
          - id:     "chunk_N" (unique within the index)
          - vector: the float embedding list
          - meta:   {"text": ...} -- the raw chunk text returned at query time

        Endee accepts up to 1,000 vectors per upsert call, so we batch.

        Args:
            index_name: The target Endee index.
            chunks:     List of raw text strings.
            embeddings: Corresponding embedding vectors.
        """
        assert len(chunks) == len(embeddings), "Chunks and embeddings must have the same length."

        index = self.client.get_index(name=index_name)
        UPSERT_BATCH = 500  # well within the 1,000 limit

        for i in range(0, len(chunks), UPSERT_BATCH):
            batch_chunks = chunks[i: i + UPSERT_BATCH]
            batch_vectors = embeddings[i: i + UPSERT_BATCH]

            records = [
                {
                    "id": f"chunk_{i + j}",
                    "vector": vec,
                    "meta": {"text": text, "chunk_index": i + j},
                }
                for j, (text, vec) in enumerate(zip(batch_chunks, batch_vectors))
            ]

            index.upsert(records)

    # ---- Similarity search -------------------------------------------------

    def search(
        self,
        index_name: str,
        query_vector: List[float],
        top_k: int = 4,
    ) -> List[str]:
        """
        Perform a semantic similarity search in Endee and return the
        matching text chunks.

        Args:
            index_name:   The Endee index to search.
            query_vector: Embedding of the user's question.
            top_k:        Number of most-similar chunks to retrieve.

        Returns:
            A list of raw text strings (the retrieved chunks), ordered by similarity.
        """
        index = self.client.get_index(name=index_name)

        results = index.query(
            vector=query_vector,
            top_k=top_k,
            ef=128,                # higher ef = better recall at slight latency cost
            include_vectors=False,  # we only need the metadata, not the raw vectors
        )

        # Extract the stored text from the `meta` field of each result
        chunks = []
        for item in results:
            meta = item.get("meta", {})
            text = meta.get("text", "")
            if text:
                chunks.append(text)

        return chunks
