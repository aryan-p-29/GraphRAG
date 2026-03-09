"""
embeddings.py
─────────────
Gemini text-embedding-004 helper with:
  - Batching   : groups texts into chunks to reduce API round-trips
  - Backoff    : exponential retry on rate-limit / transient errors
  - Graceful   : returns zero-vector on repeated failures rather than crashing
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from google import genai
from google.genai import types as genai_types

from config import GEMINI_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
EMBED_BATCH_SIZE   = 20        # texts per API call (API max is 100, keep low for free tier)
MAX_RETRIES        = 6         # total attempts before giving up
INITIAL_BACKOFF    = 2.0       # seconds before first retry
BACKOFF_MULTIPLIER = 2.0       # multiplied each retry
MAX_BACKOFF        = 64.0      # cap on wait time
EMBED_DIM          = 3072       # gemini-embedding-001 output dimension

# ── Client ────────────────────────────────────────────────────────────────────
_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set. Check your .env file.")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


# ── Core helpers ──────────────────────────────────────────────────────────────

def _embed_batch_with_retry(texts: list[str]) -> list[list[float]]:
    """
    Embed a single batch of texts with exponential backoff on failure.
    Returns a list of float vectors, same length as `texts`.
    """
    client = _get_client()
    wait = INITIAL_BACKOFF

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
            )
            # response.embeddings is a list of ContentEmbedding objects
            return [list(e.values) for e in response.embeddings]

        except Exception as exc:
            err_str = str(exc).lower()
            is_rate_limit = any(k in err_str for k in (
                "quota", "rate", "429", "resource_exhausted", "too many"
            ))
            if attempt == MAX_RETRIES:
                logger.error(
                    "Embedding failed after %d attempts: %s — returning zero vectors",
                    attempt, exc,
                )
                return [[0.0] * EMBED_DIM for _ in texts]

            if is_rate_limit:
                logger.warning(
                    "Rate limit on attempt %d/%d, waiting %.1fs: %s",
                    attempt, MAX_RETRIES, wait, exc,
                )
            else:
                logger.warning(
                    "Embedding error on attempt %d/%d, waiting %.1fs: %s",
                    attempt, MAX_RETRIES, wait, exc,
                )
            time.sleep(wait)
            wait = min(wait * BACKOFF_MULTIPLIER, MAX_BACKOFF)

    return [[0.0] * EMBED_DIM for _ in texts]  # unreachable, but satisfies type checker


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Public API — embed any number of texts.
    Splits into batches of EMBED_BATCH_SIZE and retries each batch independently.

    Parameters
    ----------
    texts : list of strings to embed.

    Returns
    -------
    list of float lists, same length as `texts`.
    """
    if not texts:
        return []

    results: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[start: start + EMBED_BATCH_SIZE]
        logger.debug("Embedding batch %d-%d / %d", start + 1, start + len(batch), total)
        vectors = _embed_batch_with_retry(batch)
        results.extend(vectors)

    return results


def embed_single(text: str) -> list[float]:
    """Convenience wrapper for embedding a single string."""
    results = embed_texts([text])
    return results[0] if results else [0.0] * EMBED_DIM
