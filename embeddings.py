"""
embeddings.py
─────────────
Gemini text-embedding-004 helper with:
  - Batching   : groups texts into chunks to reduce API round-trips
  - Backoff    : exponential retry on rate-limit / transient errors
  - Fail-Fast  : Raises exception on ultimate failure to prevent DB poisoning
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from google import genai
from config import GEMINI_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
EMBED_BATCH_SIZE   = 20        # texts per API call
MAX_RETRIES        = 6         # total attempts before giving up
INITIAL_BACKOFF    = 2.0       # seconds before first retry
BACKOFF_MULTIPLIER = 2.0       # multiplied each retry
MAX_BACKOFF        = 64.0      # cap on wait time
EMBED_DIM          = 768       # Gemini text-embedding-004 output dimension (PATCHED)

# ── Client ────────────────────────────────────────────────────────────────────
_client: Optional[genai.Client] = None

def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set. Check your config or .env file.")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client

# ── Core helpers ──────────────────────────────────────────────────────────────

def _embed_batch_with_retry(texts: list[str]) -> list[list[float]]:
    """
    Embed a single batch of texts with exponential backoff on failure.
    Raises RuntimeError if all retries are exhausted to prevent vector DB poisoning.
    """
    client = _get_client()
    wait = INITIAL_BACKOFF

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
            )
            return [list(e.values) for e in response.embeddings]

        except Exception as exc:
            err_str = str(exc).lower()
            is_rate_limit = any(k in err_str for k in (
                "quota", "rate", "429", "resource_exhausted", "too many"
            ))
            
            if attempt == MAX_RETRIES:
                logger.error("Embedding failed fatally after %d attempts: %s", attempt, exc)
                # PATCHED: We raise instead of returning zero-vectors. 
                # Let the ingestor catch this and decide whether to skip the file or halt.
                raise RuntimeError(f"Fatal embedding failure after {MAX_RETRIES} attempts. Last error: {exc}")

            if is_rate_limit:
                logger.warning("Rate limit on attempt %d/%d, waiting %.1fs...", attempt, MAX_RETRIES, wait)
            else:
                logger.warning("Embedding error on attempt %d/%d, waiting %.1fs: %s", attempt, MAX_RETRIES, wait, exc)
            
            time.sleep(wait)
            wait = min(wait * BACKOFF_MULTIPLIER, MAX_BACKOFF)

    raise RuntimeError("Unreachable")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Splits into batches of EMBED_BATCH_SIZE and retries each batch independently.
    """
    if not texts:
        return []

    results: list[list[float]] = []
    total = len(texts)
    
    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[start: start + EMBED_BATCH_SIZE]
        logger.info("Embedding batch %d-%d / %d", start + 1, min(start + EMBED_BATCH_SIZE, total), total)
        
        vectors = _embed_batch_with_retry(batch)
        results.extend(vectors)

    return results


def embed_single(text: str) -> list[float]:
    """Convenience wrapper for embedding a single string."""
    results = embed_texts([text])
    return results[0]