"""
Embedding service for generating vector embeddings

Implements chunking with overlap for long texts - the standard approach in RAG systems.
Reference: https://www.pinecone.io/learn/chunking-strategies/
"""
import logging
from typing import Optional

from openai import AsyncAzureOpenAI, AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from sociomemory.config import get_settings

logger = logging.getLogger(__name__)

# Chunking configuration for text-embedding-3-large (8192 token limit)
# Using ~7000 tokens per chunk for safety margin
MAX_CHUNK_CHARS = 24500  # ~7000 tokens at 3.5 chars/token
OVERLAP_CHARS = 2500     # ~10% overlap (~700 tokens) for context preservation


def chunk_text_with_overlap(text: str, chunk_size: int = MAX_CHUNK_CHARS,
                            overlap: int = OVERLAP_CHARS) -> list[str]:
    """
    Split text into overlapping chunks for embedding.

    Standard RAG chunking approach:
    - Each chunk is within token limits
    - Overlap preserves context at boundaries
    - No information is lost

    Args:
        text: The text to chunk
        chunk_size: Maximum characters per chunk (~7000 tokens)
        overlap: Overlap between chunks (~10-15% for context)

    Returns:
        List of text chunks with overlap
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        # Get chunk end position
        end = start + chunk_size

        # If not at the end, try to break at a natural boundary
        if end < len(text):
            # Look for paragraph break first
            para_break = text.rfind('\n\n', start + chunk_size - overlap, end)
            if para_break > start:
                end = para_break
            else:
                # Look for sentence break
                for sep in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    sent_break = text.rfind(sep, start + chunk_size - overlap, end)
                    if sent_break > start:
                        end = sent_break + len(sep)
                        break

        # Extract chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position, accounting for overlap
        start = end - overlap if end < len(text) else end

    return chunks


def mean_pool_embeddings(embeddings: list[list[float]]) -> list[float]:
    """
    Mean pool multiple embeddings into a single embedding.

    Standard approach for combining chunk embeddings.
    Reference: https://arxiv.org/abs/2212.03533 (Late Chunking)

    Args:
        embeddings: List of embedding vectors

    Returns:
        Single averaged embedding vector
    """
    if not embeddings:
        raise ValueError("No embeddings to pool")

    if len(embeddings) == 1:
        return embeddings[0]

    # Mean pooling across all embeddings
    dims = len(embeddings[0])
    pooled = [0.0] * dims

    for emb in embeddings:
        for i, val in enumerate(emb):
            pooled[i] += val

    n = len(embeddings)
    return [v / n for v in pooled]


class EmbeddingService:
    """Service for generating text embeddings using OpenAI or Azure OpenAI"""

    def __init__(self):
        settings = get_settings()
        self.dimensions = settings.embedding_dimensions
        self.model = settings.embedding_model

        if settings.use_azure_openai:
            # Azure OpenAI configuration
            self.client = AsyncAzureOpenAI(
                api_key=settings.azure_openai_key,
                api_version="2024-02-01",
                azure_endpoint=settings.azure_openai_endpoint,
            )
            self.deployment = settings.azure_openai_embedding_deployment
            self.is_azure = True
            logger.info(f"EmbeddingService initialized with Azure OpenAI: {settings.azure_openai_endpoint}")
            logger.info(f"Using deployment: {self.deployment}")
        else:
            # Standard OpenAI
            self.client = AsyncOpenAI(api_key=settings.openai_api_key)
            self.deployment = None
            self.is_azure = False
            logger.info(f"EmbeddingService initialized with OpenAI: {self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    async def get_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for text using chunking with overlap for long texts.

        For texts exceeding token limits:
        1. Split into overlapping chunks (~10% overlap)
        2. Embed each chunk separately
        3. Mean pool embeddings into single vector

        This is the standard RAG approach - no information is lost.

        Args:
            text: Text to embed (any length - will be chunked if needed)

        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        # Chunk text if it exceeds token limits (with 10% overlap)
        chunks = chunk_text_with_overlap(text)

        if len(chunks) > 1:
            logger.info(f"Chunking text ({len(text)} chars) into {len(chunks)} overlapping chunks")

        try:
            if len(chunks) == 1:
                # Single chunk - direct embedding
                if self.is_azure:
                    response = await self.client.embeddings.create(
                        input=chunks[0],
                        model=self.deployment,
                    )
                else:
                    response = await self.client.embeddings.create(
                        input=chunks[0],
                        model=self.model,
                    )

                embedding = response.data[0].embedding
                tokens_used = response.usage.total_tokens
                logger.debug(f"Generated embedding: {len(embedding)} dims, {tokens_used} tokens")
                return embedding

            else:
                # Multiple chunks - batch embed and mean pool
                if self.is_azure:
                    response = await self.client.embeddings.create(
                        input=chunks,
                        model=self.deployment,
                    )
                else:
                    response = await self.client.embeddings.create(
                        input=chunks,
                        model=self.model,
                    )

                # Get embeddings in order
                chunk_embeddings = [
                    item.embedding
                    for item in sorted(response.data, key=lambda x: x.index)
                ]
                tokens_used = response.usage.total_tokens

                # Mean pool all chunk embeddings
                embedding = mean_pool_embeddings(chunk_embeddings)

                logger.debug(
                    f"Generated pooled embedding from {len(chunks)} chunks: "
                    f"{len(embedding)} dims, {tokens_used} total tokens"
                )
                return embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    async def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batch.

        Uses chunking with overlap for long texts, consistent with get_embedding().

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (one per input text)
        """
        if not texts:
            return []

        # Process each text - some may need chunking
        embeddings = []
        for text in texts:
            if text and text.strip():
                # Use get_embedding which handles chunking
                emb = await self.get_embedding(text)
                embeddings.append(emb)

        return embeddings

    async def get_chunk_embeddings(self, text: str) -> list[tuple[str, list[float]]]:
        """
        Get separate embeddings for each chunk of a long text.

        Use this when you want to store chunks as separate memories
        for more granular retrieval (recommended for very long documents).

        Args:
            text: The text to chunk and embed

        Returns:
            List of (chunk_text, embedding) tuples
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        chunks = chunk_text_with_overlap(text)

        if len(chunks) == 1:
            # Single chunk
            emb = await self.get_embedding(chunks[0])
            return [(chunks[0], emb)]

        logger.info(f"Generating {len(chunks)} separate chunk embeddings for {len(text)} char text")

        try:
            if self.is_azure:
                response = await self.client.embeddings.create(
                    input=chunks,
                    model=self.deployment,
                )
            else:
                response = await self.client.embeddings.create(
                    input=chunks,
                    model=self.model,
                )

            # Pair chunks with embeddings in order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            result = [(chunks[i], sorted_data[i].embedding) for i in range(len(chunks))]

            logger.debug(f"Generated {len(result)} chunk embeddings, {response.usage.total_tokens} tokens")
            return result

        except Exception as e:
            logger.error(f"Chunk embedding generation failed: {e}")
            raise


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service singleton"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
