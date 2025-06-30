import asyncio
import os
import weakref
from functools import wraps
from typing import Any, Dict, List, Optional

import numpy as np
from aiolimiter import AsyncLimiter
from openai import APIConnectionError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def get_or_create_limiter(instance, limiter_attr, max_rate, time_period):
    """Get existing limiter or create new one for current event loop"""
    loop_attr = f"{limiter_attr}_loop"

    # Get current event loop
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create a new limiter
        limiter = AsyncLimiter(max_rate=max_rate, time_period=time_period)
        setattr(instance, limiter_attr, limiter)
        setattr(instance, loop_attr, None)
        return limiter

    # Check if we have a limiter and if it's for the current loop
    if (
        hasattr(instance, limiter_attr)
        and hasattr(instance, loop_attr)
        and getattr(instance, limiter_attr) is not None
    ):

        stored_loop_ref = getattr(instance, loop_attr)
        # If stored loop reference exists and is the same as current loop
        if stored_loop_ref is not None and stored_loop_ref() is current_loop:
            return getattr(instance, limiter_attr)

    # Create new limiter for current loop
    limiter = AsyncLimiter(max_rate=max_rate, time_period=time_period)
    setattr(instance, limiter_attr, limiter)
    # Store weak reference to current loop to avoid circular references
    setattr(instance, loop_attr, weakref.ref(current_loop))
    return limiter


def rate_limited(max_rate: int, time_period: int, limiter_attr: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            limiter = get_or_create_limiter(self, limiter_attr, max_rate, time_period)
            async with limiter:
                return await func(self, *args, **kwargs)

        return wrapper

    return decorator


class ChatConfig:
    """Configuration for OpenAI API"""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self._validate()

    def _validate(self):
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        if not self.base_url:
            raise ValueError("OPENAI_BASE_URL environment variable is not set")


class EmbeddingConfig:
    """Configuration for OpenAI Embedding API"""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_EMBEDDING_API_KEY")
        self.base_url = os.getenv("OPENAI_EMBEDDING_BASE_URL")
        self._validate()

    def _validate(self):
        if not self.api_key:
            raise ValueError("OPENAI_EMBEDDING_API_KEY environment variable is not set")
        if not self.base_url:
            raise ValueError(
                "OPENAI_EMBEDDING_BASE_URL environment variable is not set"
            )


class ChatClient:
    """Singleton wrapper for OpenAI async client for Chat"""

    _instance: Optional["ChatClient"] = None
    _client: Optional[AsyncOpenAI] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            config = ChatConfig()
            self._client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client


class EmbeddingClient:
    """Singleton wrapper for OpenAI async client for Embedding"""

    _instance: Optional["EmbeddingClient"] = None
    _client: Optional[AsyncOpenAI] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            config = EmbeddingConfig()
            self._client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client


# Retry decorator for API calls
api_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)


class ChatCompletion:
    """Singleton handler for OpenAI chat completions"""

    _instance: Optional["ChatCompletion"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only set attributes on first initialization
        if not hasattr(self, "_initialized"):
            self.client = ChatClient().client
            self._completion_limiter = None
            self._initialized = True

    @rate_limited(max_rate=4, time_period=1, limiter_attr="_completion_limiter")
    @api_retry
    async def complete(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Complete a chat prompt using the specified model.

        Args:
            model: The model identifier (e.g., "gpt-4o", "gpt-3.5-turbo")
            prompt: The user prompt
            system_prompt: Optional system prompt
            history_messages: Optional conversation history
            **kwargs: Additional parameters for the API call

        Returns:
            The completion response as a string
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history_messages:
            messages.extend(history_messages)

        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )

        return response.choices[0].message.content


class EmbeddingService:
    """Singleton handler for OpenAI embeddings"""

    _instance: Optional["EmbeddingService"] = None

    def __new__(cls, default_batch_size: int = 1000):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, default_batch_size: int = 1000):
        # Only set attributes on first initialization
        if not hasattr(self, "_initialized"):
            self.client = EmbeddingClient().client
            self._embedding_limiter = None
            self.default_batch_size = default_batch_size
            self._initialized = True
        else:
            # If already initialized, optionally update batch size
            if hasattr(self, "default_batch_size"):
                # Keep existing batch size, don't override
                pass
            else:
                # Fallback if somehow batch size wasn't set
                self.default_batch_size = default_batch_size

    @rate_limited(max_rate=4, time_period=1, limiter_attr="_embedding_limiter")
    @api_retry
    async def _create_embeddings_batch(
        self, texts: List[str], model: str = "text-embedding-3-small"
    ) -> np.ndarray:
        """
        Create embeddings for a single batch of texts (internal method).
        """
        response = await self.client.embeddings.create(
            model=model, input=texts, encoding_format="float"
        )
        return np.array([dp.embedding for dp in response.data])

    async def create_embeddings(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small",
        batch_size: Optional[int] = None,  # Use default_batch_size if not specified
    ) -> np.ndarray:
        """
        Create embeddings for the given texts with automatic batching for large inputs.

        Args:
            texts: List of texts to embed
            model: The embedding model to use
            batch_size: Maximum number of texts to process in a single API call (uses default if None)

        Returns:
            Numpy array of embeddings
        """
        if not texts:
            raise ValueError("texts list cannot be empty")

        # Use default batch size if not specified
        if batch_size is None:
            batch_size = self.default_batch_size

        # Validate and clean texts
        valid_texts = []
        for i, text in enumerate(texts):
            if text is None:
                raise ValueError(f"Text at index {i} is None")
            if not isinstance(text, str):
                raise ValueError(f"Text at index {i} is not a string: {type(text)}")

            cleaned_text = text.strip()
            if not cleaned_text:
                raise ValueError(
                    f"Text at index {i} is empty after stripping whitespace"
                )

            valid_texts.append(cleaned_text)

        # If batch size is small enough, process directly
        if len(valid_texts) <= batch_size:
            return await self._create_embeddings_batch(valid_texts, model)

        # Process in batches for large inputs with adaptive batch sizing
        import logging

        logger = logging.getLogger("HiRAG")
        logger.info(
            f"🔄 Processing {len(valid_texts)} texts in batches of {batch_size}"
        )

        all_embeddings = []
        current_batch_size = batch_size
        i = 0

        while i < len(valid_texts):
            batch_texts = valid_texts[i : i + current_batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(valid_texts) + batch_size - 1) // batch_size

            logger.info(
                f"📦 Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts, batch_size={current_batch_size})"
            )

            try:
                batch_embeddings = await self._create_embeddings_batch(
                    batch_texts, model
                )
                all_embeddings.append(batch_embeddings)

                logger.debug(f"✅ Batch completed successfully")
                i += current_batch_size

                # Reset batch size to original after successful batch
                if current_batch_size < batch_size:
                    current_batch_size = min(batch_size, current_batch_size * 2)
                    logger.info(
                        f"📈 Increasing batch size back to {current_batch_size}"
                    )

            except Exception as e:
                error_msg = str(e).lower()

                # Check if error is related to input size/limits
                if any(
                    keyword in error_msg
                    for keyword in [
                        "invalid_request_error",
                        "too large",
                        "limit exceeded",
                        "input invalid",
                        "request too large",
                    ]
                ):

                    if current_batch_size > 1:
                        # Reduce batch size and retry
                        new_batch_size = max(1, current_batch_size // 2)
                        logger.warning(
                            f"⚠️ API limit error, reducing batch size from {current_batch_size} to {new_batch_size}"
                        )
                        logger.warning(f"⚠️ Error details: {e}")
                        current_batch_size = new_batch_size
                        continue  # Retry with smaller batch
                    else:
                        # Even single text fails, this is a different issue
                        logger.error(
                            f"❌ Even single text embedding failed, this may be a content issue"
                        )
                        logger.error(
                            f"❌ Failed text preview: {batch_texts[0][:200]}..."
                        )
                        raise e
                else:
                    # Different type of error, don't retry
                    logger.error(
                        f"❌ Non-batch-size related error in batch processing: {e}"
                    )
                    raise e

        # Concatenate all embeddings
        result = np.concatenate(all_embeddings, axis=0)
        logger.info(f"✅ All {len(valid_texts)} embeddings processed successfully")

        return result
