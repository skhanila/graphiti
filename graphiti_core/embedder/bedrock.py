"""
Copyright Amazon.com Inc. or its affiliates.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import json
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import boto3
else:
    try:
        import boto3
    except ImportError:
        raise ImportError(
            'boto3 is required for BedrockEmbedder. '
            'Install it with: pip install graphiti-core[bedrock]'
        ) from None

from pydantic import Field

from ..helpers import semaphore_gather
from .client import EmbedderClient, EmbedderConfig

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = 'amazon.titan-embed-text-v2:0'
MAX_INPUT_LENGTH = 8000  # Titan Embed V2 character limit


class BedrockEmbedderConfig(EmbedderConfig):
    """Configuration for the Amazon Bedrock Titan Embeddings provider."""

    model_id: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    aws_region: str = Field(default='us-west-2')
    aws_profile: str | None = Field(default=None)


class BedrockEmbedder(EmbedderClient):
    """
    Amazon Bedrock Embedder using Titan Embeddings V2.

    Authenticates via AWS credential chain (profile, env vars, or IAM role).
    All embedding computation stays within your AWS account.

    The underlying boto3 client is synchronous; calls are dispatched to a
    thread via asyncio.to_thread so the event loop is never blocked. boto3
    low-level clients are thread-safe for invocation, so a single client is
    shared across concurrent calls.
    """

    def __init__(self, config: BedrockEmbedderConfig | None = None):
        if config is None:
            config = BedrockEmbedderConfig()
        self.config = config

        session_kwargs: dict[str, str] = {}
        if config.aws_profile:
            session_kwargs['profile_name'] = config.aws_profile

        session = boto3.Session(**session_kwargs)
        self._client = session.client('bedrock-runtime', region_name=config.aws_region)

    def _invoke_sync(self, text: str) -> list[float]:
        """Synchronous Bedrock InvokeModel call (runs in a worker thread)."""
        body = json.dumps(
            {
                'inputText': text[:MAX_INPUT_LENGTH],
                'dimensions': self.config.embedding_dim,
                'normalize': True,
            }
        )

        response = self._client.invoke_model(
            modelId=self.config.model_id,
            contentType='application/json',
            accept='application/json',
            body=body,
        )
        result = json.loads(response['body'].read())
        return result['embedding'][: self.config.embedding_dim]

    async def _embed_single(self, text: str) -> list[float]:
        """Embed a single text string, offloading the blocking call to a thread."""
        return await asyncio.to_thread(self._invoke_sync, text)

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """Create embedding for a single input. Returns one embedding vector."""
        if isinstance(input_data, str):
            text = input_data
        elif isinstance(input_data, list) and input_data and isinstance(input_data[0], str):
            text = input_data[0]
        else:
            text = str(input_data)

        if not text or not text.strip():
            return []

        return await self._embed_single(text)

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """
        Create embeddings for multiple inputs with bounded concurrency.

        Titan Embed V2 does not support native batch — each text requires
        a separate InvokeModel call. Uses semaphore_gather to respect
        Bedrock throttling limits.
        """
        if not input_data_list:
            return []

        return await semaphore_gather(*[self._embed_single(text) for text in input_data_list])
