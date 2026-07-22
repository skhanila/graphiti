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

import logging
import re
from typing import TYPE_CHECKING

from ..helpers import semaphore_gather
from ..llm_client import LLMConfig, RateLimitError
from .client import CrossEncoderClient

if TYPE_CHECKING:
    from anthropic import AsyncAnthropicBedrock
else:
    try:
        from anthropic import AsyncAnthropicBedrock
    except ImportError:
        raise ImportError(
            'anthropic is required for BedrockRerankerClient. '
            'Install it with: pip install graphiti-core[bedrock]'
        ) from None

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'us.anthropic.claude-sonnet-4-6'


class BedrockRerankerClient(CrossEncoderClient):
    """
    Reranker using Claude on Amazon Bedrock for per-passage relevance scoring.

    Each passage is scored individually on a 0-100 scale via a separate API
    call with bounded concurrency. Follows the same architecture as
    GeminiRerankerClient.

    Authenticates via AWS credential chain. Credentials refresh per-call.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        client: 'AsyncAnthropicBedrock | None' = None,
        aws_region: str = 'us-west-2',
        aws_profile: str | None = None,
    ):
        if config is None:
            config = LLMConfig(api_key='bedrock', model=DEFAULT_MODEL)

        self.config = config

        if client is not None:
            self.client = client
        else:
            kwargs: dict = {'aws_region': aws_region}
            if aws_profile:
                kwargs['aws_profile'] = aws_profile
            self.client = AsyncAnthropicBedrock(**kwargs)

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """
        Rank passages by relevance to query using per-passage scoring.

        Raises:
            RateLimitError: If Bedrock returns throttling errors.
            Exception: On unexpected failures (fail-loud, let caller handle).
        """
        if not passages:
            return []

        if len(passages) == 1:
            return [(passages[0], 1.0)]

        scoring_prompts = [
            f'Rate how well this passage answers or relates to the query. '
            f'Use a scale from 0 to 100.\n\n'
            f'Query: {query}\n\nPassage: {passage[:500]}\n\n'
            f'Provide only a number between 0 and 100 (no explanation, just the number):'
            for passage in passages
        ]

        try:
            responses = await semaphore_gather(
                *[
                    self.client.messages.create(
                        model=self.config.model or DEFAULT_MODEL,
                        max_tokens=5,
                        messages=[{'role': 'user', 'content': prompt}],
                    )
                    for prompt in scoring_prompts
                ]
            )

            results: list[tuple[str, float]] = []
            for passage, response in zip(passages, responses, strict=True):
                try:
                    score_text = response.content[0].text.strip()
                    score_match = re.search(r'\b(\d{1,3})\b', score_text)
                    if score_match:
                        score = float(score_match.group(1))
                        results.append((passage, max(0.0, min(1.0, score / 100.0))))
                    else:
                        logger.warning(f'Could not extract score from response: {score_text}')
                        results.append((passage, 0.0))
                except (ValueError, AttributeError, IndexError) as e:
                    logger.warning(f'Error parsing Bedrock reranker response: {e}')
                    results.append((passage, 0.0))

            results.sort(reverse=True, key=lambda x: x[1])
            return results

        except Exception as e:
            error_message = str(e).lower()
            if 'throttl' in error_message or 'rate' in error_message or '429' in str(e):
                raise RateLimitError from e
            logger.error(f'Error in Bedrock reranker: {e}')
            raise
