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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.cross_encoder.bedrock_reranker_client import BedrockRerankerClient
from graphiti_core.llm_client import LLMConfig, RateLimitError


def create_mock_response(score_text: str) -> MagicMock:
    """Create a mock Bedrock messages.create response."""
    mock_content = MagicMock()
    mock_content.text = score_text

    mock_response = MagicMock()
    mock_response.content = [mock_content]
    return mock_response


@pytest.fixture
def mock_bedrock_anthropic_client():
    """Create a mocked AsyncAnthropicBedrock client."""
    mock_client = MagicMock()
    mock_client.messages = MagicMock()
    mock_client.messages.create = AsyncMock()
    return mock_client


@pytest.fixture
def bedrock_reranker(mock_bedrock_anthropic_client):
    """Create a BedrockRerankerClient with a mocked client."""
    config = LLMConfig(api_key='bedrock', model='us.anthropic.claude-sonnet-4-6')
    client = BedrockRerankerClient(config=config, client=mock_bedrock_anthropic_client)
    return client


class TestBedrockRerankerClientInit:
    """Tests for BedrockRerankerClient initialization."""

    def test_init_with_config_and_client(self):
        mock_client = MagicMock()
        config = LLMConfig(api_key='bedrock', model='test-model')
        reranker = BedrockRerankerClient(config=config, client=mock_client)
        assert reranker.config == config
        assert reranker.client == mock_client

    @patch('graphiti_core.cross_encoder.bedrock_reranker_client.AsyncAnthropicBedrock')
    def test_init_creates_client_with_region(self, mock_cls):
        BedrockRerankerClient(aws_region='eu-west-1')
        mock_cls.assert_called_once_with(aws_region='eu-west-1')

    @patch('graphiti_core.cross_encoder.bedrock_reranker_client.AsyncAnthropicBedrock')
    def test_init_creates_client_with_profile(self, mock_cls):
        BedrockRerankerClient(aws_region='us-east-1', aws_profile='my-profile')
        mock_cls.assert_called_once_with(aws_region='us-east-1', aws_profile='my-profile')


class TestBedrockRerankerClientRank:
    """Tests for BedrockRerankerClient.rank method."""

    @pytest.mark.asyncio
    async def test_rank_basic(self, bedrock_reranker, mock_bedrock_anthropic_client):
        mock_bedrock_anthropic_client.messages.create.side_effect = [
            create_mock_response('85'),
            create_mock_response('30'),
            create_mock_response('60'),
        ]

        result = await bedrock_reranker.rank(
            'capital of France',
            ['Paris is the capital.', 'Berlin is in Germany.', 'France is in Europe.'],
        )

        assert len(result) == 3
        assert result[0][1] == 0.85
        assert result[1][1] == 0.60
        assert result[2][1] == 0.30
        # Sorted descending
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_rank_empty_passages(self, bedrock_reranker):
        result = await bedrock_reranker.rank('query', [])
        assert result == []

    @pytest.mark.asyncio
    async def test_rank_single_passage(self, bedrock_reranker, mock_bedrock_anthropic_client):
        result = await bedrock_reranker.rank('query', ['single passage'])
        assert result == [('single passage', 1.0)]
        mock_bedrock_anthropic_client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_rank_invalid_score_returns_zero(
        self, bedrock_reranker, mock_bedrock_anthropic_client
    ):
        mock_bedrock_anthropic_client.messages.create.side_effect = [
            create_mock_response('not a number'),
            create_mock_response('75'),
        ]

        result = await bedrock_reranker.rank('query', ['Passage A', 'Passage B'])

        scores = {p: s for p, s in result}
        assert scores['Passage A'] == 0.0
        assert scores['Passage B'] == 0.75

    @pytest.mark.asyncio
    async def test_rank_score_clamped(self, bedrock_reranker, mock_bedrock_anthropic_client):
        mock_bedrock_anthropic_client.messages.create.side_effect = [
            create_mock_response('150'),  # >100, should clamp to 1.0
            create_mock_response('50'),
        ]

        result = await bedrock_reranker.rank('query', ['A', 'B'])

        scores = {p: s for p, s in result}
        assert scores['A'] == 1.0
        assert scores['B'] == 0.5

    @pytest.mark.asyncio
    async def test_rank_throttle_raises_rate_limit_error(
        self, bedrock_reranker, mock_bedrock_anthropic_client
    ):
        mock_bedrock_anthropic_client.messages.create.side_effect = Exception(
            'ThrottlingException: Rate exceeded'
        )

        with pytest.raises(RateLimitError):
            await bedrock_reranker.rank('query', ['A', 'B'])

    @pytest.mark.asyncio
    async def test_rank_429_raises_rate_limit_error(
        self, bedrock_reranker, mock_bedrock_anthropic_client
    ):
        mock_bedrock_anthropic_client.messages.create.side_effect = Exception(
            'HTTP 429 Too Many Requests'
        )

        with pytest.raises(RateLimitError):
            await bedrock_reranker.rank('query', ['A', 'B'])

    @pytest.mark.asyncio
    async def test_rank_generic_error_raises(self, bedrock_reranker, mock_bedrock_anthropic_client):
        mock_bedrock_anthropic_client.messages.create.side_effect = Exception(
            'Internal server error'
        )

        with pytest.raises(Exception, match='Internal server error'):
            await bedrock_reranker.rank('query', ['A', 'B'])

    @pytest.mark.asyncio
    async def test_rank_concurrent_calls(self, bedrock_reranker, mock_bedrock_anthropic_client):
        mock_bedrock_anthropic_client.messages.create.side_effect = [
            create_mock_response('80'),
            create_mock_response('60'),
            create_mock_response('40'),
        ]

        await bedrock_reranker.rank('query', ['P1', 'P2', 'P3'])

        assert mock_bedrock_anthropic_client.messages.create.call_count == 3


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
