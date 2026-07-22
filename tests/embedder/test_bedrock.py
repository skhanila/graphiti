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

import json
from unittest.mock import MagicMock, patch

import pytest

from graphiti_core.embedder.bedrock import (
    DEFAULT_EMBEDDING_MODEL,
    MAX_INPUT_LENGTH,
    BedrockEmbedder,
    BedrockEmbedderConfig,
)
from tests.embedder.embedder_fixtures import create_embedding_values


def create_mock_bedrock_response(dimension: int = 1024, multiplier: float = 0.1) -> dict:
    """Create a mock Bedrock InvokeModel response (synchronous boto3 shape)."""
    embedding = create_embedding_values(multiplier, dimension)
    body_bytes = json.dumps({'embedding': embedding}).encode()

    mock_body = MagicMock()
    mock_body.read = MagicMock(return_value=body_bytes)

    return {'body': mock_body}


@pytest.fixture
def mock_bedrock_client():
    """Create a mocked boto3 bedrock-runtime client."""
    mock_client = MagicMock()
    mock_client.invoke_model = MagicMock(return_value=create_mock_bedrock_response())
    return mock_client


@pytest.fixture
def bedrock_embedder(mock_bedrock_client):
    """Create a BedrockEmbedder with a mocked boto3 client."""
    config = BedrockEmbedderConfig(
        embedding_dim=1024,
        aws_region='us-west-2',
        aws_profile=None,  # Avoid real AWS profile resolution in tests
    )
    # Patch boto3 so __init__ does not build a real client.
    with patch('graphiti_core.embedder.bedrock.boto3') as mock_boto3:
        mock_boto3.Session.return_value.client.return_value = mock_bedrock_client
        embedder = BedrockEmbedder(config=config)

    return embedder


class TestBedrockEmbedderConfig:
    """Tests for BedrockEmbedderConfig."""

    def test_default_config(self):
        config = BedrockEmbedderConfig()
        assert config.model_id == DEFAULT_EMBEDDING_MODEL
        assert config.aws_region == 'us-west-2'
        assert config.aws_profile is None
        assert config.embedding_dim == 1024

    def test_custom_config(self):
        config = BedrockEmbedderConfig(
            model_id='amazon.titan-embed-text-v1',
            aws_region='us-east-1',
            aws_profile='my-profile',
            embedding_dim=512,
        )
        assert config.model_id == 'amazon.titan-embed-text-v1'
        assert config.aws_region == 'us-east-1'
        assert config.aws_profile == 'my-profile'
        assert config.embedding_dim == 512


class TestBedrockEmbedderCreate:
    """Tests for BedrockEmbedder.create method."""

    @pytest.mark.asyncio
    async def test_create_with_string_input(self, bedrock_embedder, mock_bedrock_client):
        result = await bedrock_embedder.create('Test input')

        mock_bedrock_client.invoke_model.assert_called_once()
        call_kwargs = mock_bedrock_client.invoke_model.call_args[1]
        assert call_kwargs['modelId'] == DEFAULT_EMBEDDING_MODEL
        body = json.loads(call_kwargs['body'])
        assert body['inputText'] == 'Test input'
        assert body['dimensions'] == 1024
        assert body['normalize'] is True
        assert len(result) == 1024

    @pytest.mark.asyncio
    async def test_create_with_list_input(self, bedrock_embedder, mock_bedrock_client):
        result = await bedrock_embedder.create(['First text', 'Second text'])

        mock_bedrock_client.invoke_model.assert_called_once()
        call_kwargs = mock_bedrock_client.invoke_model.call_args[1]
        body = json.loads(call_kwargs['body'])
        assert body['inputText'] == 'First text'
        assert len(result) == 1024

    @pytest.mark.asyncio
    async def test_create_with_empty_string(self, bedrock_embedder, mock_bedrock_client):
        result = await bedrock_embedder.create('')

        mock_bedrock_client.invoke_model.assert_not_called()
        assert result == []

    @pytest.mark.asyncio
    async def test_create_with_whitespace_only(self, bedrock_embedder, mock_bedrock_client):
        result = await bedrock_embedder.create('   ')

        mock_bedrock_client.invoke_model.assert_not_called()
        assert result == []

    @pytest.mark.asyncio
    async def test_create_truncates_long_input(self, bedrock_embedder, mock_bedrock_client):
        long_text = 'x' * (MAX_INPUT_LENGTH + 1000)
        await bedrock_embedder.create(long_text)

        call_kwargs = mock_bedrock_client.invoke_model.call_args[1]
        body = json.loads(call_kwargs['body'])
        assert len(body['inputText']) == MAX_INPUT_LENGTH

    @pytest.mark.asyncio
    async def test_create_respects_embedding_dim(self, bedrock_embedder, mock_bedrock_client):
        # Return a larger embedding than configured dim
        large_embedding = [0.1] * 2048
        body_bytes = json.dumps({'embedding': large_embedding}).encode()
        mock_body = MagicMock()
        mock_body.read = MagicMock(return_value=body_bytes)
        mock_bedrock_client.invoke_model.return_value = {'body': mock_body}

        result = await bedrock_embedder.create('Test')

        assert len(result) == 1024  # Trimmed to config.embedding_dim


class TestBedrockEmbedderCreateBatch:
    """Tests for BedrockEmbedder.create_batch method."""

    @pytest.mark.asyncio
    async def test_create_batch_multiple_inputs(self, bedrock_embedder, mock_bedrock_client):
        inputs = ['Text 1', 'Text 2', 'Text 3']
        result = await bedrock_embedder.create_batch(inputs)

        assert mock_bedrock_client.invoke_model.call_count == 3
        assert len(result) == 3
        assert all(len(emb) == 1024 for emb in result)

    @pytest.mark.asyncio
    async def test_create_batch_empty_list(self, bedrock_embedder, mock_bedrock_client):
        result = await bedrock_embedder.create_batch([])

        mock_bedrock_client.invoke_model.assert_not_called()
        assert result == []


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
