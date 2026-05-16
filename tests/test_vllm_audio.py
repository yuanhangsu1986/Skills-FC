# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import os
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from nemo_skills.inference.model.vllm import VLLMModel, audio_file_to_base64


# -----------------------
# Unit tests - no server required
# -----------------------

def test_audio_file_to_base64():
    """Test basic audio file encoding to base64."""
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.wav', delete=False) as f:
        test_content = b'RIFF' + b'\x00' * 100
        f.write(test_content)
        temp_path = f.name

    try:
        result = audio_file_to_base64(temp_path)
        assert isinstance(result, str)
        assert len(result) > 0
        decoded = base64.b64decode(result)
        assert decoded == test_content
    finally:
        os.unlink(temp_path)


@pytest.fixture
def vllm_model(tmp_path):
    """Create a VLLMModel instance for testing."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    model = VLLMModel(model="test-model", data_dir=str(tmp_path), base_url="http://localhost:5000")
    return model


def test_content_text_to_list_with_audio(vllm_model, tmp_path):
    """Test converting string content with audio to list format."""
    audio_path = tmp_path / "audio" / "test.wav"
    audio_path.parent.mkdir(exist_ok=True)
    with open(audio_path, 'wb') as f:
        f.write(b'RIFF' + b'\x00' * 100)

    message = {"role": "user", "content": "Describe this audio", "audio": {"path": "audio/test.wav"}}

    result = vllm_model.content_text_to_list(message)

    assert isinstance(result["content"], list)
    assert len(result["content"]) == 2
    assert result["content"][0]["type"] == "text"
    assert result["content"][1]["type"] == "audio_url"
    assert result["content"][1]["audio_url"]["url"].startswith("data:audio/wav;base64,")


def test_content_text_to_list_with_multiple_audios(vllm_model, tmp_path):
    """Test handling message with multiple audio files."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(exist_ok=True)

    for i in range(2):
        with open(audio_dir / f"test_{i}.wav", 'wb') as f:
            f.write(b'RIFF' + b'\x00' * 100)

    message = {
        "role": "user",
        "content": "Compare these",
        "audios": [{"path": "audio/test_0.wav"}, {"path": "audio/test_1.wav"}],
    }

    result = vllm_model.content_text_to_list(message)

    assert isinstance(result["content"], list)
    assert len(result["content"]) == 3
    assert result["content"][0]["type"] == "text"
    assert result["content"][1]["type"] == "audio_url"
    assert result["content"][2]["type"] == "audio_url"


# -----------------------
# Request building tests with audio
# -----------------------

def test_build_chat_request_with_audio(tmp_path, vllm_model):
    """Test that chat request params are correctly built with audio content."""
    # Create audio file
    audio_path = tmp_path / "audio" / "test.wav"
    audio_path.parent.mkdir(exist_ok=True)
    with open(audio_path, 'wb') as f:
        f.write(b'RIFF' + b'\x00' * 100)

    messages = [{"role": "user", "content": "Test audio", "audio": {"path": "audio/test.wav"}}]

    # Build request params - this doesn't make any network calls
    params = vllm_model._build_chat_request_params(messages=messages, stream=False, tokens_to_generate=10)

    # Validate request structure
    assert "messages" in params
    assert len(params["messages"]) == 1
    content_items = params["messages"][0]["content"]
    assert isinstance(content_items, list)
    assert len(content_items) == 2
    assert content_items[0]["type"] == "text"
    assert content_items[1]["type"] == "audio_url"

    # Verify base64 encoding is valid
    audio_url = content_items[1]["audio_url"]["url"]
    assert audio_url.startswith("data:audio/wav;base64,")
    audio_b64 = audio_url.split(",", 1)[1]
    decoded = base64.b64decode(audio_b64)
    assert decoded.startswith(b'RIFF')


@pytest.mark.asyncio
async def test_generate_with_audio_mocked_response(tmp_path, vllm_model):
    """Test generate_async with audio by mocking the response (no real server call)."""
    # Create audio file
    audio_path = tmp_path / "audio" / "test.wav"
    audio_path.parent.mkdir(exist_ok=True)
    with open(audio_path, 'wb') as f:
        f.write(b'RIFF' + b'\x00' * 100)

    messages = [{"role": "user", "content": "Describe this audio", "audio": {"path": "audio/test.wav"}}]

    # Mock the entire generate_async method - no actual API call made
    mock_response = {"generation": "This audio contains speech", "num_generated_tokens": 5}
    
    with patch.object(vllm_model, "generate_async", new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_response

        # Call the mocked method
        response = await vllm_model.generate_async(prompt=messages, tokens_to_generate=50, temperature=0.0)

        # Verify the mock was called correctly
        assert response["generation"] == "This audio contains speech"
        assert response["num_generated_tokens"] == 5
        mock_generate.assert_awaited_once()


