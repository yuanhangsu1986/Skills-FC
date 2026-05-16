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

"""
Backend implementations for the Unified NeMo Inference Server.

Available backends:
- salm: Speech-Augmented Language Model (text output from text/audio input)
- tts: Text-to-Speech using MagpieTTS (audio output from text input)
- s2s: Speech-to-Speech using DuplexS2S offline (text output from audio input)
- s2s_incremental: Speech-to-Speech using NemotronVoiceChat incremental (text+audio from audio)
- s2s_session: Speech-to-Speech with session support for multi-turn conversations
"""

from .base import BackendConfig, GenerationRequest, GenerationResult, InferenceBackend, Modality

__all__ = [
    "InferenceBackend",
    "GenerationRequest",
    "GenerationResult",
    "BackendConfig",
    "Modality",
    "get_backend",
    "list_backends",
]

# Registry of available backends
BACKEND_REGISTRY = {
    "salm": ("salm_backend", "SALMBackend"),
    "tts": ("tts_backend", "TTSBackend"),
    "s2s": ("s2s_backend", "S2SBackend"),
    "s2s_voicechat": ("s2s_voicechat_infer_backend", "S2SVoiceChatInferBackend"),
    "s2s_incremental": ("s2s_incremental_backend", "S2SIncrementalBackend"),
    "s2s_incremental_v2": ("s2s_incremental_backend_v2", "S2SIncrementalBackendV2"),
    "s2s_session": ("s2s_session_backend", "S2SSessionBackend"),
}


def list_backends() -> list:
    """Return list of available backend names."""
    return list(BACKEND_REGISTRY.keys())


def get_backend(backend_name: str) -> type:
    """
    Get backend class by name with lazy loading.

    Args:
        backend_name: One of 'salm', 'tts', 's2s'

    Returns:
        Backend class (not instance)

    Raises:
        ValueError: If backend name is unknown
        ImportError: If backend dependencies are not available
    """
    if backend_name not in BACKEND_REGISTRY:
        available = ", ".join(BACKEND_REGISTRY.keys())
        raise ValueError(f"Unknown backend: '{backend_name}'. Available backends: {available}")

    module_name, class_name = BACKEND_REGISTRY[backend_name]

    import importlib

    try:
        module = importlib.import_module(f".{module_name}", package=__name__)
        return getattr(module, class_name)
    except ImportError as e:
        raise ImportError(
            f"Failed to import backend '{backend_name}'. Make sure required dependencies are installed. Error: {e}"
        ) from e
