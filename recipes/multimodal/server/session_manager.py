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
Session manager for S2S session backend.

Manages session state (LLM KV cache, frame index, etc.) across HTTP requests
to enable multi-turn conversations.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class TurnData:
    """Data for a single turn in a conversation."""

    turn_idx: int
    user_audio_bytes: Optional[bytes] = None  # Input audio from user
    agent_audio_bytes: Optional[bytes] = None  # Output audio from agent
    agent_text: str = ""  # Text response for this turn
    user_duration_sec: float = 0.0  # Duration of user audio
    agent_duration_sec: float = 0.0  # Duration of agent audio


@dataclass
class SessionState:
    """State that persists between turns in a session."""

    session_id: str

    # LLM state
    llm_cache: Any = None  # DynamicCache (for non-Mamba models)
    input_embeds_history: Any = None  # List of embeddings (for Mamba models)
    frame_idx: int = 0

    # Token history (for turn-taking logic)
    gen_text: Optional[torch.Tensor] = None
    gen_asr_text: Optional[torch.Tensor] = None

    # Audio buffer state
    audio_buffer: Optional[torch.Tensor] = None
    buffer_fill_level: int = 0

    # Turn tracking
    turn_count: int = 0

    # Per-turn data for session audio generation
    turns: List[TurnData] = field(default_factory=list)

    # Timestamps
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def touch(self):
        """Update last_accessed timestamp."""
        self.last_accessed = time.time()


class SessionManager:
    """
    Manages session state for S2S multi-turn conversations.

    Thread-safe implementation with TTL-based cleanup.
    """

    def __init__(self, ttl_seconds: float = 300.0, max_sessions: int = 100):
        """
        Initialize SessionManager.

        Args:
            ttl_seconds: Time-to-live for sessions in seconds (default: 5 minutes)
            max_sessions: Maximum number of concurrent sessions
        """
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.sessions: Dict[str, SessionState] = {}
        self._lock = threading.RLock()

    def create_session(self, session_id: Optional[str] = None) -> SessionState:
        """
        Create a new session.

        Args:
            session_id: Optional session ID. If None, generates a UUID.

        Returns:
            New SessionState object
        """
        with self._lock:
            if session_id is None:
                session_id = str(uuid.uuid4())

            # Clean up expired sessions first
            self._cleanup_expired_locked()

            # Evict oldest if at capacity
            if len(self.sessions) >= self.max_sessions:
                self._evict_oldest_locked()

            state = SessionState(session_id=session_id)
            self.sessions[session_id] = state
            print(f"[SessionManager] Created session: {session_id}")
            return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """
        Get existing session by ID.

        Args:
            session_id: Session ID to look up

        Returns:
            SessionState if found and not expired, None otherwise
        """
        with self._lock:
            state = self.sessions.get(session_id)
            if state is None:
                return None

            # Check if expired
            if time.time() - state.last_accessed > self.ttl_seconds:
                print(f"[SessionManager] Session expired: {session_id}")
                del self.sessions[session_id]
                return None

            state.touch()
            return state

    def get_or_create_session(self, session_id: Optional[str] = None) -> SessionState:
        """
        Get existing session or create new one.

        Args:
            session_id: Session ID. If None, creates new session.

        Returns:
            SessionState (existing or new)
        """
        if session_id:
            state = self.get_session(session_id)
            if state is not None:
                return state

        return self.create_session(session_id)

    def save_session(self, session_id: str, state: SessionState):
        """
        Save/update session state.

        Args:
            session_id: Session ID
            state: SessionState to save
        """
        with self._lock:
            state.touch()
            self.sessions[session_id] = state

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session ID to delete

        Returns:
            True if session was deleted, False if not found
        """
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                print(f"[SessionManager] Deleted session: {session_id}")
                return True
            return False

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session info without full state.

        Args:
            session_id: Session ID

        Returns:
            Dict with session metadata or None
        """
        with self._lock:
            state = self.sessions.get(session_id)
            if state is None:
                return None

            return {
                "session_id": state.session_id,
                "frame_idx": state.frame_idx,
                "turn_count": state.turn_count,
                "created_at": state.created_at,
                "last_accessed": state.last_accessed,
                "has_llm_cache": state.llm_cache is not None,
                "has_input_embeds_history": state.input_embeds_history is not None
                and len(state.input_embeds_history) > 0,
            }

    def list_sessions(self) -> list:
        """List all active session IDs."""
        with self._lock:
            return list(self.sessions.keys())

    def cleanup_expired(self):
        """Clean up expired sessions (called periodically)."""
        with self._lock:
            self._cleanup_expired_locked()

    def _cleanup_expired_locked(self):
        """Clean up expired sessions (must hold lock)."""
        now = time.time()
        expired = [sid for sid, state in self.sessions.items() if now - state.last_accessed > self.ttl_seconds]
        for sid in expired:
            print(f"[SessionManager] Cleaning up expired session: {sid}")
            del self.sessions[sid]

    def _evict_oldest_locked(self):
        """Evict oldest session to make room (must hold lock)."""
        if not self.sessions:
            return

        oldest_id = min(self.sessions.keys(), key=lambda sid: self.sessions[sid].last_accessed)
        print(f"[SessionManager] Evicting oldest session: {oldest_id}")
        del self.sessions[oldest_id]

    def __len__(self) -> int:
        """Return number of active sessions."""
        with self._lock:
            return len(self.sessions)
