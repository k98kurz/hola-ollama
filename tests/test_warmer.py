"""
Tests for ModelWarmer functionality.
"""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import unittest
from hola.ollama.warmer import ModelWarmer
from hola.ollama.config import WarmerConfig
from hola.ollama.client import OllamaClient


class TestWarmer(unittest.TestCase):
    """Test cases for ModelWarmer."""
    def setUp(self):
        """Set up test fixtures."""
        self.config = WarmerConfig(
            ping_interval_seconds=120, check_interval_seconds=1
        )
        self.mock_client = Mock(spec=OllamaClient)
        self.mock_client.load_model = AsyncMock()

        async def mock_chat(request):
            yield

        self.mock_client.chat = mock_chat
        self.warmer = ModelWarmer(self.mock_client, self.config)

    def tearDown(self):
        """Clean up after tests."""
        if self.warmer._thread is not None and self.warmer._thread.is_alive():
            self.warmer.stop()

    def test_model_warmer_initialization(self):
        """Test ModelWarmer initialization."""
        assert self.warmer._client == self.mock_client, (
            self.warmer._client,
            self.mock_client,
        )
        assert self.warmer._config == self.config, (
            self.warmer._config,
            self.config,
        )
        assert len(self.warmer._model_timers) == 0, len(self.warmer._model_timers)
        assert self.warmer._thread is None, self.warmer._thread

    def test_warm_model(self):
        """Test warming a model."""
        self.warmer.warm_model("qwen3:0.6b")

        assert "qwen3:0.6b" in self.warmer._model_timers, (
            list(self.warmer._model_timers.keys()),
        )
        self.mock_client.load_model.assert_called_once_with(
            "qwen3:0.6b",
            "4h",
        )

    def test_unwarm_model(self):
        """Test unwarming a model."""
        self.warmer.warm_model("qwen3:0.6b")
        self.warmer.unwarm_model("qwen3:0.6b")

        assert "qwen3:0.6b" not in self.warmer._model_timers, (
            list(self.warmer._model_timers.keys()),
        )

    def test_unwarm_nonexistent_model(self):
        """Test unwarming a non-existent model."""
        self.warmer.unwarm_model("nonexistent")

        assert len(self.warmer._model_timers) == 0, len(self.warmer._model_timers)

    def test_multiple_models(self):
        """Test tracking multiple models."""
        self.warmer.warm_model("qwen3:0.6b")
        self.warmer.warm_model("gemma3:1b")
        self.warmer.warm_model("tinyllama")

        assert len(self.warmer._model_timers) == 3, len(self.warmer._model_timers)
        for model_name in ["qwen3:0.6b", "gemma3:1b", "tinyllama"]:
            assert model_name in self.warmer._model_timers, model_name

    def test_get_warm_models(self):
        """Test getting list of warm models."""
        self.warmer.warm_model("qwen3:0.6b")
        self.warmer.warm_model("gemma3:1b")

        warm_models = self.warmer.get_warm_models()

        assert len(warm_models) == 2, len(warm_models)
        assert "qwen3:0.6b" in warm_models, warm_models
        assert "gemma3:1b" in warm_models, warm_models

    def test_is_model_warm(self):
        """Test checking if model is warm."""
        self.warmer.warm_model("qwen3:0.6b")

        assert self.warmer.is_model_warm("qwen3:0.6b"), self.warmer.is_model_warm(
            "qwen3:0.6b",
        )
        assert not self.warmer.is_model_warm("gemma3:1b"), (
            self.warmer.is_model_warm("gemma3:1b"),
        )

    def test_background_loop_start_stop(self):
        """Test starting and stopping background loop."""
        self.warmer.start()

        assert self.warmer._thread is not None, self.warmer._thread
        assert self.warmer._thread.is_alive(), "Thread not alive"

        self.warmer.stop()

        assert self.warmer._thread is None or not self.warmer._thread.is_alive(), (
            "Thread still alive",
        )

    def test_background_loop_already_running(self):
        """Test starting background loop when already running."""
        self.warmer.start()
        original_thread = self.warmer._thread

        self.warmer.start()

        assert self.warmer._thread == original_thread, (
            self.warmer._thread,
            original_thread,
        )

        self.warmer.stop()

    def test_graceful_shutdown(self):
        """Test graceful shutdown of warmer."""
        self.warmer.warm_model("qwen3:0.6b")
        self.warmer.start()

        time.sleep(0.1)

        self.warmer.stop()

        assert self.warmer._stop_event.is_set(), (
            self.warmer._stop_event.is_set(),
        )

    def test_graceful_shutdown_timeout(self):
        """Test graceful shutdown with timeout."""
        self.warmer.start()

        with patch.object(self.warmer._thread, 'join') as mock_join:
            mock_join.return_value = None
            self.warmer._thread.is_alive = Mock(return_value=True)

            self.warmer.stop()

            mock_join.assert_called_once_with(timeout=5.0)

    def test_signal_model_used_warmed_model(self):
        """Test signaling that a warmed model was just used."""
        self.warmer.warm_model("qwen3:0.6b")
        initial_timer = self.warmer._model_timers["qwen3:0.6b"]

        time.sleep(0.1)
        self.warmer.signal_model_used("qwen3:0.6b")

        new_timer = self.warmer._model_timers["qwen3:0.6b"]

        assert new_timer > initial_timer, (new_timer, initial_timer)

    def test_signal_model_used_non_warmed_model(self):
        """Test signaling that a non-warmed model was just used."""
        self.warmer.signal_model_used("nonexistent")

        assert "nonexistent" not in self.warmer._model_timers, (
            list(self.warmer._model_timers.keys()),
        )

    def test_signal_prevents_unnecessary_ping(self):
        """Test that signaling prevents immediate ping."""
        self.warmer.warm_model("qwen3:0.6b")
        self.warmer.start()

        time.sleep(0.1)
        self.warmer.signal_model_used("qwen3:0.6b")

        time.sleep(0.1)

        self.warmer.stop()


if __name__ == "__main__":
    unittest.main()
