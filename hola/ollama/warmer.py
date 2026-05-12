"""
Model warmer for keeping models loaded in memory.
"""

from typing import Self
from hola.ollama.client import OllamaClient
from hola.ollama.config import WarmerConfig
from hola.ollama.types import ChatRequest, GenerationOptions, Message, Role
import asyncio
import logging
import threading
import time


logger = logging.getLogger(__name__)


class ModelWarmer:
    def __init__(
            self,
            client: OllamaClient,
            config: WarmerConfig,
        ) -> None:
        self._client = client
        self._config = config
        self._model_timers: dict[str, float] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def warm_model(
            self,
            model: str,
        ) -> None:
        """Mark model to keep warm."""
        with self._lock:
            self._model_timers[model] = (
                time.time() + self._config.ping_interval_seconds
            )

        logger.info(
            "Warming model %s (ping_interval: %ds)",
            model,
            self._config.ping_interval_seconds,
        )

        asyncio.run(self._load_model(model))

    def unwarm_model(
            self,
            model: str,
        ) -> None:
        """Remove model from warming."""
        with self._lock:
            if model in self._model_timers:
                del self._model_timers[model]

        logger.info("Unwarming model %s", model)

    def signal_model_used(
            self,
            model: str,
        ) -> None:
        """Reset timer for model that was just used."""
        with self._lock:
            if model not in self._model_timers:
                return

            self._model_timers[model] = (
                time.time() + self._config.ping_interval_seconds
            )

        logger.debug("Reset timer for model %s", model)

    def start(self) -> None:
        """Start background warming thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Background thread already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_background_loop,
            daemon=True,
        )
        self._thread.start()
        logger.info("Started background warming thread")

    def stop(self) -> None:
        """Stop background warming thread."""
        if self._thread is None or not self._thread.is_alive():
            return

        self._stop_event.set()
        self._thread.join(timeout=5.0)

        if self._thread.is_alive():
            logger.warning("Background thread did not stop gracefully")

        with self._lock:
            self._model_timers.clear()

        self._thread = None
        logger.info("Stopped background warming thread")

    def get_warm_models(
            self,
        ) -> list[str]:
        """List currently warm models."""
        with self._lock:
            return list(self._model_timers.keys())

    def is_model_warm(
            self,
            model: str,
        ) -> bool:
        """Check if model is warm."""
        with self._lock:
            return model in self._model_timers

    async def _background_ping_loop(self) -> None:
        """Background loop for pinging models."""
        logger.info("Background ping loop started")

        while not self._stop_event.is_set():
            current_time = time.time()
            models_to_ping = []

            with self._lock:
                for model, next_ping_time in self._model_timers.items():
                    if current_time >= next_ping_time:
                        models_to_ping.append(model)

            for model in models_to_ping:
                if self._stop_event.is_set():
                    break

                try:
                    await self._ping_model(model)

                    with self._lock:
                        if model in self._model_timers:
                            self._model_timers[model] = (
                                time.time() + self._config.ping_interval_seconds
                            )

                except Exception as e:
                    logger.error("Error pinging model %s: %s", model, e)

            await asyncio.sleep(self._config.check_interval_seconds)

        logger.info("Background ping loop stopped")

    async def _ping_model(self, model: str) -> None:
        """Send 1-token request to keep model loaded."""
        request = ChatRequest(
            model=model,
            messages=[Message(role=Role.USER, content="Hi")],
            options=GenerationOptions(num_predict=1),
            keep_alive=self._config.default_keep_alive,
        )

        async for _ in self._client.chat(request):
            pass

    async def _load_model(
            self,
            model: str,
        ) -> None:
        """Trigger immediate model load."""
        await self._client.load_model(model, self._config.default_keep_alive)

    def _run_background_loop(self) -> None:
        """Run the background loop in a thread."""
        asyncio.run(self._background_ping_loop())

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        self.start()
        return self

    async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb,
        ) -> None:
        """Async context manager exit."""
        self.stop()
