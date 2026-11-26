"""
Stats collection and normalization for LLM client responses.

Resolves each MessageStats field from a list of known aliases,
converting llama.cpp timings (ms) to nanoseconds, and computing
tps from llama.cpp per_second values with a fallback to
tokens/duration.
"""

from __future__ import annotations
from collections.abc import AsyncGenerator
from dataclasses import fields
from hola.types import (
    ChatResponse,
    ChatResponseChunk,
    GenerateChunk,
    GenerateResponse,
    MessageStats,
)


def _resolve_path(data: dict, aliases: list[str]) -> object | None:
    """Try each alias path (supports dotted paths like
        `usage.prompt_tokens`); return the first non-`None` value.
    """
    for alias in aliases:
        parts = alias.split(".")
        val: object = data
        found = True
        for part in parts:
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                found = False
                break
        if found and val is not None:
            return val
    return None


def extract_stats(data: dict) -> dict[str, object]:
    """Resolve stats fields from an OpenAI/llama.cpp or Ollama
        response/chunk dict, returning a normalized dict with
        nanosecond durations and token counts.
    """
    result: dict[str, object] = {}

    prompt_tokens = _resolve_path(
        data,
        [
            "usage.prompt_tokens",
            "timings.prompt_n",
            "prompt_tokens",
            "prompt_eval_count",
        ],
    )
    result["prompt_tokens"] = prompt_tokens if prompt_tokens is not None else 0

    output_tokens = _resolve_path(
        data,
        [
            "usage.completion_tokens",
            "timings.predicted_n",
            "output_tokens",
            "eval_count",
        ],
    )
    result["output_tokens"] = output_tokens if output_tokens is not None else 0

    prompt_ms = _resolve_path(data, ["timings.prompt_ms"])
    if prompt_ms is not None:
        result["prompt_duration"] = int(float(prompt_ms) * 1_000_000)
    else:
        result["prompt_duration"] = (
            _resolve_path(data, ["prompt_duration", "prompt_eval_duration"]) or 0
        )

    pred_ms = _resolve_path(data, ["timings.predicted_ms"])
    if pred_ms is not None:
        result["output_duration"] = int(float(pred_ms) * 1_000_000)
    else:
        result["output_duration"] = (
            _resolve_path(data, ["output_duration", "eval_duration"]) or 0
        )

    total_duration = _resolve_path(data, ["total_duration"])
    if total_duration is not None:
        result["total_duration"] = total_duration
    else:
        result["total_duration"] = (
            (result["prompt_duration"] or 0) + (result["output_duration"] or 0)
        )

    result["load_duration"] = _resolve_path(data, ["load_duration"]) or 0

    prompt_tps = _resolve_path(data, ["timings.prompt_per_second"])
    if prompt_tps is not None:
        result["prompt_tps"] = float(prompt_tps)
    elif result["prompt_tokens"] and result["prompt_duration"]:
        result["prompt_tps"] = (
            result["prompt_tokens"]
            / ((result["prompt_duration"] or 0) / 1_000_000_000)
        )
    else:
        result["prompt_tps"] = 0.0

    output_tps = _resolve_path(data, ["timings.predicted_per_second"])
    if output_tps is not None:
        result["output_tps"] = float(output_tps)
    elif result["output_tokens"] and result["output_duration"]:
        result["output_tps"] = (
            result["output_tokens"]
            / ((result["output_duration"] or 0) / 1_000_000_000)
        )
    else:
        result["output_tps"] = 0.0

    return result


def stats_from_response(
        response: ChatResponse | GenerateResponse,
    ) -> MessageStats:
    """Build a MessageStats from a ChatResponse or GenerateResponse."""
    stat_fields = [f.name for f in fields(MessageStats)]
    stats = MessageStats()
    for f in stat_fields:
        if f in ("prompt_tps", "output_tps"):
            continue
        val = getattr(response, f, None)
        if val is not None:
            setattr(stats, f, val)
    stats.total_duration /= 1_000_000_000
    stats.load_duration /= 1_000_000_000
    stats.prompt_duration /= 1_000_000_000
    stats.output_duration /= 1_000_000_000
    if stats.prompt_tokens and stats.prompt_duration:
        stats.prompt_tps = stats.prompt_tokens / stats.prompt_duration
    if stats.output_tokens and stats.output_duration:
        stats.output_tps = stats.output_tokens / stats.output_duration
    return stats


async def stats_from_chunks(
        chunks: AsyncGenerator[ChatResponseChunk | GenerateChunk, None],
    ) -> MessageStats:
    """Accumulate MessageStats from an async stream of chunks."""
    stat_fields = [f.name for f in fields(MessageStats)]
    stats = MessageStats()
    async for chunk in chunks:
        for f in stat_fields:
            if f in ("prompt_tps", "output_tps"):
                continue
            val = getattr(chunk, f, None)
            if val is not None:
                current = getattr(stats, f)
                setattr(stats, f, current + val)
    stats.total_duration /= 1_000_000_000
    stats.load_duration /= 1_000_000_000
    stats.prompt_duration /= 1_000_000_000
    stats.output_duration /= 1_000_000_000
    if stats.prompt_tokens and stats.prompt_duration:
        stats.prompt_tps = stats.prompt_tokens / stats.prompt_duration
    if stats.output_tokens and stats.output_duration:
        stats.output_tps = stats.output_tokens / stats.output_duration
    return stats
