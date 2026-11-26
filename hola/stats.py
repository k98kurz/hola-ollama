"""
Stats collection and normalization for LLM client responses.

Resolves each ResponseStats field from a list of known aliases,
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
    ResponseStats,
)


def _resolve_path(data: dict, aliases: list[str]) -> int | float:
    """Try each alias path (supports dotted paths like
        `usage.prompt_tokens`); return the first non-`None` value or 0.
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
    return 0


def _extract_stats(data: dict, time_divisor: int = 1) -> dict[str, object]:
    """Resolve stats fields from an OpenAI/llama.cpp or Ollama
        response/chunk dict, returning a normalized dict with
        nanosecond durations and token counts.
    """
    result: dict[str, object] = {}

    result["prompt_tokens"] = _resolve_path(
        data,
        [
            "usage.prompt_tokens",
            "timings.prompt_n",
            "prompt_tokens",
            "prompt_eval_count",
        ],
    )

    result["output_tokens"] = _resolve_path(
        data,
        [
            "usage.completion_tokens",
            "timings.predicted_n",
            "output_tokens",
            "eval_count",
        ],
    )

    prompt_ms = _resolve_path(data, ["timings.prompt_ms"])
    if prompt_ms:
        result["prompt_duration"] = float(prompt_ms) / time_divisor
    else:
        result["prompt_duration"] = _resolve_path(
            data, ["prompt_duration", "prompt_eval_duration"]
        ) / time_divisor

    pred_ms = _resolve_path(data, ["timings.predicted_ms"])
    if pred_ms:
        result["output_duration"] = float(pred_ms) / time_divisor
    else:
        result["output_duration"] = _resolve_path(
            data, ["output_duration", "eval_duration"]
        ) / time_divisor

    total_duration = _resolve_path(data, ["total_duration"]) / time_divisor
    if total_duration:
        result["total_duration"] = total_duration
    else:
        result["total_duration"] = (
            result["prompt_duration"] + result["output_duration"]
        )

    result["load_duration"] = _resolve_path(data, ["load_duration"]) / time_divisor

    prompt_tps = _resolve_path(data, ["timings.prompt_per_second"])
    if prompt_tps:
        result["prompt_tps"] = float(prompt_tps)
    elif result["prompt_tokens"] and result["prompt_duration"]:
        result["prompt_tps"] = result["prompt_tokens"] / result["prompt_duration"]
    else:
        result["prompt_tps"] = 0.0

    output_tps = _resolve_path(data, ["timings.predicted_per_second"])
    if output_tps:
        result["output_tps"] = float(output_tps)
    elif result["output_tokens"] and result["output_duration"]:
        result["output_tps"] = result["output_tokens"] / result["output_duration"]
    else:
        result["output_tps"] = 0.0

    return result


def extract_stats(data: dict, time_divisor: int = 1) -> ResponseStats:
    stat_fields = [f.name for f in fields(ResponseStats)]
    stats = ResponseStats()
    data = _extract_stats(data, time_divisor)
    for f in stat_fields:
        val = data.get(f, 0)
        setattr(stats, f, val)
    if stats.prompt_tokens and stats.prompt_duration and not stats.prompt_tps:
        stats.prompt_tps = stats.prompt_tokens / stats.prompt_duration
    if stats.output_tokens and stats.output_duration and not stats.output_tps:
        stats.output_tps = stats.output_tokens / stats.output_duration
    return stats
