"""
Test endpoint compatibility for all available Ollama models.
"""

import asyncio
import time
from hola.ollama.client import OllamaClient
from hola.ollama.types import (
    ChatRequest, GenerateRequest, GenerationOptions, Message, Role
)


ERROR_CODES: dict[str, str] = {}


def add_error_code(exception: Exception) -> str:
    error_type = type(exception).__name__
    error_msg = str(exception)

    for code, desc in ERROR_CODES.items():
        if error_type in desc:
            return code

    new_code = f"E{len(ERROR_CODES) + 1}"
    ERROR_CODES[new_code] = f"{error_type}: {error_msg}"
    return new_code


async def test_chat_endpoint(client: OllamaClient, model_name: str) -> str:
    timeout = 30.0
    try:
        start_time = time.time()
        request = ChatRequest(
            model=model_name,
            messages=[Message(role=Role.USER, content="hi")],
            options=GenerationOptions(num_predict=5),
            stream=False,
        )
        response = await asyncio.wait_for(
            client._chat_sync_async(request),
            timeout=timeout,
        )
        duration = time.time() - start_time
        return f"{duration:.1f}s"
    except asyncio.TimeoutError:
        return add_error_code(TimeoutError(f"Request timed out after {timeout}s"))
    except Exception as e:
        return add_error_code(e)


async def test_generate_endpoint(client: OllamaClient, model_name: str) -> str:
    timeout = 30.0
    try:
        start_time = time.time()
        request = GenerateRequest(
            model=model_name,
            prompt="hi",
            options=GenerationOptions(num_predict=5),
            stream=False,
        )
        response = await asyncio.wait_for(
            client._generate_sync_async(request),
            timeout=timeout,
        )
        duration = time.time() - start_time
        return f"{duration:.1f}s"
    except asyncio.TimeoutError:
        return add_error_code(TimeoutError(f"Request timed out after {timeout}s"))
    except Exception as e:
        return add_error_code(e)


def print_compatibility_table(
    results: dict[str, dict[str, str]],
) -> None:
    print("Ollama Model Endpoint Response Times")
    print("=" * 60)
    print(f"{'Model Name':<30} {'/api/chat':<12} {'/api/generate':<12}")
    print("-" * 60)
    for model_name, result in results.items():
        print(f"{model_name:<30} {result['chat']:<12} {result['generate']:<12}")


def print_error_legend() -> None:
    print("\nError Code Legend:")
    print("-" * 40)
    for code, description in ERROR_CODES.items():
        print(f"{code}: {description}")


async def check_endpoint_compatibility() -> None:
    client = OllamaClient()
    models = await client.list_models()

    local_models = [m for m in models if m.name[-5:] != "cloud"]
    print(
        f"Testing {len(local_models)} models (excluding "
        f"{len(models) - len(local_models)} cloud models)...\n",
        flush=True
    )

    results = {}
    for i, model in enumerate(local_models, 1):
        print(f"[{i}/{len(local_models)}] Testing {model.name}...", end=" ",
              flush=True)
        await client.load_model(model.name)
        chat_result = await test_chat_endpoint(client, model.name)
        generate_result = await test_generate_endpoint(client, model.name)
        await client.unload_model(model.name)
        results[model.name] = {
            "chat": chat_result,
            "generate": generate_result,
        }
        print(f"chat: {chat_result}, generate: {generate_result}", flush=True)

    await client.close()
    print("\n", flush=True)
    print_compatibility_table(results)

    if ERROR_CODES:
        print_error_legend()


if __name__ == "__main__":
    asyncio.run(check_endpoint_compatibility())
