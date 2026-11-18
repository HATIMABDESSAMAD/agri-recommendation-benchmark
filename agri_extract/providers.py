"""Paid providers are inert unless the caller explicitly enables live mode."""
from __future__ import annotations

import os
import time
from math import ceil
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Completion:
    text: str
    provider: str
    model: str
    endpoint: str
    input_tokens: int | None
    output_tokens: int | None
    latency_seconds: float
    stop_reason: str | None


class Provider(Protocol):
    name: str
    model: str
    def complete(self, system: str, user: str, *, json_schema: dict | None = None,
                 max_output_tokens: int = 8192) -> Completion: ...


class MockProvider:
    name = "mock"

    def __init__(self, responses: list[str], model: str = "mock-v1"):
        self.responses = responses[:]
        self.model = model
        self.calls: list[dict] = []

    def complete(self, system: str, user: str, *, json_schema: dict | None = None,
                 max_output_tokens: int = 8192) -> Completion:
        self.calls.append({"system": system, "user": user, "schema": json_schema,
                           "max_output_tokens": max_output_tokens})
        if not self.responses:
            raise RuntimeError("mock response queue exhausted")
        return Completion(self.responses.pop(0), self.name, self.model, "mock://local", None, None, 0.0, "end_turn")


class LiveProvider:
    def __init__(self, name: str, model: str, *, allow_live: bool = False,
                 endpoint: str | None = None, api_key_env: str | None = None,
                 protocol: str = "common", snapshot: str | None = None,
                 underlying_model_family: str | None = None,
                 input_usd_per_million: float | None = None,
                 output_usd_per_million: float | None = None,
                 api_protocol: str | None = None, auth_scheme: str | None = None,
                 context_window_tokens: int | None = None,
                 temperature: float | None = None, seed: int | None = None):
        if not allow_live:
            raise PermissionError("Paid API calls require explicit --live mode")
        if name not in {"anthropic", "openai", "deepinfra"}:
            raise ValueError(name)
        if name == "deepinfra" and (not endpoint or not api_key_env or
                                    api_protocol != "openai-compatible" or auth_scheme != "bearer" or
                                    not underlying_model_family):
            raise ValueError("deepinfra endpoint, key env, protocol, bearer auth and underlying model family are required")
        if context_window_tokens is not None and (not isinstance(context_window_tokens, int)
                                                  or context_window_tokens <= 0):
            raise ValueError("context_window_tokens must be a positive integer")
        if temperature is not None and not isinstance(temperature, (int, float)):
            raise ValueError("temperature must be numeric")
        if seed is not None and (not isinstance(seed, int) or name == "anthropic"):
            raise ValueError("seed is supported only for OpenAI-compatible adapters")
        self.name, self.model, self.endpoint, self.api_key_env, self.protocol = name, model, endpoint, api_key_env, protocol
        self.snapshot = snapshot
        self.underlying_model_family = underlying_model_family
        self.input_usd_per_million = input_usd_per_million
        self.output_usd_per_million = output_usd_per_million
        self.api_protocol = api_protocol
        self.auth_scheme = auth_scheme
        self.context_window_tokens = context_window_tokens
        self.temperature = temperature
        self.seed = seed

    def complete(self, system: str, user: str, *, json_schema: dict | None = None,
                 max_output_tokens: int = 8192) -> Completion:
        # A provider-independent estimate with a margin for typical English prose.
        # This is not a guaranteed upper bound or a substitute for the model tokenizer.
        if self.context_window_tokens is not None:
            estimated_input = ceil(len((system + user).encode("utf-8")) / 3)
            if estimated_input + max_output_tokens > self.context_window_tokens:
                raise RuntimeError(
                    f"CONTEXT_LIMIT_ESTIMATE_EXCEEDED: estimated_input={estimated_input}, "
                    f"max_output={max_output_tokens}, window={self.context_window_tokens}")
        key_name = self.api_key_env or ("ANTHROPIC_API_KEY" if self.name == "anthropic" else "OPENAI_API_KEY")
        key = os.environ.get(key_name)
        if not key:
            raise RuntimeError(f"BLOCKED_MISSING_CONFIGURATION: {key_name}")
        start = time.monotonic()
        if self.name == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=key, max_retries=0)
            args = {"model": self.model, "max_tokens": max_output_tokens,
                    "system": system, "messages": [{"role": "user", "content": user}]}
            if self.temperature is not None:
                args["temperature"] = self.temperature
            if self.protocol == "native" and json_schema is not None:
                args["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}
            with client.messages.stream(**args) as stream:
                response = stream.get_final_message()
            result = "\n".join(b.text for b in response.content if b.type == "text")
            return Completion(result, self.name, self.model, "https://api.anthropic.com/v1/messages",
                              response.usage.input_tokens, response.usage.output_tokens,
                              time.monotonic() - start, response.stop_reason)
        import openai
        if self.name == "deepinfra" and self.protocol not in {"openai-compatible", "common", "native"}:
            raise ValueError("unsupported deepinfra protocol; configure custom adapter")
        client = openai.OpenAI(api_key=key, base_url=self.endpoint if self.name == "deepinfra" else None, max_retries=0)
        role = "developer" if self.name == "openai" else "system"
        output_limit = ("max_tokens" if self.name == "deepinfra" else "max_completion_tokens")
        args = {"model": self.model, output_limit: max_output_tokens,
                "messages": [{"role": role, "content": system},
                             {"role": "user", "content": user}]}
        if self.temperature is not None:
            args["temperature"] = self.temperature
        if self.seed is not None:
            args["seed"] = self.seed
        if self.protocol == "native" and json_schema is not None:
            args["response_format"] = {"type": "json_schema", "json_schema":
                {"name": "agri_rules", "strict": True, "schema": json_schema}}
        response = client.chat.completions.create(**args)
        choice = response.choices[0]
        return Completion(choice.message.content or "", self.name, self.model,
            self.endpoint or "https://api.openai.com/v1/chat/completions",
            response.usage.prompt_tokens if response.usage else None,
            response.usage.completion_tokens if response.usage else None,
            time.monotonic() - start, choice.finish_reason)
