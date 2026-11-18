"""Exercise paid-adapter payloads with local fakes only; no network calls."""
import json
import sys
from types import SimpleNamespace

import pytest

from agri_extract.experiments import execute
from agri_extract.providers import Completion, LiveProvider, MockProvider
from agri_extract.synthetic import artifact, responses


class FakeAnthropicStream:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def get_final_message(self):
        return self.response


@pytest.mark.parametrize("provider,key_name", [
    ("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY")])
def test_provider_payload_offline(monkeypatch, provider, key_name):
    seen = []
    monkeypatch.setenv(key_name, "local-test-key-never-sent")

    class FakeClient:
        def __init__(self, **kwargs):
            seen.append(("client", kwargs))
            self.messages = SimpleNamespace(
                stream=lambda **kwargs: FakeAnthropicStream(self.anthropic_create(**kwargs)))
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.openai_create))

        def anthropic_create(self, **kwargs):
            seen.append(("request", kwargs))
            return SimpleNamespace(content=[SimpleNamespace(type="text", text='{"ok":true}')],
                usage=SimpleNamespace(input_tokens=12, output_tokens=4), stop_reason="end_turn")

        def openai_create(self, **kwargs):
            seen.append(("request", kwargs))
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"ok":true}'), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4))

    monkeypatch.setitem(sys.modules, provider, SimpleNamespace(**{
        "Anthropic" if provider == "anthropic" else "OpenAI": FakeClient}))
    result = LiveProvider(provider, "test-model", allow_live=True).complete("system", "user", max_output_tokens=256)
    assert result.text == '{"ok":true}'
    assert (result.input_tokens, result.output_tokens) == (12, 4)
    assert seen[0][1]["api_key"] == "local-test-key-never-sent"
    assert seen[0][1]["max_retries"] == 0
    assert seen[1][1]["model"] == "test-model"
    if provider == "anthropic":
        assert seen[1][1]["max_tokens"] == 256
        assert seen[1][1]["messages"] == [{"role": "user", "content": "user"}]
    else:
        assert seen[1][1]["max_completion_tokens"] == 256
        assert seen[1][1]["messages"][0] == {"role": "developer", "content": "system"}


def test_deepinfra_compatible_payload_uses_max_tokens(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "offline-only-key")
    captured = []

    class FakeClient:
        def __init__(self, **kwargs):
            captured.append(("client", kwargs))
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            captured.append(("request", kwargs))
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="{}"), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeClient))
    provider = LiveProvider("deepinfra", "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        allow_live=True, endpoint="https://api.deepinfra.com/v1/openai",
        api_key_env="DEEPINFRA_API_KEY", api_protocol="openai-compatible",
        auth_scheme="bearer", underlying_model_family="DeepSeek V4.1 Flash")
    provider.complete("system", "user", max_output_tokens=4096)
    assert captured[0][1]["base_url"] == "https://api.deepinfra.com/v1/openai"
    assert captured[1][1]["max_tokens"] == 4096
    assert "max_completion_tokens" not in captured[1][1]
    assert captured[1][1]["messages"][0]["role"] == "system"


@pytest.mark.parametrize("provider,key_name", [
    ("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY")])
def test_native_schema_payload_offline(monkeypatch, provider, key_name):
    captured = []
    monkeypatch.setenv(key_name, "offline-only-key")

    class FakeClient:
        def __init__(self, **kwargs):
            self.messages = SimpleNamespace(
                stream=lambda **kwargs: FakeAnthropicStream(self.create_anthropic(**kwargs)))
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create_openai))

        def create_anthropic(self, **kwargs):
            captured.append(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="{}")],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1), stop_reason="end_turn")

        def create_openai(self, **kwargs):
            captured.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="{}"), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    monkeypatch.setitem(sys.modules, provider, SimpleNamespace(**{
        "Anthropic" if provider == "anthropic" else "OpenAI": FakeClient}))
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    LiveProvider(provider, "offline-model", allow_live=True, protocol="native").complete(
        "system", "user", json_schema=schema)
    if provider == "anthropic":
        assert captured[0]["output_config"]["format"]["schema"] == schema
    else:
        assert captured[0]["response_format"]["json_schema"]["schema"] == schema


def test_inference_parameters_are_explicit_and_traced(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-only-key")
    captured = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            captured.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="{}"), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeClient))
    provider = LiveProvider("openai", "offline-model", allow_live=True,
                            temperature=0.2, seed=123)
    provider.complete("system", "user")
    assert captured[0]["temperature"] == 0.2
    assert captured[0]["seed"] == 123
    with pytest.raises(ValueError, match="seed"):
        LiveProvider("anthropic", "offline-model", allow_live=True, seed=123)


@pytest.mark.parametrize("strategy", ["E1", "E2"])
def test_separate_run_outputs_for_s3(tmp_path, strategy):
    config = {"strategy": strategy, "schema": "S3", "validation": "V1", "parser": "P4"}
    folder = execute(config, artifact(parser="P4"), [MockProvider(responses(strategy, "S3"))],
                     "offline", output_root=tmp_path)
    assert json.loads((folder / "raw.json").read_text())["rules"]
    assert json.loads((folder / "validated.json").read_text())["rules"]
    evidence = json.loads((folder / "evidence.json").read_text())
    assert evidence["status"] == "SOURCE_PASSAGES_UNLINKED"
    assert evidence["source_passages"]
    assert (folder / "cost.json").exists() and (folder / "run.log").exists()
    assert (folder / "candidates.json").exists() == (strategy == "E2")


@pytest.mark.parametrize("reason,expected", [
    ("max_tokens", "TRUNCATED_OUTPUT"),
    ("length", "TRUNCATED_OUTPUT"),
    ("content_filter", "INCOMPLETE_OUTPUT")])
def test_valid_json_with_nonfinal_stop_is_not_accepted(tmp_path, reason, expected):
    class InterruptedProvider:
        name = "openai"
        model = "offline-test"
        input_usd_per_million = 1.0
        output_usd_per_million = 2.0

        def complete(self, system, user, *, json_schema=None, max_output_tokens=8192):
            return Completion('{"document_id":"synthetic","rules":[]}', self.name,
                self.model, "mock://local", 100, 20, 0.2, reason)

    config = {"strategy": "E1", "schema": "S3", "validation": "V1", "parser": "P4"}
    folder = execute(config, artifact(parser="P4"), [InterruptedProvider()], "offline", output_root=tmp_path)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["status"] == expected
    assert manifest["calls"] == 1
    assert manifest["cost"] == pytest.approx(0.00014)
    assert json.loads((folder / "partial_trace.json").read_text())[0]["stop_reason"] == reason
    assert not (folder / "validated.json").exists()


def test_context_preflight_blocks_before_api_import(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-test-key")
    provider = LiveProvider("anthropic", "offline-test", allow_live=True,
                            context_window_tokens=10)
    with pytest.raises(RuntimeError, match="CONTEXT_LIMIT_ESTIMATE_EXCEEDED"):
        provider.complete("system", "a long document", max_output_tokens=8)


def test_api_context_error_has_explicit_run_status(tmp_path):
    class ContextErrorProvider:
        name = "openai"
        model = "offline-test"

        def complete(self, system, user, *, json_schema=None, max_output_tokens=8192):
            raise RuntimeError("maximum context length exceeded")

    config = {"strategy": "E1", "schema": "S3", "validation": "V1", "parser": "P4"}
    folder = execute(config, artifact(parser="P4"), [ContextErrorProvider()], "offline", output_root=tmp_path)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["status"] == "CONTEXT_LIMIT_EXCEEDED"
    assert manifest["calls"] == 0


@pytest.mark.parametrize("message", [
    "Your credit balance is too low to access the Anthropic API",
    "You have no credits remaining. Add credits to continue using the API",
    "insufficient_quota",
])
def test_insufficient_credits_are_distinct_from_rate_limits(tmp_path, message):
    class UnfundedProvider:
        name = "openai"
        model = "offline-test"

        def complete(self, system, user, *, json_schema=None, max_output_tokens=8192):
            raise RuntimeError(message)

    config = {"strategy": "E1", "schema": "S3", "validation": "V1", "parser": "P4"}
    folder = execute(config, artifact(parser="P4"), [UnfundedProvider()],
                     "offline", output_root=tmp_path)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["status"] == "INSUFFICIENT_CREDITS"
    assert manifest["calls"] == 0
