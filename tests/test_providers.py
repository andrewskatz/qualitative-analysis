"""Tests for concrete LLM provider behavior."""

import asyncio
from unittest.mock import patch

from qualitative_analysis.core.providers import OllamaProvider


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, payload):
        self.payload = payload
        self.last_json = None

    async def post(self, url, json):
        self.last_json = json
        return _FakeResponse(self.payload)


def test_ollama_provider_suppresses_thinking_by_default():
    payload = {
        "message": {
            "content": '<think>hidden reasoning</think>{"entities_and_concepts": ["heat"]}',
            "thinking": "hidden reasoning",
        }
    }
    fake_client = _FakeAsyncClient(payload)
    provider = OllamaProvider("gpt-oss:120b", log_responses=True)

    with patch.object(provider, "_get_client", return_value=fake_client):
        with patch("builtins.print") as mock_print:
            response = asyncio.run(provider.generate("Find entities"))

    assert response == '{"entities_and_concepts": ["heat"]}'
    assert fake_client.last_json["think"] is False

    printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
    assert "[LLM THINKING]" not in printed
    assert "hidden reasoning" not in printed


def test_ollama_provider_preserves_thinking_when_enabled():
    payload = {
        "message": {
            "content": "final answer",
            "thinking": "visible reasoning",
        }
    }
    fake_client = _FakeAsyncClient(payload)
    provider = OllamaProvider(
        "gpt-oss:120b",
        enable_thinking=True,
        log_responses=True,
    )

    with patch.object(provider, "_get_client", return_value=fake_client):
        with patch("builtins.print") as mock_print:
            response = asyncio.run(provider.generate("Find entities"))

    assert response == "final answer"
    assert fake_client.last_json["think"] is True

    printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
    assert "[LLM THINKING]" in printed
    assert "visible reasoning" in printed


def test_ollama_provider_falls_back_to_thinking_when_content_is_empty():
    payload = {
        "message": {
            "content": "",
            "thinking": '{"entities_and_concepts": ["heat"]}',
        }
    }
    fake_client = _FakeAsyncClient(payload)
    provider = OllamaProvider("gpt-oss:120b", log_responses=False)

    with patch.object(provider, "_get_client", return_value=fake_client):
        response = asyncio.run(provider.generate("Find entities"))

    assert response == '{"entities_and_concepts": ["heat"]}'
    assert fake_client.last_json["think"] is False
