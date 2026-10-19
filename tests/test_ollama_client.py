import json

import httpx
import pytest

from labdino.ollama_client import (
    NGROK_HEADER,
    OllamaClient,
    OllamaConfigError,
    OllamaConnectionError,
    OllamaError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)

BASE = "http://ollama.test"


def make_client(handler, **kwargs) -> OllamaClient:
    return OllamaClient(BASE, "dino-model", transport=httpx.MockTransport(handler), **kwargs)


def test_generate_success_and_payload():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"response": "hola"})

    out = make_client(handler, temperature=0.7, seed=3).generate("di hola", system="sé breve")
    assert out == "hola"
    body = seen["body"]
    assert body["stream"] is False and body["model"] == "dino-model" and body["system"] == "sé breve"
    assert body["options"] == {"temperature": 0.7, "seed": 3}


def test_call_overrides_instance_defaults():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"response": "x"})

    make_client(handler, temperature=0.7, seed=3).generate("p", temperature=0.0, seed=9)
    assert seen["body"]["options"] == {"temperature": 0.0, "seed": 9}


def test_chat_prepends_system_and_returns_content():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "rawr"}})

    out = make_client(handler).chat([{"role": "user", "content": "hola"}], system="eres un dino")
    assert out == "rawr"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "eres un dino"}
    assert seen["body"]["messages"][1]["content"] == "hola" and seen["body"]["stream"] is False


def test_health_reports_version_and_model_availability():
    def handler(request):
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "9.9.9"})
        return httpx.Response(200, json={"models": [{"name": "dino-model:latest"}, {"name": "otro:1b"}]})

    h = make_client(handler).health()
    assert h["ollama_version"] == "9.9.9" and h["model_available"] is True


def test_health_model_missing():
    def handler(request):
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "1"})
        return httpx.Response(200, json={"models": [{"name": "otro:1b"}]})

    assert make_client(handler).health()["model_available"] is False


def test_read_timeout_message_mentions_variable():
    def handler(request):
        raise httpx.ReadTimeout("lento", request=request)

    with pytest.raises(OllamaTimeoutError, match="OLLAMA_READ_TIMEOUT"):
        make_client(handler).generate("p")


def test_connection_error_message_mentions_base_url():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(OllamaConnectionError, match="OLLAMA_BASE_URL"):
        make_client(handler).version()


def test_model_not_found_mentions_variable():
    def handler(request):
        return httpx.Response(404, json={"error": "model 'dino-model' not found"})

    with pytest.raises(OllamaModelNotFoundError, match="OLLAMA_MODEL"):
        make_client(handler).generate("p")


def test_non_json_response_is_clear_error():
    def handler(request):
        return httpx.Response(200, text="<html>ngrok</html>")

    with pytest.raises(OllamaError, match="OLLAMA_NGROK_SKIP_WARNING"):
        make_client(handler).version()


@pytest.mark.parametrize("enabled", [True, False])
def test_ngrok_header_is_configurable(enabled):
    seen = {}

    def handler(request):
        seen["headers"] = request.headers
        return httpx.Response(200, json={"version": "1"})

    make_client(handler, ngrok_skip_warning=enabled).version()
    assert (NGROK_HEADER in seen["headers"]) is enabled


def test_from_env_requires_variables(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    with pytest.raises(OllamaConfigError, match="OLLAMA_BASE_URL"):
        OllamaClient.from_env()
    monkeypatch.setenv("OLLAMA_BASE_URL", BASE)
    with pytest.raises(OllamaConfigError, match="OLLAMA_MODEL"):
        OllamaClient.from_env()


def test_from_env_reads_settings(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", BASE + "/")
    monkeypatch.setenv("OLLAMA_MODEL", "m")
    monkeypatch.setenv("OLLAMA_NGROK_SKIP_WARNING", "1")
    monkeypatch.setenv("OLLAMA_READ_TIMEOUT", "42")
    c = OllamaClient.from_env()
    assert c.base_url == BASE and c.model == "m"
    assert c._http.headers[NGROK_HEADER] == "true"
    assert c._http.timeout.read == 42 and c._http.timeout.connect == 5


def test_format_think_and_num_predict_are_forwarded():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"response": "{}"})

    make_client(handler).generate("p", format="json", think=False, num_predict=50)
    assert seen["body"]["format"] == "json" and seen["body"]["think"] is False
    assert seen["body"]["options"]["num_predict"] == 50


def test_thinking_is_not_mixed_into_the_answer():
    def handler(request):
        return httpx.Response(200, json={"response": "final", "thinking": "razonamiento largo..."})

    assert make_client(handler).generate("p") == "final"


def test_empty_answer_with_thinking_explains_token_budget():
    def handler(request):
        return httpx.Response(200, json={"response": "", "thinking": "...", "done_reason": "length"})

    with pytest.raises(OllamaError, match="num_predict"):
        make_client(handler).generate("p", num_predict=5)
