import json

import httpx
import pytest

from labdino.image_client import STYLE, ImageClient, ImageError, build_image_prompt
from labdino.ollama_client import NGROK_HEADER

BASE = "http://img.test"


def make_client(handler, **kwargs) -> ImageClient:
    return ImageClient(BASE, transport=httpx.MockTransport(handler), **kwargs)


def test_build_image_prompt_uses_visual_prompt_and_style():
    out = build_image_prompt({"visual_prompt_en": "A brown dinosaur."})
    assert out == f"A brown dinosaur, {STYLE}"


def test_build_image_prompt_fallback():
    assert build_image_prompt({}).startswith("a fictional dinosaur, ")


def test_generate_success_and_payload():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"})

    assert make_client(handler).generate("un dino", seed=5) == b"\x89PNG"
    assert seen["url"] == f"{BASE}/generate"
    assert seen["body"] == {"prompt": "un dino", "seed": 5}


def test_ngrok_header_only_when_requested():
    seen = []

    def handler(request):
        seen.append(request.headers.get(NGROK_HEADER))
        return httpx.Response(200, content=b"x", headers={"content-type": "image/png"})

    make_client(handler, ngrok_skip_warning=True).generate("p")
    make_client(handler).generate("p")
    assert seen == ["true", None]


def test_http_error_raises_image_error():
    with pytest.raises(ImageError, match="HTTP 500"):
        make_client(lambda r: httpx.Response(500, text="boom")).generate("p")


def test_non_image_content_type_raises():
    with pytest.raises(ImageError, match="HTTP 200"):
        make_client(lambda r: httpx.Response(200, json={"ok": 1})).generate("p")


def test_connection_error_raises_image_error():
    def handler(request):
        raise httpx.ConnectError("sin red")

    with pytest.raises(ImageError, match="IMAGE_API_URL"):
        make_client(handler).generate("p")


def test_from_env_none_without_url(monkeypatch):
    monkeypatch.delenv("IMAGE_API_URL", raising=False)
    assert ImageClient.from_env() is None


def test_from_env_reads_settings(monkeypatch):
    monkeypatch.setenv("IMAGE_API_URL", "http://x.test/")
    monkeypatch.setenv("IMAGE_NGROK_SKIP_WARNING", "true")
    monkeypatch.setenv("IMAGE_TIMEOUT", "12")
    client = ImageClient.from_env()
    assert client.base_url == "http://x.test"
    assert client._http.headers[NGROK_HEADER] == "true"
    assert client._http.timeout.read == 12
