from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import server
from labdino.chat import system_prompt
from labdino.image_client import ImageError
from labdino.ollama_client import OllamaError


class FakeOllama:
    model = "fake-model"
    calls: list = []

    def chat(self, messages, system=None, **kwargs):
        FakeOllama.calls.append({"messages": messages, "system": system})
        return "respuesta falsa"

    def close(self):
        pass


@pytest.fixture
def client(monkeypatch, tmp_path):
    FakeOllama.calls = []
    monkeypatch.setattr(server, "ollama", lambda: FakeOllama())
    monkeypatch.setattr(server, "IMAGES", tmp_path)  # no ensucia app/data/images
    monkeypatch.delenv("IMAGE_API_URL", raising=False)
    server.load_seed_dino()
    return TestClient(server.app)


def test_info(client):
    data = client.get("/api/info").json()["name_model"]
    assert data["nombres_entrenamiento"] == len(server.art.known_names) > 0
    assert data["T"] == server.art.max_len


def test_dino_is_seed(client):
    d = client.get("/api/dino").json()
    assert d["id"] == "doractodon" and d["name"] == "Doractodon"
    assert d["identity"]["resumen_visual"] and d["greeting"].startswith("¡Hola! Soy Doractodon")


def test_new_dino_generates_novel_name_without_retraining(client, monkeypatch):
    identity = {"apariencia": "a", "habitat": "h", "alimentacion": "al", "comportamiento": "c",
                "caracteristica_distintiva": "d", "resumen_visual": "r", "visual_prompt_en": "v"}
    monkeypatch.setattr(server, "generate_clean_identity", lambda c, name, **kw: SimpleNamespace(
        identity=identity, seed=7, choice_reason="limpia"))
    r = client.post("/api/dino/new")
    assert r.status_code == 200
    d = r.json()
    assert d["name"].lower() not in server.art.known_names
    assert d["image_url"] is None and d["meta"]["ollama_model"] == "fake-model"
    assert client.get("/api/dino").json()["id"] == d["id"]  # pasa a ser el actual


def test_new_dino_image_error_is_reported(client, monkeypatch):
    identity = {"resumen_visual": "r"}
    monkeypatch.setattr(server, "generate_clean_identity", lambda c, name, **kw: SimpleNamespace(
        identity=identity, seed=1, choice_reason="x"))

    def boom(*a, **k):
        raise ImageError("sin servidor de imágenes")

    monkeypatch.setattr(server, "make_image", boom)
    d = client.post("/api/dino/new").json()
    assert d["image_url"] is None and "sin servidor" in d["image_error"]


def test_new_dino_ollama_error_is_502(client, monkeypatch):
    def boom(c, name, **kw):
        raise OllamaError("caído")

    monkeypatch.setattr(server, "generate_clean_identity", boom)
    assert client.post("/api/dino/new").status_code == 502


def test_chat_returns_image_only_when_asked(client):
    dino = server.dinos["doractodon"]
    dino["image_url"] = "/images/doractodon.png"
    body = lambda text: {"dino_id": "doractodon", "messages": [{"role": "user", "content": text}]}
    r = client.post("/api/chat", json=body("Preséntate y descríbete")).json()
    assert r == {"reply": "respuesta falsa", "image_url": "/images/doractodon.png"}
    r = client.post("/api/chat", json=body("¿Qué comes?")).json()
    assert r["image_url"] is None


def test_chat_system_prompt_includes_sheet(client):
    client.post("/api/chat", json={"dino_id": "doractodon", "messages": [{"role": "user", "content": "¿Qué comes?"}]})
    dino = server.dinos["doractodon"]
    system = FakeOllama.calls[-1]["system"]
    assert system == system_prompt(dino["name"], dino["identity"])
    assert dino["identity"]["alimentacion"] in system


def test_chat_validation(client):
    assert client.post("/api/chat", json={"dino_id": "nope", "messages": []}).status_code == 404
    bad = {"dino_id": "doractodon", "messages": [{"role": "assistant", "content": "hola"}]}
    assert client.post("/api/chat", json=bad).status_code == 400


def test_image_endpoint(client, tmp_path):
    (tmp_path / "abc.png").write_bytes(b"png")
    assert client.get("/images/abc.png").content == b"png"
    assert client.get("/images/zzz.png").status_code == 404
