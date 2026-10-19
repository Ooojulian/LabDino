"""Verificación real contra Ollama; se omite si OLLAMA_BASE_URL u OLLAMA_MODEL no están definidas."""

import os

import pytest

from labdino.ollama_client import OllamaClient

pytestmark = pytest.mark.skipif(
    not (os.environ.get("OLLAMA_BASE_URL") and os.environ.get("OLLAMA_MODEL")),
    reason="Define OLLAMA_BASE_URL y OLLAMA_MODEL para la verificación real",
)


def test_live_health_generate_and_chat():
    client = OllamaClient.from_env()
    health = client.health()
    print("\nhealth:", {k: v for k, v in health.items() if k != "available_models"})
    assert health["model_available"], f"{client.model} no está instalado: {health['available_models']}"

    out = client.generate("Responde únicamente con la palabra: OK", temperature=0.0, seed=1)
    print("generate ->", repr(out))
    assert out.strip()

    out = client.chat([{"role": "user", "content": "¿Cuánto es 2+2? Responde solo con el número."}],
                      system="Eres un asistente breve.", temperature=0.0, seed=1)
    print("chat ->", repr(out))
    assert out.strip()
