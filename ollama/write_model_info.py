"""Genera ollama/model_info.md leyendo /api/version, /api/tags y /api/show (nada se escribe a mano).

Uso (con OLLAMA_BASE_URL y OLLAMA_MODEL en el entorno):
    uv run python ollama/write_model_info.py
Se puede reejecutar en SageMaker para registrar el modelo allí. No se escribe la URL base en el archivo.
"""

from datetime import datetime, timezone
from pathlib import Path

from labdino.ollama_client import OllamaClient

OUT = Path(__file__).with_name("model_info.md")


def main() -> None:
    client = OllamaClient.from_env()
    health = client.health()
    if not health["model_available"]:
        raise SystemExit(f"El modelo {client.model!r} no está instalado. Disponibles: {health['available_models']}")
    show = client.show()
    tags = next(m for m in client.list_models() if m["name"] in {client.model, f"{client.model}:latest"})
    details = show.get("details", {})

    rows = [
        ("Modelo (nombre exacto)", tags["name"]),
        ("Digest", tags["digest"]),
        ("Tamaño en disco (bytes)", tags["size"]),
        ("Familia", details.get("family")),
        ("Parámetros", details.get("parameter_size")),
        ("Cuantización", details.get("quantization_level")),
        ("Formato", details.get("format")),
        ("Capacidades", ", ".join(show.get("capabilities", [])) or None),
        ("Versión de Ollama", health["ollama_version"]),
        ("Registrado (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")),
    ]
    lines = ["# Modelo de Ollama utilizado", "", "Generado por `ollama/write_model_info.py` a partir de la API de Ollama.", "",
             "| Campo | Valor |", "|---|---|", *(f"| {k} | {v} |" for k, v in rows if v is not None), ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
