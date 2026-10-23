"""Cliente HTTP del servidor de imágenes (image_generator/, en Colab o local).

Variables de entorno:
    IMAGE_API_URL              URL base del servidor de imágenes (p. ej. la de ngrok en Colab)
    IMAGE_NGROK_SKIP_WARNING   opcional: "1"/"true" añade la cabecera ngrok-skip-browser-warning
    IMAGE_TIMEOUT              opcional, segundos (por defecto 300)

Contrato: POST {IMAGE_API_URL}/generate  {"prompt": str, "seed": int|None}
          -> 200 image/png (bytes)
"""

import os

import httpx

from labdino.ollama_client import NGROK_HEADER, env_bool

STYLE = ("wildlife photograph, photorealistic, National Geographic, full body, wide shot, side view, "
         "in its natural habitat, natural daylight, realistic skin texture")


class ImageError(Exception):
    """Error al generar la imagen; el mensaje indica qué revisar."""


def build_image_prompt(identity: dict, name: str | None = None) -> str:
    """Prompt en inglés: nombre (si viene) + frase visual que escribió Ollama + estilo fijo."""
    visual = identity.get("visual_prompt_en", "").strip().rstrip(".")
    if name:
        return f"{name.strip().capitalize()}, a fictional dinosaur species: {visual or 'a fictional dinosaur'}, {STYLE}"
    return f"{visual}, {STYLE}" if visual else f"a fictional dinosaur, {STYLE}"


class ImageClient:
    def __init__(self, base_url: str, *, timeout: float = 300.0, ngrok_skip_warning: bool = False,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        headers = {NGROK_HEADER: "true"} if ngrok_skip_warning else {}
        self._http = httpx.Client(base_url=self.base_url, headers=headers,
                                  timeout=httpx.Timeout(timeout, connect=10.0), transport=transport)

    @classmethod
    def from_env(cls) -> "ImageClient | None":
        """None si IMAGE_API_URL no está definida (la app sigue funcionando sin imagen)."""
        url = os.environ.get("IMAGE_API_URL", "").strip()
        if not url:
            return None
        return cls(url, timeout=float(os.environ.get("IMAGE_TIMEOUT", "300") or 300),
                   ngrok_skip_warning=env_bool("IMAGE_NGROK_SKIP_WARNING", False))

    def generate(self, prompt: str, seed: int | None = None) -> bytes:
        try:
            resp = self._http.post("/generate", json={"prompt": prompt, "seed": seed})
        except httpx.HTTPError as e:
            raise ImageError(f"No se pudo contactar el servidor de imágenes en {self.base_url}: {e}. "
                             "Revisa IMAGE_API_URL y que el notebook de Colab siga activo.") from e
        if resp.status_code >= 400 or not resp.headers.get("content-type", "").startswith("image/"):
            raise ImageError(f"El servidor de imágenes respondió HTTP {resp.status_code} "
                             f"({resp.headers.get('content-type')}): {resp.text[:200]}")
        return resp.content

    def close(self) -> None:
        self._http.close()
