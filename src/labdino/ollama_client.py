"""Cliente HTTP mínimo para Ollama (sin streaming).

La configuración sale del entorno; el repo no trae valores por defecto:
    OLLAMA_BASE_URL              URL base de Ollama (p. ej. la de tu máquina o la de ngrok)
    OLLAMA_MODEL                 nombre del modelo (p. ej. el que descargaste con `ollama pull`)
    OLLAMA_NGROK_SKIP_WARNING    opcional: "1"/"true" añade la cabecera ngrok-skip-browser-warning
    OLLAMA_CONNECT_TIMEOUT       opcional, segundos (por defecto 5)
    OLLAMA_READ_TIMEOUT          opcional, segundos (por defecto 300)
    OLLAMA_IDENTITY_THINK        opcional: razonamiento al generar la identidad (por defecto activado)
    OLLAMA_CHAT_THINK            opcional: razonamiento en el chat (por defecto desactivado)
"""

import os

import httpx

NGROK_HEADER = "ngrok-skip-browser-warning"


class OllamaError(Exception):
    """Error al hablar con Ollama; el mensaje indica qué revisar."""


class OllamaConfigError(OllamaError):
    pass


class OllamaConnectionError(OllamaError):
    pass


class OllamaTimeoutError(OllamaError):
    pass


class OllamaModelNotFoundError(OllamaError):
    pass


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise OllamaConfigError(f"Falta la variable de entorno {name}. Defínela en tu .env (ver .env.example).")
    return value


def env_bool(name: str, default: bool) -> bool:
    """Lee un booleano del entorno ("1/true/yes/on" o "0/false/no/off"); `default` si no está definido."""
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise OllamaConfigError(f"{name} debe ser true o false, no {value!r}.")


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        connect_timeout: float = 5.0,
        read_timeout: float = 300.0,
        temperature: float | None = None,
        seed: int | None = None,
        ngrok_skip_warning: bool = False,
        chat_think: bool = False,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.chat_think = chat_think  # el chat responde rápido: sin razonamiento salvo que se pida lo contrario
        headers = {NGROK_HEADER: "true"} if ngrok_skip_warning else {}
        self._http = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(connect=connect_timeout, read=read_timeout, write=30.0, pool=connect_timeout),
            transport=transport,
        )

    @classmethod
    def from_env(cls, **kwargs) -> "OllamaClient":
        """Crea el cliente con OLLAMA_BASE_URL y OLLAMA_MODEL (obligatorias)."""
        kwargs.setdefault("chat_think", env_bool("OLLAMA_CHAT_THINK", False))
        kwargs.setdefault("ngrok_skip_warning", env_bool("OLLAMA_NGROK_SKIP_WARNING", False))
        for env, key in (("OLLAMA_CONNECT_TIMEOUT", "connect_timeout"), ("OLLAMA_READ_TIMEOUT", "read_timeout")):
            if os.environ.get(env, "").strip() and key not in kwargs:
                try:
                    kwargs[key] = float(os.environ[env])
                except ValueError:
                    raise OllamaConfigError(f"{env} debe ser un número de segundos, no {os.environ[env]!r}.") from None
        return cls(_require_env("OLLAMA_BASE_URL"), _require_env("OLLAMA_MODEL"), **kwargs)

    def close(self) -> None:
        self._http.close()

    # --- HTTP ---------------------------------------------------------------------------------------------

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        try:
            resp = self._http.request(method, path, json=payload)
        except httpx.ConnectTimeout as e:
            raise OllamaConnectionError(
                f"Tiempo de conexión agotado con {self.base_url}. Revisa OLLAMA_BASE_URL y que Ollama/ngrok estén activos."
            ) from e
        except httpx.ConnectError as e:
            raise OllamaConnectionError(
                f"No se pudo conectar con Ollama en {self.base_url}. Revisa OLLAMA_BASE_URL y que el servicio esté activo."
            ) from e
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(
                f"Ollama no respondió a tiempo ({path}). El modelo puede estar cargándose o la instancia ser lenta: "
                "aumenta OLLAMA_READ_TIMEOUT."
            ) from e
        except httpx.HTTPError as e:
            raise OllamaConnectionError(f"Error de red con Ollama ({self.base_url}): {e}. Revisa OLLAMA_BASE_URL.") from e

        if resp.status_code >= 400:
            self._raise_for_status(resp)
        try:
            return resp.json()
        except ValueError as e:
            raise OllamaError(
                f"Respuesta no válida de {self.base_url}{path} (no es JSON). ¿Es realmente un servidor Ollama? "
                "Revisa OLLAMA_BASE_URL; con ngrok activa OLLAMA_NGROK_SKIP_WARNING=1."
            ) from e

    def _raise_for_status(self, resp: httpx.Response) -> None:
        try:
            detail = resp.json().get("error", "")
        except ValueError:
            detail = resp.text[:200]
        if resp.status_code == 404 and "model" in detail.lower():
            raise OllamaModelNotFoundError(
                f"El modelo {self.model!r} no existe en Ollama ({detail}). Revisa OLLAMA_MODEL o descárgalo con "
                f"`ollama pull {self.model}`."
            )
        hint = " Revisa OLLAMA_BASE_URL." if resp.status_code in (401, 403, 404, 502, 503) else ""
        raise OllamaError(f"Ollama respondió HTTP {resp.status_code}: {detail or 'sin detalle'}.{hint}")

    def _final_text(self, text: str, data: dict) -> str:
        """Solo el contenido final; el razonamiento (`thinking`) llega aparte y nunca se mezcla."""
        if not text.strip():
            hint = (
                " El razonamiento parece haber agotado el límite de tokens: sube num_predict o desactiva think."
                if data.get("thinking") or data.get("message", {}).get("thinking")
                else ""
            )
            raise OllamaError(f"Ollama devolvió una respuesta vacía (done_reason={data.get('done_reason')!r}).{hint}")
        return text

    def _options(self, temperature: float | None, seed: int | None, num_predict: int | None = None) -> dict:
        temperature = self.temperature if temperature is None else temperature
        seed = self.seed if seed is None else seed
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if seed is not None:
            options["seed"] = seed
        if num_predict is not None:
            options["num_predict"] = num_predict
        return options

    # --- API ----------------------------------------------------------------------------------------------

    def version(self) -> str:
        return self._request("GET", "/api/version")["version"]

    def list_models(self) -> list[dict]:
        """Modelos instalados (/api/tags): nombre, digest, tamaño y detalles."""
        return self._request("GET", "/api/tags").get("models", [])

    def show(self) -> dict:
        """Información del modelo configurado (/api/show)."""
        return self._request("POST", "/api/show", {"model": self.model})

    def health(self) -> dict:
        """Versión de Ollama y si el modelo configurado está disponible."""
        names = [m["name"] for m in self.list_models()]
        wanted = {self.model, f"{self.model}:latest"}
        return {
            "ollama_version": self.version(),
            "model": self.model,
            "model_available": any(n in wanted for n in names),
            "available_models": names,
        }

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        *,
        temperature: float | None = None,
        seed: int | None = None,
        format: str | dict | None = None,
        think: bool | None = None,
        num_predict: int | None = None,
    ) -> str:
        """Texto final. `format` ("json" o un esquema JSON) fuerza salida estructurada; `think` activa/desactiva
        el razonamiento del modelo (None = comportamiento por defecto de Ollama)."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": self._options(temperature, seed, num_predict),
        }
        if system:
            payload["system"] = system
        if format is not None:
            payload["format"] = format
        if think is not None:
            payload["think"] = think
        data = self._request("POST", "/api/generate", payload)
        return self._final_text(data["response"], data)

    def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        *,
        temperature: float | None = None,
        seed: int | None = None,
        format: str | dict | None = None,
        think: bool | None = None,
        num_predict: int | None = None,
    ) -> str:
        """messages: [{"role": "user"|"assistant", "content": ...}]; `system` se antepone como mensaje de sistema."""
        full = ([{"role": "system", "content": system}] if system else []) + list(messages)
        payload = {
            "model": self.model,
            "messages": full,
            "stream": False,
            "options": self._options(temperature, seed, num_predict),
        }
        if format is not None:
            payload["format"] = format
        payload["think"] = self.chat_think if think is None else think
        data = self._request("POST", "/api/chat", payload)
        return self._final_text(data["message"]["content"], data)
