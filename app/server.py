"""Backend de LabDino: Modelo de caracteres → Nombre → Ollama → Identidad → Imagen → Chat.

Ejecutar (desde la raíz del repo):
    set -a; source .env; set +a
    uv run uvicorn app.server:app --host 0.0.0.0 --port 8000

Endpoints (todo JSON salvo las imágenes):
    GET  /api/info                -> datos del modelo de nombres y de Ollama
    GET  /api/dino                -> dinosaurio actual
    POST /api/dino/new            -> genera otro (nombre → identidad → imagen), SIN reentrenar
    POST /api/chat  {dino_id, messages:[{role, content}]} -> {reply, image_url|null}
    GET  /images/{dino_id}.png    -> imagen (generada una sola vez por dinosaurio)
"""

import json
import random
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from labdino.artifacts import generate_novel_name, load_artifacts
from labdino.chat import greeting, system_prompt, wants_image
from labdino.identity import generate_clean_identity
from labdino.image_client import ImageClient, ImageError, build_image_prompt
from labdino.ollama_client import OllamaClient, OllamaError

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "name_generator" / "artifacts"
IMAGES = ROOT / "app" / "data" / "images"
IMAGES.mkdir(parents=True, exist_ok=True)
SEED_DINO = ROOT / "ollama" / "identity.json"  # Doractodon: el dinosaurio de las secciones 2-4
SEED_IMAGE = ROOT / "image_generator" / "outputs" / "doractodon.png"

art = load_artifacts(ARTIFACTS)  # se carga una vez; el modelo nunca se reentrena aquí
dinos: dict[str, dict] = {}
current: dict = {"id": None}
lock = threading.Lock()  # una generación a la vez (Ollama en m5.xlarge es lento)

app = FastAPI(title="LabDino")


def ollama() -> OllamaClient:
    return OllamaClient.from_env()


def name_model_info() -> dict:
    c = art.config
    return {
        "arquitectura": c["architecture"], "capas": c["num_layers"], "hidden_size": c["hidden_size"],
        "dropout": c["dropout"], "T": art.max_len, "vocabulario": len(art.vocab.itos),
        "muestreo": c["default_sampling"], "seleccion": c.get("selection"),
        "nombres_entrenamiento": len(art.known_names),
    }


def make_image(dino_id: str, name: str, identity: dict) -> str | None:
    """Genera la imagen UNA vez y la guarda en disco; devuelve su URL o None si no hay servidor."""
    path = IMAGES / f"{dino_id}.png"
    if path.exists():
        return f"/images/{dino_id}.png"
    client = ImageClient.from_env()
    if client is None:
        return None
    try:
        path.write_bytes(client.generate(build_image_prompt(identity, name), seed=random.randint(0, 2**31 - 1)))
    finally:
        client.close()
    return f"/images/{dino_id}.png"


def register(name: str, identity: dict, meta: dict, image_url: str | None, image_error: str | None = None) -> dict:
    dino_id = uuid.uuid4().hex[:12] if not meta.get("id") else meta["id"]
    dino = {"id": dino_id, "name": name.capitalize(), "identity": identity, "meta": meta,
            "image_url": image_url, "image_error": image_error, "greeting": greeting(name, identity)}
    dinos[dino_id] = dino
    current["id"] = dino_id
    return dino


def load_seed_dino() -> None:
    if not SEED_DINO.exists():
        return
    data = json.loads(SEED_DINO.read_text(encoding="utf-8"))
    dino_id = "doractodon"
    if SEED_IMAGE.exists() and not (IMAGES / f"{dino_id}.png").exists():
        (IMAGES / f"{dino_id}.png").write_bytes(SEED_IMAGE.read_bytes())
    url = f"/images/{dino_id}.png" if (IMAGES / f"{dino_id}.png").exists() else None
    register(data["name"], data["identity"],
             {"id": dino_id, "ollama_model": data.get("model"), "seed": data["parameters"]["seed"]}, url)


load_seed_dino()


@app.get("/api/info")
def info():
    return {"name_model": name_model_info()}


@app.get("/api/dino")
def get_dino():
    if not current["id"]:
        raise HTTPException(404, "Aún no hay dinosaurio: pulsa Nuevo Dinosaurio.")
    return dinos[current["id"]]


@app.post("/api/dino/new")
def new_dino():
    if not lock.acquire(blocking=False):
        raise HTTPException(409, "Ya se está creando un dinosaurio; espera a que termine.")
    try:
        name = generate_novel_name(art)  # inferencia con el modelo ya entrenado
        client = ollama()
        base = random.randint(1, 10**6)
        try:
            res = generate_clean_identity(client, name, seeds=(base, base + 1, base + 2), real_names=art.known_names)
        finally:
            client.close()
        dino_id = uuid.uuid4().hex[:12]
        meta = {"id": dino_id, "ollama_model": client.model, "seed": res.seed, "eleccion": res.choice_reason}
        try:
            url, err = make_image(dino_id, name, res.identity), None
        except ImageError as e:
            url, err = None, str(e)
        return register(name, res.identity, meta, url, err)
    except OllamaError as e:
        raise HTTPException(502, str(e)) from e
    finally:
        lock.release()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    dino_id: str
    messages: list[Message]


@app.post("/api/chat")
def chat(req: ChatRequest):
    dino = dinos.get(req.dino_id)
    if dino is None:
        raise HTTPException(404, "Dinosaurio desconocido; recarga la página.")
    msgs = [m.model_dump() for m in req.messages if m.role in ("user", "assistant")][-20:]
    if not msgs or msgs[-1]["role"] != "user":
        raise HTTPException(400, "El último mensaje debe ser del usuario.")
    client = ollama()
    try:
        reply = client.chat(msgs, system_prompt(dino["name"], dino["identity"]), temperature=0.7)
    except OllamaError as e:
        raise HTTPException(502, str(e)) from e
    finally:
        client.close()
    return {"reply": reply, "image_url": dino["image_url"] if wants_image(msgs[-1]["content"]) else None}


@app.get("/images/{dino_id}.png")
def image(dino_id: str):
    path = IMAGES / f"{Path(dino_id).name}.png"
    if not path.exists():
        raise HTTPException(404, "Sin imagen")
    return FileResponse(path, media_type="image/png")


app.mount("/", StaticFiles(directory=ROOT / "app" / "static", html=True), name="static")
