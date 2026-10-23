"""Servidor HTTP de imágenes (sd-turbo). Mismo contrato que el notebook de Colab.

    POST /generate {"prompt": str, "seed": int|null} -> image/png
    GET  /health -> {"ok": true, "model": ...}

Ejecutar: python image_generator/server.py   (puerto 8001; en SageMaker lo arranca sagemaker/on-start.sh)
"""

import io
import sys
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_image import MODEL_ID, generate, load_pipeline  # noqa: E402

pipe = load_pipeline()
gpu = threading.Lock()  # una imagen a la vez
api = FastAPI()


class Req(BaseModel):
    prompt: str
    seed: int | None = None


@api.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID}


@api.post("/generate")
def gen(req: Req):
    with gpu:
        im = generate(pipe, req.prompt[:400], seed=req.seed)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8001)
