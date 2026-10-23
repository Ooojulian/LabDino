"""Genera la imagen fotorrealista de un dinosaurio con stabilityai/sd-turbo (diffusers).

Modelo: sd-turbo en vez de amused-512 (la guía permite 'otra alternativa de tamaño similar'). Razones: (1) más realista:
da fotografía de vida salvaje, mientras amused sale caricaturesco/plano; (2) tamaño similar (~1B de parámetros, ~2 GB
en fp16); (3) corre en CPU: 512x512 con 1-4 pasos y guidance 0 (destilado), ~50 s en 4 vCPU. Por eso el prompt negativo
no se usa en la inferencia (guidance 0) y solo se registra en el json.

Uso (Colab/GPU: segundos; CPU de 4 vCPU: ~50 s):
    pip install -r image_generator/requirements.txt
    python image_generator/generate_image.py                       # usa ollama/identity.json (Doractodon)
    python image_generator/generate_image.py --identity otra.json --out salida.png --seed 7

Entrada: identity.json producido por ollama/generate_identity.py (usa visual_prompt_en).
Salida:  PNG 512x512 en image_generator/outputs/<nombre>.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "stabilityai/sd-turbo"
STYLE = ("wildlife photograph, photorealistic, National Geographic, full body, wide shot, side view, "
         "dry open landscape, natural daylight, realistic skin texture")
NEGATIVE = "cartoon, illustration, drawing, 3d render, anime, toy, cgi, painting, text, watermark"  # sd-turbo (guidance 0) lo ignora; se registra en el json
STEPS = 4
GUIDANCE = 0.0


def load_pipeline() -> AutoPipelineForText2Image:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    kwargs = {"variant": "fp16"} if device == "cuda" else {}
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    return AutoPipelineForText2Image.from_pretrained(MODEL_ID, torch_dtype=dtype, **kwargs).to(device)


def build_prompt(identity: dict) -> str:
    visual = identity.get("visual_prompt_en", "").strip().rstrip(".")
    return f"{visual}, {STYLE}" if visual else f"a fictional dinosaur, {STYLE}"


def generate(pipe: AutoPipelineForText2Image, prompt: str, seed: int | None = 0, steps: int = STEPS):
    gen = torch.Generator(pipe.device).manual_seed(seed) if seed is not None else None
    return pipe(prompt, generator=gen, num_inference_steps=steps, guidance_scale=GUIDANCE,
                height=512, width=512).images[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--identity", type=Path, default=ROOT / "ollama" / "identity.json")
    p.add_argument("--out", type=Path)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    data = json.loads(a.identity.read_text(encoding="utf-8"))
    prompt = build_prompt(data["identity"])
    out = a.out or ROOT / "image_generator" / "outputs" / f"{data['name']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    print("Prompt:", prompt, file=sys.stderr)
    generate(load_pipeline(), prompt, a.seed).save(out)
    (out.with_suffix(".json")).write_text(json.dumps(
        {"name": data["name"], "model": MODEL_ID, "prompt": prompt, "negative_prompt": NEGATIVE, "seed": a.seed,
         "steps": STEPS, "guidance_scale": GUIDANCE, "resolution": "512x512"}, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
