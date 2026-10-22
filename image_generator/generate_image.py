"""Genera la imagen de un dinosaurio con amused/amused-512 (diffusers).

Uso (en Colab o una máquina con GPU; en CPU funciona pero tarda varios minutos):
    pip install -r image_generator/requirements.txt
    python image_generator/generate_image.py                       # usa ollama/identity.json (Doractodon)
    python image_generator/generate_image.py --identity otra.json --out salida.png --seed 7

Entrada: identity.json producido por ollama/generate_identity.py (usa visual_prompt_en).
Salida:  PNG 512x512 en image_generator/outputs/<nombre>.png
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from diffusers import AmusedPipeline

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "amused/amused-512"
STYLE = ("full body, side view, paleontological illustration, natural lighting, detailed skin texture, "
         "plain background, no text")
NEGATIVE = "text, watermark, low quality, blurry, deformed, extra limbs, human"


def load_pipeline() -> AmusedPipeline:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    kwargs = {"variant": "fp16"} if device == "cuda" else {}
    return AmusedPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, **kwargs).to(device)


def build_prompt(identity: dict) -> str:
    visual = identity.get("visual_prompt_en", "").strip().rstrip(".")
    return f"{visual}, {STYLE}" if visual else f"a fictional dinosaur, {STYLE}"


def generate(pipe: AmusedPipeline, prompt: str, seed: int | None = 0, steps: int = 12):
    gen = torch.Generator(pipe.device).manual_seed(seed) if seed is not None else None
    return pipe(prompt, negative_prompt=NEGATIVE, generator=gen, num_inference_steps=steps,
                guidance_scale=10.0).images[0]


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
         "steps": 12, "guidance_scale": 10.0, "resolution": "512x512"}, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
