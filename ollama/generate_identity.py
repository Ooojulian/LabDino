"""Identidad del dinosaurio elegido con Ollama.

    uv run python ollama/generate_identity.py --show-prompt        # solo escribe ollama/identity_prompt.md (sin Ollama)
    uv run python ollama/generate_identity.py                      # think según OLLAMA_IDENTITY_THINK (activado por defecto)
    uv run python ollama/generate_identity.py --think off          # fuerza think=False (o --think on)
    ... --out-dir RUTA   # dónde guardar identity.md/identity.json (por defecto, esta carpeta)

Requiere OLLAMA_BASE_URL y OLLAMA_MODEL en el entorno para ejecutar (no para --show-prompt).
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from labdino.identity import (ALL_KEYS, CLEAN_SEEDS, generate_clean_identity, identity_think, HARD_MAX, HARD_MIN, KEYS, MAX_WORDS, MIN_WORDS,
                              SCHEMA, TEMPERATURE, build_prompts)
from labdino.ollama_client import OllamaClient

HERE = Path(__file__).parent
ROOT = HERE.parent
CHOSEN = json.loads((ROOT / "evidence" / "sampling_configs.json").read_text(encoding="utf-8"))["chosen_name"]


def real_names() -> set[str]:
    """Nombres de dinosaurios reales (todo el dataset) para la advertencia de la salida."""
    path = ROOT / "name_generator" / "artifacts" / "train_names.txt"
    return {n for n in path.read_text(encoding="utf-8").splitlines() if n}


def write_prompt_md(name: str) -> None:
    system, user = build_prompts(name)
    text = f"""# Prompt de identidad — {name.capitalize()}

Prompt exacto (lo construye `labdino.identity.build_prompts`) que se enviará a Ollama.

## Parámetros

| Parámetro | Valor |
|---|---|
| Endpoint | `/api/generate` (sin streaming) |
| `format` | esquema JSON (abajo): 7 claves fijas, sin claves extra |
| temperature | {TEMPERATURE} |
| seed | se prueban {', '.join(map(str, CLEAN_SEEDS))} en orden; se elige la primera ficha válida sin advertencias (si ninguna, la de menos advertencias). Se registran las semillas probadas y el motivo en `identity.json` |
| Reintentos | un intento por semilla; el número de intentos y las semillas probadas se registran en `identity.json` |
| Modo thinking | `think=True` por defecto para la identidad (configurable con `OLLAMA_IDENTITY_THINK`); el chat usa `think=False` (`OLLAMA_CHAT_THINK`). El razonamiento llega aparte y no se mezcla con la respuesta |
| Longitud pedida en el prompt | {MIN_WORDS}-{MAX_WORDS} palabras (suma de las 6 claves de la ficha; `visual_prompt_en` no cuenta) |
| Longitud validada | reintento fuera de {HARD_MIN}-{HARD_MAX}; advertencia entre {HARD_MIN}-{MIN_WORDS} y {MAX_WORDS}-{HARD_MAX} |

## Prompt de sistema

```text
{system}
```

## Prompt de usuario

```text
{user}
```

## Esquema JSON (`format`)

```json
{json.dumps(SCHEMA, indent=2, ensure_ascii=False)}
```

## Chequeos automáticos tras la respuesta

**Reintento (error):**
- El JSON parsea y es un objeto con exactamente las claves {", ".join(f"`{k}`" for k in ALL_KEYS)}.
- Ningún campo está vacío y todos son texto.
- La suma de palabras de las 6 claves de la ficha está entre {HARD_MIN} y {HARD_MAX}.
- `resumen_visual` contiene la palabra «dinosaurio».

**Advertencia (no bloquea):**
- Longitud entre {HARD_MIN}-{MIN_WORDS} o {MAX_WORDS}-{HARD_MAX} palabras.
- Aparece en la salida el nombre de una especie real (los {len(real_names())} nombres del dataset más unos célebres como T-Rex o Velociraptor).
- `visual_prompt_en` no menciona «dinosaur» o es demasiado largo.
- La ubicación anatómica del rasgo distintivo (cráneo, nuca, cuello, dorso, cola, hocico) no coincide entre `caracteristica_distintiva`, `resumen_visual` y `visual_prompt_en` (`apariencia` no menciona el rasgo y no se revisa).
- `resumen_visual` o `visual_prompt_en` contienen el nombre propio.
- El resumen visual no comparte vocabulario con la característica distintiva.
"""
    (HERE / "identity_prompt.md").write_text(text, encoding="utf-8")


def run(name: str, think: bool | None, out_dir: Path) -> None:
    client = OllamaClient.from_env()
    health = client.health()
    start = time.perf_counter()
    result = generate_clean_identity(client, name, think=think, real_names=real_names())
    elapsed = time.perf_counter() - start
    system, user = build_prompts(name)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    think = identity_think() if think is None else think
    think_label = "activado" if think else "desactivado"

    record = {
        "name": name,
        "identity": result.identity,
        "model": health["model"],
        "ollama_version": health["ollama_version"],
        "parameters": {"temperature": TEMPERATURE, "seed": result.seed, "think": think, "format": "json_schema",
                       "seeds_to_try": list(CLEAN_SEEDS)},
        "n_attempts": result.n_attempts,
        "seeds_tried": [t["seed"] for t in result.trials],
        "seed_trials": result.trials,
        "chosen_seed": result.seed,
        "choice_reason": result.choice_reason,
        "elapsed_seconds": round(elapsed, 1),
        "prompt": {"system": system, "user": user},
        "checks": result.report,
        "attempts": result.attempts,
        "generated_utc": stamp,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "identity.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [f"# {name.capitalize()}", "", f"*Ficha ficticia generada con `{health['model']}` (Ollama {health['ollama_version']}) "
          f"el {stamp} UTC; temperature {TEMPERATURE}, seed {result.seed}, think {think_label}, "
          f"{result.n_attempts} intento(s), {elapsed:.1f} s.*", ""]
    for key in ALL_KEYS:
        md += [f"## {key.replace('_', ' ').capitalize()}", "", result.identity[key], ""]
    md += ["## Selección de la semilla", "", result.choice_reason, "",
           *(f"- Seed {t['seed']}: " + ("válida, " + (f"{len(t['advertencias'])} advertencia(s): {t['advertencias']}"
              if t["advertencias"] else "sin advertencias") if t["valida"] else f"no válida ({t['errores']})")
             for t in result.trials), ""]
    md += ["## Chequeos", "", f"- Palabras de la ficha: {result.report['palabras']} (pedido {MIN_WORDS}-{MAX_WORDS})",
           "- Claves, campos no vacíos y longitud: OK",
           *(f"- Advertencia: {w}" for w in result.report["advertencias"]), ""]
    (out_dir / "identity.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-prompt", action="store_true", help="solo escribe identity_prompt.md")
    ap.add_argument("--think", choices=["env", "on", "off"], default="env")
    ap.add_argument("--name", default=CHOSEN)
    ap.add_argument("--out-dir", type=Path, default=HERE)
    args = ap.parse_args()
    if args.show_prompt:
        write_prompt_md(args.name)
    else:
        run(args.name, {"env": None, "on": True, "off": False}[args.think], args.out_dir)
