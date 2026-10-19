"""Identidad del dinosaurio: prompt, esquema JSON, validación y generación con reintentos."""

import json
import re
from dataclasses import dataclass, field

from labdino.ollama_client import OllamaClient, OllamaError, env_bool

KEYS = ["apariencia", "habitat", "alimentacion", "comportamiento", "caracteristica_distintiva", "resumen_visual"]
VISUAL_PROMPT_KEY = "visual_prompt_en"  # frase corta en inglés para el generador de imágenes; fuera del conteo
ALL_KEYS = KEYS + [VISUAL_PROMPT_KEY]
MIN_WORDS, MAX_WORDS = 150, 200  # lo que se pide en el prompt
HARD_MIN, HARD_MAX = 120, 230  # fuera de este rango se reintenta; entre ambos rangos solo hay advertencia
TEMPERATURE, BASE_SEED, MAX_ATTEMPTS = 0.7, 42, 4
SUFFIXES = {"-saurus": "lagarto", "-raptor": "ladrón o rapaz", "-odon": "diente", "-venator": "cazador"}
# Nombres comunes de dinosaurios reales que además de los del dataset se buscan en la salida (advertencia)
FAMOUS_REAL = {"t-rex", "tiranosaurio", "tyrannosaurus", "velociraptor", "triceratops", "stegosaurus", "estegosaurio",
               "brontosaurus", "brontosaurio", "diplodocus", "spinosaurus", "allosaurus", "carnotaurus"}

SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "string"} for k in ALL_KEYS},
    "required": ALL_KEYS,
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
Eres un paleontólogo y escritor de fichas de especies ficticias. Escribes en español, en tercera persona y en \
presente, con un tono científico claro. Respondes SIEMPRE y únicamente con un objeto JSON válido, sin texto \
adicional, sin markdown y sin comentarios.

Principios de tu trabajo:
- Coherencia: todos los rasgos de la especie encajan entre sí. El hábitat explica el cuerpo, el cuerpo explica la \
dieta y la dieta explica el comportamiento.
- La especie es inventada: no copies ni parafrasees ninguna especie real, no menciones ni compares con especies \
reales, y no afirmes hallazgos fósiles ni lugares de excavación reales.
- El nombre de la especie es un nombre propio inventado: lo usas tal cual, sin corregirlo ni sustituirlo."""

USER_TEMPLATE = """\
Crea la ficha de una especie de dinosaurio ficticia llamada «{name}».

«{name}» es un nombre propio inventado: no lo corrijas, no lo sustituyas por otro ni lo reinterpretes como una \
especie real.

Contexto sobre sufijos de nombres reales (solo orientación, no plantilla):
{suffix_context}
{relevant_suffix}
No repitas estas pistas en la ficha ni conviertas el nombre en un chiste.

Antes de escribir, define mentalmente un perfil coherente: hábitat, tamaño, locomoción, dentición, dieta, \
comportamiento y UN rasgo físico distintivo visible. Después escribe la ficha con estas claves, todas en texto \
(cadena de caracteres), con esta orientación de extensión:
- "apariencia": tamaño, postura, cuerpo, piel o plumas y colores (unas 40 palabras).
- "habitat": clima, terreno y vegetación donde vive (unas 25 palabras).
- "alimentacion": qué come y cómo lo obtiene, acorde con su dentición y su cuerpo (unas 25 palabras).
- "comportamiento": hábitos sociales, actividad y relación con otros animales (unas 30 palabras).
- "caracteristica_distintiva": un único rasgo FÍSICO y visible e inconfundible (por ejemplo la forma, textura o \
color de una parte del cuerpo), no un hábito ni una conducta (unas 25 palabras).
- "resumen_visual": una sola oración que empiece describiendo al dinosaurio y use la palabra «dinosaurio»; solo lo \
que se vería en una imagen, con términos concretos (formas, colores, texturas, tamaño aproximado, postura, \
entorno), incluyendo la característica distintiva y sin compararlo con especies reales (unas 30 palabras).
- "visual_prompt_en": una frase corta en inglés (máximo 35 palabras), pensada para un generador de imágenes, que \
resuma el aspecto del dinosaurio con términos concretos; no cuenta para el límite de palabras.

Reglas:
- La ficha (suma de las seis primeras claves, sin "visual_prompt_en") debe tener entre 150 y 200 palabras.
- Tercera persona y presente; sin listas, sin saltos de línea dentro de los campos.
- Ningún campo puede quedar vacío y no añadas claves distintas a las siete indicadas."""


def build_prompts(name: str) -> tuple[str, str]:
    """(system, user) para el nombre dado. El sufijo relevante se detecta por el final del nombre."""
    context = "\n".join(f"- {suf}: «{meaning}»" for suf, meaning in SUFFIXES.items())
    relevant = next((suf for suf in SUFFIXES if name.lower().endswith(suf.lstrip("-"))), None)
    if relevant:
        others = ", ".join(s for s in SUFFIXES if s != relevant)
        relevant_text = (f"El sufijo relevante para este nombre es {relevant} («{SUFFIXES[relevant]}»); "
                         f"{others} son solo contexto.")
    else:
        relevant_text = "Ninguno de estos sufijos aparece en el nombre; son solo contexto."
    user = USER_TEMPLATE.format(name=name.capitalize(), suffix_context=context, relevant_suffix=relevant_text)
    return SYSTEM_PROMPT, user


def count_words(text: str) -> int:
    return len(re.findall(r"\w+(?:[-']\w+)*", text, flags=re.UNICODE))


def find_real_names(text: str, real_names: set[str]) -> list[str]:
    """Nombres de especies reales (del dataset y célebres) que aparecen como palabra en el texto."""
    words = set(re.findall(r"[\w-]+", text.lower()))
    return sorted(words & (real_names | FAMOUS_REAL))


def validate_identity(raw: str, real_names: set[str] = frozenset()) -> tuple[dict | None, dict]:
    """Parsea y revisa la respuesta. Devuelve (identidad o None, informe de chequeos).

    Errores (reintento): JSON inválido, claves faltantes/sobrantes, campos vacíos, longitud fuera de 120-230
    palabras, resumen visual sin «dinosaurio». Advertencias (no bloquean): longitud entre 120-150 o 200-230,
    nombres de especies reales en la salida, resumen sin el vocabulario del rasgo distintivo.
    """
    report = {"json_valido": False, "claves_ok": False, "campos_no_vacios": False, "palabras": None,
              "longitud_ok": False, "errores": [], "advertencias": []}
    try:
        data = json.loads(raw)
    except ValueError as e:
        report["errores"].append(f"JSON inválido: {e}")
        return None, report
    if not isinstance(data, dict):
        report["errores"].append("La respuesta no es un objeto JSON")
        return None, report
    report["json_valido"] = True

    missing, extra = [k for k in ALL_KEYS if k not in data], [k for k in data if k not in ALL_KEYS]
    report["claves_ok"] = not missing and not extra
    if missing:
        report["errores"].append(f"Faltan claves: {missing}")
    if extra:
        report["errores"].append(f"Claves no permitidas: {extra}")
    if missing:
        return None, report

    bad = [k for k in ALL_KEYS if not isinstance(data[k], str) or not data[k].strip()]
    report["campos_no_vacios"] = not bad
    if bad:
        report["errores"].append(f"Campos vacíos o que no son texto: {bad}")
        return None, report

    identity = {k: data[k].strip() for k in ALL_KEYS}
    n = report["palabras"] = sum(count_words(identity[k]) for k in KEYS)  # visual_prompt_en no cuenta
    report["longitud_ok"] = HARD_MIN <= n <= HARD_MAX
    if not report["longitud_ok"]:
        report["errores"].append(f"Longitud total {n} palabras (límite de reintento {HARD_MIN}-{HARD_MAX})")
    elif not MIN_WORDS <= n <= MAX_WORDS:
        report["advertencias"].append(f"Longitud {n} palabras fuera de lo pedido ({MIN_WORDS}-{MAX_WORDS}) pero aceptable")

    if "dinosaurio" not in identity["resumen_visual"].lower():
        report["errores"].append("resumen_visual no menciona «dinosaurio»")
    if "dinosaur" not in identity[VISUAL_PROMPT_KEY].lower():
        report["advertencias"].append("visual_prompt_en no menciona «dinosaur»")
    if count_words(identity[VISUAL_PROMPT_KEY]) > 45:
        report["advertencias"].append("visual_prompt_en es más largo de lo pedido")

    found = find_real_names(" ".join(identity.values()), set(real_names))
    if found:
        report["advertencias"].append(f"Aparecen nombres de especies reales: {found}")

    # Heurística: el resumen visual debería compartir alguna palabra (por prefijo de 5 letras) con el rasgo distintivo
    stems = lambda t: {w[:5] for w in re.findall(r"\w+", t.lower()) if len(w) >= 5}
    if not stems(identity["caracteristica_distintiva"]) & stems(identity["resumen_visual"]):
        report["advertencias"].append("El resumen visual no parece mencionar la característica distintiva")

    return (identity if not report["errores"] else None), report


@dataclass
class IdentityResult:
    identity: dict
    report: dict
    attempts: list[dict] = field(default_factory=list)  # un registro por intento (semilla, chequeos)
    seed: int = BASE_SEED

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)


def identity_think() -> bool:
    """Razonamiento al generar la identidad: OLLAMA_IDENTITY_THINK, activado por defecto."""
    return env_bool("OLLAMA_IDENTITY_THINK", True)


def generate_identity(
    client: OllamaClient, name: str, *, max_attempts: int = MAX_ATTEMPTS, base_seed: int = BASE_SEED,
    temperature: float = TEMPERATURE, think: bool | None = None, real_names: set[str] = frozenset(),
) -> IdentityResult:
    """Pide la ficha con salida JSON estructurada; reintenta (semilla base+intento) si no valida.

    `think=None` toma OLLAMA_IDENTITY_THINK (activado por defecto)."""
    think = identity_think() if think is None else think
    system, user = build_prompts(name)
    attempts = []
    for i in range(max_attempts):
        seed = base_seed + i
        try:
            raw = client.generate(user, system, temperature=temperature, seed=seed, format=SCHEMA, think=think)
        except OllamaError as e:
            attempts.append({"intento": i + 1, "seed": seed, "errores": [f"Ollama: {e}"]})
            continue
        identity, report = validate_identity(raw, real_names)
        attempts.append({"intento": i + 1, "seed": seed, "chequeos": report})
        if identity is not None:
            return IdentityResult(identity, report, attempts, seed)
    last = attempts[-1].get("chequeos", {}).get("errores") or attempts[-1].get("errores")
    raise OllamaError(f"No se obtuvo una identidad válida tras {max_attempts} intentos. Último error: {last}")
