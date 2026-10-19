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
CLEAN_SEEDS = (42, 43, 44)  # semillas que se prueban en orden hasta obtener una ficha sin errores ni advertencias
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
comportamiento y UN único rasgo físico distintivo visible, con su ubicación anatómica exacta (una sola parte del \
cuerpo, por ejemplo «en la parte superior del hocico» o «a lo largo del lomo»). Después escribe la ficha con estas \
claves, todas en texto (cadena de caracteres), con esta orientación de extensión:
- "apariencia": el cuerpo en general (tamaño, postura, proporciones, piel o plumas y colores generales). NO \
menciones el rasgo distintivo (unas 40 palabras).
- "habitat": clima, terreno y vegetación donde vive (unas 25 palabras).
- "alimentacion": qué come y cómo lo obtiene, acorde con su dentición y su cuerpo (unas 25 palabras).
- "comportamiento": hábitos sociales, actividad y relación con otros animales (unas 30 palabras).
- "caracteristica_distintiva": el único rasgo FÍSICO y visible e inconfundible (por ejemplo la forma, textura o \
color de una parte del cuerpo), no un hábito ni una conducta, con su ubicación anatómica (unas 25 palabras).
- "resumen_visual": una sola oración que describe al dinosaurio y usa la palabra «dinosaurio» como sujeto, SIN el \
nombre propio; solo lo que se vería en una imagen, con términos concretos (formas, colores, texturas, tamaño \
aproximado, postura, entorno), incluyendo el rasgo distintivo y sin compararlo con especies reales (unas 30 \
palabras).
- "visual_prompt_en": una frase corta en inglés (máximo 35 palabras) para un generador de imágenes, con «a dinosaur» \
como sujeto y SIN el nombre propio, que resuma el aspecto con términos concretos e incluya el rasgo distintivo; no \
cuenta para el límite de palabras.

El rasgo distintivo (único, con la misma ubicación y el mismo término):
- Aparece solo en "caracteristica_distintiva", "resumen_visual" y "visual_prompt_en". "apariencia" no lo menciona.
- Usa la misma ubicación anatómica y la misma palabra para el rasgo en esos tres campos; en "visual_prompt_en", \
su traducción literal al inglés.
- No lo sitúes en otra parte del cuerpo ni lo describas en dos lugares distintos, y no añadas un segundo rasgo \
distintivo (por ejemplo, no mezcles cráneo, nuca, cuello, espalda y lomo para el mismo rasgo).

Estilo: redacta de forma natural y directa, con sujeto claro; evita las construcciones impersonales o de \
observación como «se detecta que», «se observa que» o «se aprecia que».

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


# Zonas del cuerpo donde puede ubicarse el rasgo distintivo (español e inglés). Las cuatro primeras son las que
# suelen confundirse entre sí.
LOCATION_PATTERNS = {
    "cráneo": r"\b(cr[aá]neo|craneal|cabeza|skull|head)\b",
    "nuca": r"\b(nuca|nape)\b",
    "cuello": r"\b(cuello|neck)\b",
    "dorso": r"\b(dorsal(es)?|dorso|espalda|lomo|back|spine|spinal)\b",
    "cola": r"\b(cola|tail)\b",
    "hocico": r"\b(hocico|morro|snout|muzzle)\b",
}
CONFUSABLE = {"cráneo", "nuca", "cuello", "dorso"}


def body_locations(text: str) -> set[str]:
    """Zonas del cuerpo mencionadas en el texto."""
    text = text.lower()
    return {zone for zone, pattern in LOCATION_PATTERNS.items() if re.search(pattern, text)}


def location_warnings(identity: dict) -> list[str]:
    """Advertencias si la ubicación del rasgo distintivo no es la misma en los cuatro campos.

    La referencia es `caracteristica_distintiva`: debe nombrar una sola zona. En `resumen_visual` y
    `visual_prompt_en` debe aparecer esa zona y no otra de las zonas confundibles (cráneo, nuca, cuello, dorso).
    `apariencia` no menciona el rasgo, así que no se revisa.
    """
    ref = body_locations(identity["caracteristica_distintiva"])
    if not ref:
        return ["caracteristica_distintiva no indica una ubicación anatómica reconocible"]
    warnings = []
    if len(ref) > 1:
        warnings.append(f"caracteristica_distintiva menciona varias ubicaciones: {sorted(ref)}")
    for key in ("resumen_visual", VISUAL_PROMPT_KEY):
        found = body_locations(identity[key])
        missing = ref - found
        stray = (found & CONFUSABLE) - ref
        if missing:
            warnings.append(f"{key} no menciona la ubicación del rasgo {sorted(missing)}")
        if stray:
            warnings.append(f"{key} sitúa algo en otra zona ({sorted(stray)}) que difiere del rasgo {sorted(ref)}")
    return warnings


def find_real_names(text: str, real_names: set[str]) -> list[str]:
    """Nombres de especies reales (del dataset y célebres) que aparecen como palabra en el texto."""
    words = set(re.findall(r"[\w-]+", text.lower()))
    return sorted(words & (real_names | FAMOUS_REAL))


def validate_identity(
    raw: str, real_names: set[str] = frozenset(), name: str | None = None
) -> tuple[dict | None, dict]:
    """Parsea y revisa la respuesta. Devuelve (identidad o None, informe de chequeos).

    Errores (reintento): JSON inválido, claves faltantes/sobrantes, campos vacíos, longitud fuera de 120-230
    palabras, resumen visual sin «dinosaurio». Advertencias (no bloquean): longitud entre 120-150 o 200-230,
    nombres de especies reales en la salida, nombre propio dentro de resumen_visual o visual_prompt_en, ubicación del rasgo distintivo inconsistente entre campos, resumen sin
    el vocabulario del rasgo distintivo.
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

    if name:
        for key in ("resumen_visual", VISUAL_PROMPT_KEY):
            if re.search(rf"\b{re.escape(name.lower())}\b", identity[key].lower()):
                report["advertencias"].append(f"{key} contiene el nombre propio «{name.capitalize()}»")
    report["advertencias"] += location_warnings(identity)

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
    trials: list[dict] = field(default_factory=list)  # semillas probadas por generate_clean_identity
    choice_reason: str = ""

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
        identity, report = validate_identity(raw, real_names, name)
        attempts.append({"intento": i + 1, "seed": seed, "chequeos": report})
        if identity is not None:
            return IdentityResult(identity, report, attempts, seed)
    last = attempts[-1].get("chequeos", {}).get("errores") or attempts[-1].get("errores")
    raise OllamaError(f"No se obtuvo una identidad válida tras {max_attempts} intentos. Último error: {last}")


def generate_clean_identity(
    client: OllamaClient, name: str, *, seeds: tuple[int, ...] = CLEAN_SEEDS, temperature: float = TEMPERATURE,
    think: bool | None = None, real_names: set[str] = frozenset(),
) -> IdentityResult:
    """Prueba las semillas en orden y se queda con la primera ficha sin errores NI advertencias.

    Si ninguna sale limpia, elige la válida con menos advertencias (la primera en caso de empate) y lo dice en
    `choice_reason`. Registra en `trials` cada semilla probada. Lanza OllamaError si ninguna es válida.
    """
    trials, attempts, valid = [], [], []
    for seed in seeds:
        try:
            res = generate_identity(
                client, name, max_attempts=1, base_seed=seed, temperature=temperature, think=think,
                real_names=real_names,
            )
        except OllamaError as e:
            trials.append({"seed": seed, "valida": False, "errores": str(e), "advertencias": []})
            continue
        attempts += res.attempts
        warns = res.report["advertencias"]
        trials.append({"seed": seed, "valida": True, "errores": [], "advertencias": warns,
                       "palabras": res.report["palabras"]})
        valid.append(res)
        if not warns:
            res.attempts, res.trials = attempts, trials
            res.choice_reason = (f"Seed {seed}: primera ficha válida y sin advertencias "
                                 f"(semillas probadas: {[t['seed'] for t in trials]}).")
            return res
    if not valid:
        raise OllamaError(f"Ninguna semilla de {list(seeds)} produjo una ficha válida: {trials}")
    best = min(valid, key=lambda r: len(r.report["advertencias"]))
    best.attempts, best.trials = attempts, trials
    best.choice_reason = (f"Ninguna semilla salió limpia; se eligió la seed {best.seed} por tener menos advertencias "
                          f"({len(best.report['advertencias'])}).")
    return best
