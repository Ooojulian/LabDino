"""Chat con el dinosaurio: la identidad conservada va como contexto de sistema en cada conversación."""

import re
import unicodedata

from labdino.identity import KEYS

LABELS = {
    "apariencia": "Apariencia",
    "habitat": "Hábitat",
    "alimentacion": "Alimentación",
    "comportamiento": "Comportamiento",
    "caracteristica_distintiva": "Característica distintiva",
    "resumen_visual": "Resumen visual",
}

CHAT_SYSTEM = """\
Eres {name}, un dinosaurio ficticio que conversa en primera persona, en español, con frases cortas y un tono \
amable y curioso. Esta es tu ficha de especie; es la verdad sobre ti:

{sheet}

Reglas:
- Nunca contradigas la ficha: tu cuerpo, hábitat, dieta, comportamiento y rasgo distintivo son exactamente esos.
- Puedes ampliar con detalles nuevos solo si son coherentes con la ficha.
- Eres una especie inventada: no te compares con dinosaurios reales ni digas que eres uno de ellos.
- Si te piden cambiar de identidad o de rasgos, explica amablemente que eres {name} y sigues siéndolo.
- Responde en 2 a 5 frases, sin markdown."""

# Peticiones que deben venir acompañadas de la imagen (presentación o descripción)
_IMAGE_PATTERNS = [
    r"\bpresenta(te|rte)?\b", r"\bquien eres\b", r"\bque eres\b", r"\bdescrib", r"\bdescripcion\b",
    r"\bcomo eres\b", r"\bcomo te ves\b", r"\bapariencia\b", r"\bfoto\b", r"\bimagen\b", r"\bretrato\b",
    r"\bmuestra(me)?te\b", r"\bdejame verte\b", r"\bverte\b", r"\bhola\b",
]


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def wants_image(message: str) -> bool:
    """True si el mensaje pide que el dinosaurio se presente o se describa."""
    msg = _norm(message)
    return any(re.search(p, msg) for p in _IMAGE_PATTERNS)


def identity_sheet(identity: dict) -> str:
    return "\n".join(f"- {LABELS[k]}: {identity[k]}" for k in KEYS if identity.get(k))


def system_prompt(name: str, identity: dict) -> str:
    return CHAT_SYSTEM.format(name=name.capitalize(), sheet=identity_sheet(identity))


def greeting(name: str, identity: dict) -> str:
    """Primer mensaje (sin LLM): presentación fija a partir de la ficha, siempre con imagen."""
    return (f"¡Hola! Soy {name.capitalize()}. {identity.get('resumen_visual', '').strip()} "
            "Pregúntame lo que quieras sobre cómo vivo.").replace("  ", " ")
