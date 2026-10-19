import json

import httpx
import pytest

from labdino.identity import (ALL_KEYS, KEYS, SCHEMA, VISUAL_PROMPT_KEY, build_prompts, count_words, find_real_names,
                              generate_clean_identity, generate_identity, location_warnings, validate_identity)
from labdino.ollama_client import OllamaClient, OllamaError


def make_identity(words_per_field: int = 28, **overrides) -> dict:
    ident = {k: " ".join(["palabra"] * words_per_field) for k in KEYS}
    ident["resumen_visual"] = "dinosaurio " + " ".join(["palabra"] * (words_per_field - 1))
    ident[VISUAL_PROMPT_KEY] = "a green dinosaur with a blue crest"
    ident.update(overrides)
    return ident


def test_prompts_contain_name_suffixes_and_rules():
    system, user = build_prompts("doractodon")
    assert "«Doractodon»" in user
    for suffix in ("-saurus", "-raptor", "-odon", "-venator"):
        assert suffix in user
    for key in ALL_KEYS:
        assert f'"{key}"' in user
    assert "150 y 200" in user and "JSON" in system
    assert set(SCHEMA["required"]) == set(ALL_KEYS) and SCHEMA["additionalProperties"] is False
    # (2) el sistema no nombra especies reales; (3) (4) redacción pedida
    assert not any(n in system for n in ("Tyrannosaurus", "Velociraptor", "Triceratops", "Spinosaurus"))
    assert "define mentalmente un perfil coherente" in user and "razonamiento" not in user
    assert "no lo corrijas" in user and "sufijo relevante para este nombre es -odon" in user
    assert "-saurus, -raptor, -venator son solo contexto" in user
    assert "FÍSICO" in user and "«dinosaurio»" in user
    # rasgo único; apariencia no lo menciona; sin nombre propio en resumen/inglés; estilo natural
    assert "UN único rasgo físico distintivo" in user and "NO menciones el rasgo distintivo" in user
    assert "SIN el nombre propio" in user and "«a dinosaur»" in user
    assert "«se detecta que»" in user and "Aparece solo en" in user


def test_relevant_suffix_depends_on_name():
    assert "sufijo relevante para este nombre es -raptor" in build_prompts("xraptor")[1]
    assert "sufijo relevante para este nombre es -saurus" in build_prompts("mamusaurus")[1]
    assert "Ninguno de estos sufijos" in build_prompts("revecterris")[1]


def test_count_words():
    assert count_words("Un pulpo marrón, de tres corazones.") == 6


def test_valid_identity_passes():
    ident, report = validate_identity(json.dumps(make_identity()))  # 6 * 28 = 168 palabras (visual_prompt_en no cuenta)
    assert ident is not None and report["palabras"] == 168 and report["longitud_ok"]
    assert [w for w in report["advertencias"] if "ubicación" not in w] == []  # solo la de ubicación (el relleno no la tiene)


@pytest.mark.parametrize("per_field, warns", [(22, True), (26, False), (33, False), (36, True)])
def test_length_warning_bands(per_field, warns):
    # 6*22=132 y 6*36=216 caen en las bandas de advertencia; 6*26=156 y 6*33=198 están en 150-200
    ident, report = validate_identity(json.dumps(make_identity(per_field)))
    assert ident is not None
    assert any("fuera de lo pedido" in w for w in report["advertencias"]) is warns


def test_hard_length_limits():
    assert validate_identity(json.dumps(make_identity(19)))[0] is None  # 114 < 120
    assert validate_identity(json.dumps(make_identity(39)))[0] is None  # 234 > 230
    assert validate_identity(json.dumps(make_identity(20)))[0] is not None  # 120
    assert validate_identity(json.dumps(make_identity(38)))[0] is not None  # 228


def test_resumen_must_mention_dinosaurio():
    ident, report = validate_identity(json.dumps(make_identity(resumen_visual="Un animal verde " * 8)))
    assert ident is None and any("dinosaurio" in e for e in report["errores"])


def test_real_species_names_raise_warning_only():
    raw = json.dumps(make_identity(comportamiento="Caza como un Velociraptor y como el ornitholestes"))
    ident, report = validate_identity(raw, {"ornitholestes"})
    assert ident is not None
    assert any("velociraptor" in w and "ornitholestes" in w for w in report["advertencias"])
    assert find_real_names("Un dinosaurio verde", {"ornitholestes"}) == []


@pytest.mark.parametrize("raw, error", [
    ("no es json", "JSON inválido"),
    ("[1, 2]", "no es un objeto"),
    (json.dumps({k: "x" for k in KEYS[:-1]}), "Faltan claves"),
    (json.dumps({**make_identity(), "extra": "x"}), "no permitidas"),
    (json.dumps(make_identity(habitat="  ")), "vacíos"),
    (json.dumps(make_identity(5)), "Longitud"),
    (json.dumps(make_identity(60)), "Longitud"),
    (json.dumps({k: v for k, v in make_identity().items() if k != VISUAL_PROMPT_KEY}), "Faltan claves"),
])
def test_invalid_identity_is_rejected(raw, error):
    ident, report = validate_identity(raw)
    assert ident is None and any(error in e for e in report["errores"])


def client_with(responses: list) -> tuple[OllamaClient, list]:
    seeds = []

    def handler(request):
        body = json.loads(request.content)
        seeds.append(body["options"]["seed"])
        assert body["format"] == SCHEMA and body["options"]["temperature"] == 0.7
        item = responses.pop(0)
        return httpx.Response(200, json={"response": item})

    return OllamaClient("http://x", "m", transport=httpx.MockTransport(handler)), seeds


def test_retries_until_valid_with_new_seed_each_time():
    client, seeds = client_with(["basura", json.dumps(make_identity(5)), json.dumps(make_identity())])
    result = generate_identity(client, "doractodon")
    assert seeds == [42, 43, 44] and len(result.attempts) == 3 and result.seed == 44


def test_gives_up_after_max_attempts():
    client, seeds = client_with(["basura"] * 3)
    with pytest.raises(OllamaError, match="3 intentos"):
        generate_identity(client, "doractodon", max_attempts=3)
    assert len(seeds) == 3


def test_identity_think_enabled_by_default_and_configurable(monkeypatch):
    from labdino.identity import identity_think

    monkeypatch.delenv("OLLAMA_IDENTITY_THINK", raising=False)
    assert identity_think() is True
    monkeypatch.setenv("OLLAMA_IDENTITY_THINK", "false")
    assert identity_think() is False


def test_generate_identity_sends_think_true_by_default(monkeypatch):
    monkeypatch.delenv("OLLAMA_IDENTITY_THINK", raising=False)
    seen = []

    def handler(request):
        seen.append(json.loads(request.content)["think"])
        return httpx.Response(200, json={"response": json.dumps(make_identity())})

    client = OllamaClient("http://x", "m", transport=httpx.MockTransport(handler))
    generate_identity(client, "doractodon")
    monkeypatch.setenv("OLLAMA_IDENTITY_THINK", "0")
    generate_identity(client, "doractodon")
    assert seen == [True, False]


def located(**overrides) -> dict:
    filler = " ".join(["palabra"] * 40)  # para que la longitud total sea válida (120-230)
    base = make_identity(
        habitat=filler, alimentacion=filler, comportamiento=filler,
        apariencia="Posee una cresta ósea en la parte superior del hocico.",
        caracteristica_distintiva="Una cresta ósea en la parte superior del hocico.",
        resumen_visual="Un dinosaurio verde con una cresta ósea en la parte superior del hocico.",
    )
    base[VISUAL_PROMPT_KEY] = "A green dinosaur with a bony crest on the top of the snout"
    base.update(overrides)
    return base


def test_consistent_location_has_no_warning():
    assert location_warnings(located()) == []


def test_location_contradiction_between_fields_is_warned():
    # el caso anterior: nuca en la característica y «dorsal» en el resumen y el prompt en inglés
    ident = located(
        caracteristica_distintiva="Placa ósea que se extiende a lo largo de la nuca.",
        resumen_visual="Un dinosaurio con una placa ósea dorsal gruesa.",
    )
    ident[VISUAL_PROMPT_KEY] = "A dinosaur with a heavy dorsal bone plate"
    warnings = location_warnings(ident)
    assert any(w.startswith("resumen_visual") and "dorso" in w for w in warnings)
    assert any(w.startswith(VISUAL_PROMPT_KEY) for w in warnings)


def test_apariencia_is_not_checked_for_location():
    ident = located(apariencia="Cuerpo robusto, cola larga, cabeza pequeña y espalda arqueada.")
    assert location_warnings(ident) == []


def test_proper_name_in_resumen_or_visual_prompt_is_warned():
    ident = located(resumen_visual="El dinosaurio Doractodon tiene una cresta ósea en la parte superior del hocico.")
    ident[VISUAL_PROMPT_KEY] = "Doractodon, a dinosaur with a bony crest on the top of the snout"
    _, report = validate_identity(json.dumps(ident), name="doractodon")
    warns = [w for w in report["advertencias"] if "nombre propio" in w]
    assert len(warns) == 2 and any(VISUAL_PROMPT_KEY in w for w in warns)
    _, clean = validate_identity(json.dumps(located()), name="doractodon")
    assert not any("nombre propio" in w for w in clean["advertencias"])


def test_trait_with_several_locations_is_warned():
    ident = located(caracteristica_distintiva="Placa ósea en el cráneo que baja por la nuca y la espalda.")
    assert any("varias ubicaciones" in w for w in location_warnings(ident))


def test_missing_location_in_trait_is_warned():
    assert location_warnings(located(caracteristica_distintiva="Un color azul intenso."))[0].startswith(
        "caracteristica_distintiva no indica"
    )


def test_location_warning_does_not_block_validation():
    ident, report = validate_identity(json.dumps(make_identity()))
    assert ident is not None  # sin ubicación reconocible solo hay advertencia
    assert any("ubicación" in w for w in report["advertencias"])


def clean_identity(**overrides) -> dict:
    return located(**overrides)


def seeded_client(by_seed: dict) -> tuple[OllamaClient, list]:
    seeds = []

    def handler(request):
        seed = json.loads(request.content)["options"]["seed"]
        seeds.append(seed)
        return httpx.Response(200, json={"response": by_seed[seed]})

    return OllamaClient("http://x", "m", transport=httpx.MockTransport(handler)), seeds


def test_clean_identity_stops_at_first_clean_seed():
    dirty = located(resumen_visual="Un dinosaurio con una placa dorsal.")  # ubicación distinta -> advertencia
    client, seeds = seeded_client({42: json.dumps(dirty), 43: json.dumps(clean_identity()), 44: "no se llama"})
    res = generate_clean_identity(client, "doractodon", think=True)
    assert seeds == [42, 43] and res.seed == 43
    assert [t["seed"] for t in res.trials] == [42, 43] and res.trials[0]["advertencias"] and not res.trials[1]["advertencias"]
    assert "Seed 43" in res.choice_reason


def test_clean_identity_falls_back_to_fewest_warnings():
    two = located(resumen_visual="Un dinosaurio con una placa dorsal.")
    one = located(caracteristica_distintiva="Un color azul intenso.")  # 1 advertencia de ubicación
    client, seeds = seeded_client({42: json.dumps(two), 43: "basura", 44: json.dumps(one)})
    res = generate_clean_identity(client, "doractodon", think=True)
    assert seeds == [42, 43, 44]
    assert [t["valida"] for t in res.trials] == [True, False, True]
    assert res.seed == min(
        (t for t in res.trials if t["valida"]), key=lambda t: len(t["advertencias"])
    )["seed"]
    assert "Ninguna semilla salió limpia" in res.choice_reason


def test_clean_identity_raises_if_none_valid():
    client, _ = seeded_client({42: "x", 43: "y", 44: "z"})
    with pytest.raises(OllamaError, match="Ninguna semilla"):
        generate_clean_identity(client, "doractodon", think=True)
