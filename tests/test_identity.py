import json

import httpx
import pytest

from labdino.identity import (ALL_KEYS, KEYS, SCHEMA, VISUAL_PROMPT_KEY, build_prompts, count_words, find_real_names,
                              generate_identity, validate_identity)
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


def test_relevant_suffix_depends_on_name():
    assert "sufijo relevante para este nombre es -raptor" in build_prompts("xraptor")[1]
    assert "sufijo relevante para este nombre es -saurus" in build_prompts("mamusaurus")[1]
    assert "Ninguno de estos sufijos" in build_prompts("revecterris")[1]


def test_count_words():
    assert count_words("Un pulpo marrón, de tres corazones.") == 6


def test_valid_identity_passes():
    ident, report = validate_identity(json.dumps(make_identity()))  # 6 * 28 = 168 palabras (visual_prompt_en no cuenta)
    assert ident is not None and report["palabras"] == 168 and report["longitud_ok"] and not report["advertencias"]


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
