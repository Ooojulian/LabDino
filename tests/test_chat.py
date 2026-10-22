from labdino.chat import greeting, identity_sheet, system_prompt, wants_image

IDENTITY = {
    "apariencia": "Cuerpo robusto.",
    "habitat": "Llanuras áridas.",
    "alimentacion": "Herbívoro.",
    "comportamiento": "Solitario.",
    "caracteristica_distintiva": "Protuberancia cónica.",
    "resumen_visual": "Animal mediano de color marrón.",
}


def test_wants_image_true_for_presentation_and_description():
    for msg in ["Preséntate", "¿Quién eres?", "Descríbete por favor", "¿Cómo eres?", "Muéstrame una foto", "hola"]:
        assert wants_image(msg), msg


def test_wants_image_false_for_other_messages():
    for msg in ["¿Qué comes?", "Cuéntame sobre tu hábitat", "¿Con quién vives?"]:
        assert not wants_image(msg), msg


def test_identity_sheet_lists_all_fields_and_skips_empty():
    sheet = identity_sheet({**IDENTITY, "habitat": ""})
    assert "- Apariencia: Cuerpo robusto." in sheet
    assert "Hábitat" not in sheet
    assert "- Característica distintiva: Protuberancia cónica." in sheet


def test_system_prompt_includes_name_and_sheet():
    prompt = system_prompt("doractodon", IDENTITY)
    assert "Eres Doractodon" in prompt
    for value in IDENTITY.values():
        assert value in prompt


def test_greeting_uses_name_and_summary():
    text = greeting("doractodon", IDENTITY)
    assert text.startswith("¡Hola! Soy Doractodon.")
    assert IDENTITY["resumen_visual"] in text
