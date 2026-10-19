# Prompt de identidad — Doractodon

Prompt exacto (lo construye `labdino.identity.build_prompts`) que se enviará a Ollama.

## Parámetros

| Parámetro | Valor |
|---|---|
| Endpoint | `/api/generate` (sin streaming) |
| `format` | esquema JSON (abajo): 7 claves fijas, sin claves extra |
| temperature | 0.7 |
| seed | se prueban 42, 43, 44 en orden; se elige la primera ficha válida sin advertencias (si ninguna, la de menos advertencias). Se registran las semillas probadas y el motivo en `identity.json` |
| Reintentos | un intento por semilla; el número de intentos y las semillas probadas se registran en `identity.json` |
| Modo thinking | `think=True` por defecto para la identidad (configurable con `OLLAMA_IDENTITY_THINK`); el chat usa `think=False` (`OLLAMA_CHAT_THINK`). El razonamiento llega aparte y no se mezcla con la respuesta |
| Longitud pedida en el prompt | 150-200 palabras (suma de las 6 claves de la ficha; `visual_prompt_en` no cuenta) |
| Longitud validada | reintento fuera de 120-230; advertencia entre 120-150 y 200-230 |

## Prompt de sistema

```text
Eres un paleontólogo y escritor de fichas de especies ficticias. Escribes en español, en tercera persona y en presente, con un tono científico claro. Respondes SIEMPRE y únicamente con un objeto JSON válido, sin texto adicional, sin markdown y sin comentarios.

Principios de tu trabajo:
- Coherencia: todos los rasgos de la especie encajan entre sí. El hábitat explica el cuerpo, el cuerpo explica la dieta y la dieta explica el comportamiento.
- La especie es inventada: no copies ni parafrasees ninguna especie real, no menciones ni compares con especies reales, y no afirmes hallazgos fósiles ni lugares de excavación reales.
- El nombre de la especie es un nombre propio inventado: lo usas tal cual, sin corregirlo ni sustituirlo.
```

## Prompt de usuario

```text
Crea la ficha de una especie de dinosaurio ficticia llamada «Doractodon».

«Doractodon» es un nombre propio inventado: no lo corrijas, no lo sustituyas por otro ni lo reinterpretes como una especie real.

Contexto sobre sufijos de nombres reales (solo orientación, no plantilla):
- -saurus: «lagarto»
- -raptor: «ladrón o rapaz»
- -odon: «diente»
- -venator: «cazador»
El sufijo relevante para este nombre es -odon («diente»); -saurus, -raptor, -venator son solo contexto.
No repitas estas pistas en la ficha ni conviertas el nombre en un chiste.

Antes de escribir, define mentalmente un perfil coherente: hábitat, tamaño, locomoción, dentición, dieta, comportamiento y UN único rasgo físico distintivo visible, con su ubicación anatómica exacta (una sola parte del cuerpo, por ejemplo «en la parte superior del hocico» o «a lo largo del lomo»). Después escribe la ficha con estas claves, todas en texto (cadena de caracteres), con esta orientación de extensión:
- "apariencia": el cuerpo en general (tamaño, postura, proporciones, piel o plumas y colores generales). NO menciones el rasgo distintivo (unas 40 palabras).
- "habitat": clima, terreno y vegetación donde vive (unas 25 palabras).
- "alimentacion": qué come y cómo lo obtiene, acorde con su dentición y su cuerpo (unas 25 palabras).
- "comportamiento": hábitos sociales, actividad y relación con otros animales (unas 30 palabras).
- "caracteristica_distintiva": el único rasgo FÍSICO y visible e inconfundible (por ejemplo la forma, textura o color de una parte del cuerpo), no un hábito ni una conducta, con su ubicación anatómica (unas 25 palabras).
- "resumen_visual": una sola oración que describe al dinosaurio y usa la palabra «dinosaurio» como sujeto, SIN el nombre propio; solo lo que se vería en una imagen, con términos concretos (formas, colores, texturas, tamaño aproximado, postura, entorno), incluyendo el rasgo distintivo y sin compararlo con especies reales (unas 30 palabras).
- "visual_prompt_en": una frase corta en inglés (máximo 35 palabras) para un generador de imágenes, con «a dinosaur» como sujeto y SIN el nombre propio, que resuma el aspecto con términos concretos e incluya el rasgo distintivo; no cuenta para el límite de palabras.

El rasgo distintivo (único, con la misma ubicación y el mismo término):
- Aparece solo en "caracteristica_distintiva", "resumen_visual" y "visual_prompt_en". "apariencia" no lo menciona.
- Usa la misma ubicación anatómica y la misma palabra para el rasgo en esos tres campos; en "visual_prompt_en", su traducción literal al inglés.
- No lo sitúes en otra parte del cuerpo ni lo describas en dos lugares distintos, y no añadas un segundo rasgo distintivo (por ejemplo, no mezcles cráneo, nuca, cuello, espalda y lomo para el mismo rasgo).

Estilo: redacta de forma natural y directa, con sujeto claro; evita las construcciones impersonales o de observación como «se detecta que», «se observa que» o «se aprecia que».

Reglas:
- La ficha (suma de las seis primeras claves, sin "visual_prompt_en") debe tener entre 150 y 200 palabras.
- Tercera persona y presente; sin listas, sin saltos de línea dentro de los campos.
- Ningún campo puede quedar vacío y no añadas claves distintas a las siete indicadas.
```

## Esquema JSON (`format`)

```json
{
  "type": "object",
  "properties": {
    "apariencia": {
      "type": "string"
    },
    "habitat": {
      "type": "string"
    },
    "alimentacion": {
      "type": "string"
    },
    "comportamiento": {
      "type": "string"
    },
    "caracteristica_distintiva": {
      "type": "string"
    },
    "resumen_visual": {
      "type": "string"
    },
    "visual_prompt_en": {
      "type": "string"
    }
  },
  "required": [
    "apariencia",
    "habitat",
    "alimentacion",
    "comportamiento",
    "caracteristica_distintiva",
    "resumen_visual",
    "visual_prompt_en"
  ],
  "additionalProperties": false
}
```

## Chequeos automáticos tras la respuesta

**Reintento (error):**
- El JSON parsea y es un objeto con exactamente las claves `apariencia`, `habitat`, `alimentacion`, `comportamiento`, `caracteristica_distintiva`, `resumen_visual`, `visual_prompt_en`.
- Ningún campo está vacío y todos son texto.
- La suma de palabras de las 6 claves de la ficha está entre 120 y 230.
- `resumen_visual` contiene la palabra «dinosaurio».

**Advertencia (no bloquea):**
- Longitud entre 120-150 o 200-230 palabras.
- Aparece en la salida el nombre de una especie real (los 1524 nombres del dataset más unos célebres como T-Rex o Velociraptor).
- `visual_prompt_en` no menciona «dinosaur» o es demasiado largo.
- La ubicación anatómica del rasgo distintivo (cráneo, nuca, cuello, dorso, cola, hocico) no coincide entre `caracteristica_distintiva`, `resumen_visual` y `visual_prompt_en` (`apariencia` no menciona el rasgo y no se revisa).
- `resumen_visual` o `visual_prompt_en` contienen el nombre propio.
- El resumen visual no comparte vocabulario con la característica distintiva.
