# 🦖 LabDino

**Una app que inventa un dinosaurio: le da nombre, le escribe una ficha científica, le pinta un retrato y te deja conversar con él.**

Laboratorio II de Procesamiento de Lenguaje Natural · Universidad Sergio Arboleda · 2026 S02
([guía del laboratorio](docs/guia_laboratorio.pdf))

```mermaid
flowchart LR
    A[RNN de caracteres<br/>GRU en PyTorch] -->|nombre| B[Ollama<br/>gemma4:e2b]
    B -->|identidad JSON| C[Text-to-image]
    C -->|imagen| D[Chat con el dinosaurio]
    B -->|descripción como contexto| D
```

> **Alcance actual.** Están implementados y probados el **generador de nombres** y la **identidad con Ollama** (local).
> La generación de imagen, el backend/frontend con chat, el despliegue en AWS y la configuración de SageMaker son las
> siguientes etapas ([ver hoja de ruta](#hoja-de-ruta)).

## Ejemplo

El modelo de caracteres propone un nombre nuevo, que no está en el dataset, y Ollama construye la especie:

> **Doractodon** — *Este dinosaurio es un animal de tamaño mediano, de color marrón terroso, con una postura erguida y
> una protuberancia cónica en la parte superior del hocico que modifica su dentición.*
> Hábitat: llanuras áridas y matorrales semiáridos · Dieta: herbívora especializada · Carácter: solitario y cauteloso.

Ficha completa en [ollama/identity.md](ollama/identity.md).

## Inicio rápido

Requisitos: [uv](https://docs.astral.sh/uv/) y Python 3.12. Para la parte de identidad, un [Ollama](https://ollama.com/) accesible
con el modelo descargado (`ollama pull gemma4:e2b`).

```bash
uv sync                                  # instala dependencias (PyTorch, httpx, pytest, jupyterlab…)
uv run pytest                            # 74 tests; uno más, contra Ollama real, se omite sin variables de entorno
```

**Generar un nombre sin reentrenar** (usa los artefactos versionados):

```python
from pathlib import Path
from labdino.artifacts import load_artifacts, generate_novel_name

art = load_artifacts(Path("name_generator/artifacts"))
print(generate_novel_name(art).capitalize())   # p. ej. "Teratonia"; cada llamada usa una semilla distinta
```

**Generar la identidad** (necesita Ollama):

```bash
cp .env.example .env                     # completa OLLAMA_BASE_URL y OLLAMA_MODEL
set -a; source .env; set +a              # el repo no lee .env por sí solo
uv run python ollama/generate_identity.py
```

## Componentes

| Ruta | Qué hay |
|---|---|
| [`src/labdino/`](src/labdino) | Código compartido que usan el notebook y el futuro backend |
| [`name_generator/`](name_generator) | Notebook de entrenamiento y evaluación + [`artifacts/`](name_generator/artifacts) (lo único que carga el backend) |
| [`ollama/`](ollama) | Registro del modelo, prompt de identidad, scripts y la ficha generada |
| [`evidence/`](evidence) | Curvas de pérdida, resumen de entrenamiento, configuraciones de muestreo, 10 nombres y su análisis |
| [`data/`](data) | `dinos.csv` |
| [`tests/`](tests) | Pruebas con `pytest` |

Módulos principales de `src/labdino/`:

| Módulo | Responsabilidad |
|---|---|
| `preprocessing` | Normalización, vocabulario, tokens `<PAD>/<SOS>/<EOS>`, T y pares X/Y desplazados |
| `model`, `training` | `CharRNN` (Embedding → RNN/LSTM/GRU → Linear), pérdida sin `<PAD>`, split y bucle con early stopping |
| `sampling` | Muestreo autoregresivo con temperatura, top-k y top-p; métricas de novedad |
| `artifacts` | Guardar y cargar el modelo para generar nombres sin reentrenar |
| `ollama_client`, `identity` | Cliente HTTP de Ollama; prompt, esquema JSON, validación y reintentos de la ficha |

## Cómo funciona

### 1. Generador de nombres

Un modelo de lenguaje a nivel de carácter aprende a predecir la siguiente letra de un nombre de dinosaurio y, al generar,
usa cada carácter producido como entrada del siguiente paso (autoregresivo).

- **Datos:** [`data/dinos.csv`](data/dinos.csv), copia de [jpospinalo/MachineLearning](https://github.com/jpospinalo/MachineLearning/blob/main/nlp/dinos.csv).
  Normalizado quedan 1524 nombres, vocabulario de 29 tokens y T = 27.
- **Preprocesamiento:** secuencias `[<SOS>, c₁…cₙ, <EOS>]` rellenadas con `<PAD>`; entrada `X = [x₀…x_{T-1}]`, objetivo `Y = [x₁…x_T]`.
- **Entrenamiento:** Adam, entropía cruzada que ignora `<PAD>`, CPU y semilla 42. Dos configuraciones con el mismo split (90/10), batch y
  learning rate, y early stopping sobre la pérdida de validación. Se guarda el checkpoint de la **mejor** época, no el último.

| Config | Arquitectura | Parámetros | Mejor val loss | Mejor época |
|---|---|---|---|---|
| A | GRU · 1 capa · hidden 128 | 66 877 | 1.5716 | 13 |
| **B** (elegida) | GRU · 2 capas · hidden 256 · dropout 0.2 | 625 853 | **1.5589** | 8 |

La configuración se eligió por menor pérdida de validación, criterio fijado antes de entrenar. B gana por poco y se sobreajusta antes:

![Curvas de pérdida de las configuraciones A y B](evidence/loss_curves_comparison.png)

- **Muestreo:** se compararon seis estrategias con 1000 muestras cada una (copias del dataset, repetidos, longitud, nombres cortos).
  Se usa **top-p = 0.9 con temperatura 1.0**: casi no copia nombres reales (1.2 %), apenas repite (1.1 %) y no produce nombres degenerados.
  Detalle y análisis en [evidence/generated_names.md](evidence/generated_names.md).
- **Novedad:** un nombre solo se acepta si no coincide, con la misma normalización, con ninguno de los 1524 del dataset
  (`train_names.txt` viaja con los artefactos).

Los 10 nombres entregados: Alapentosaurus, Salimimus, Gatarisaurus, **Doractodon** (elegido), Kematentaceratops, Revecterris,
Mamusaurus, Sauroreptor, Yadanoceratops y Drachyrosuchus.

### 2. Identidad con Ollama

El nombre elegido pasa a un LLM local (`gemma4:e2b`, Ollama 0.40.6; versiones exactas en [ollama/model_info.md](ollama/model_info.md)).

- **Prompt** ([ollama/identity_prompt.md](ollama/identity_prompt.md)): pide apariencia, hábitat, alimentación, comportamiento y un único rasgo
  físico distintivo, con pistas de sufijos reales (-saurus, -raptor, -odon, -venator) como orientación y no como plantilla.
  Exige rasgos coherentes entre sí y una especie inventada.
- **Salida estructurada:** JSON con claves fijas (el esquema se envía en el parámetro `format`), 150–200 palabras, más `visual_prompt_en`,
  una frase en inglés pensada para el generador de imágenes.
- **Validación automática:** reintenta si el JSON no parsea, faltan claves, hay campos vacíos o la longitud se sale de 120–230. Avisa si aparecen
  nombres de especies reales, si la ubicación del rasgo no coincide entre campos o si el nombre propio se cuela en el prompt de imagen.
- **Semillas:** prueba 42, 43 y 44 y conserva la primera ficha sin advertencias. La elegida y el motivo quedan en
  [ollama/identity.json](ollama/identity.json), junto con el prompt, el modelo, los parámetros y la fecha.

## Configuración

El repo no trae URLs ni credenciales. Las variables se leen del entorno (plantilla en [`.env.example`](.env.example); `.env` está en `.gitignore`):

| Variable | Descripción | Por defecto |
|---|---|---|
| `OLLAMA_BASE_URL` | URL base de Ollama | obligatoria |
| `OLLAMA_MODEL` | Modelo a usar | obligatoria |
| `OLLAMA_NGROK_SKIP_WARNING` | Añade la cabecera `ngrok-skip-browser-warning` | `false` |
| `OLLAMA_CONNECT_TIMEOUT` / `OLLAMA_READ_TIMEOUT` | Tiempos de espera en segundos | `5` / `300` |
| `OLLAMA_IDENTITY_THINK` | Razonamiento del modelo al crear la ficha | `true` |
| `OLLAMA_CHAT_THINK` | Razonamiento del modelo en el chat | `false` |

Los errores de conexión, de modelo inexistente o de configuración indican qué variable revisar.

## Reproducir los resultados

```bash
# Reentrenar, evaluar, muestrear y regenerar evidence/ y name_generator/artifacts/ (≈20 s en CPU)
uv run jupyter nbconvert --to notebook --execute --inplace name_generator/train_and_evaluate.ipynb

# Regenerar el registro del modelo de Ollama (sirve también en SageMaker)
uv run python ollama/write_model_info.py

# Reescribir solo el prompt de identidad, sin llamar a Ollama
uv run python ollama/generate_identity.py --show-prompt
```

El entrenamiento es determinista: reproduce los mismos checkpoints y los mismos 10 nombres. El filtro `nbstripout` quita las salidas del
notebook al hacer commit.

### Artefactos del generador

| Archivo | Contenido |
|---|---|
| `model.pt` | Pesos de la mejor época de la configuración B |
| `vocab.json` | Vocabulario (el índice es el id) |
| `config.json` | Arquitectura, capas, hidden, dropout, T, ids especiales, muestreo por defecto y criterio de selección |
| `train_names.txt` | Los 1524 nombres normalizados, para rechazar copias |

Los checkpoints de A y B están en `evidence/checkpoints/` como evidencia.

## Hoja de ruta

- [x] Generador de nombres: entrenamiento, comparación de dos configuraciones, muestreo y artefactos
- [x] Identidad con Ollama en local
- [ ] Ollama en SageMaker (`m5.xlarge`, Docker con `data-root` persistente y script `on-start`)
- [ ] Imagen con un modelo text-to-image a partir de la ficha
- [ ] Aplicación web con el chat y el botón **Nuevo Dinosaurio**, desplegada en AWS

Nota para la app: en *Nuevo Dinosaurio* la semilla del nombre y la de la identidad deben ser aleatorias. `generate_novel_name` ya lo es sin
generador; para la identidad hay que pasar semillas distintas a `generate_clean_identity(seeds=...)`.
