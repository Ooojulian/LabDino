# LabDino — Laboratorio II de PLN (Universidad Sergio Arboleda, 2026 S02)

App que crea un dinosaurio ficticio: modelo de caracteres → nombre → Ollama → identidad → imagen → chat.
Guía del laboratorio: [docs/guia_laboratorio.pdf](docs/guia_laboratorio.pdf).

## Estructura

| Ruta | Contenido |
|---|---|
| `src/labdino/` | Código compartido (preprocesamiento, modelo, muestreo, carga/guardado) |
| `name_generator/` | Notebook de entrenamiento/evaluación y artefactos del modelo |
| `ollama/` | Cliente, prompt de identidad y registro del modelo |
| `evidence/` | Curvas de pérdida, configuraciones, nombres generados y análisis |
| `data/` | Datos de entrenamiento |
| `docs/` | Documentación y guía |
| `tests/` | Pruebas (pytest) |

## Datos

`data/dinos.csv` (un nombre por línea, sin encabezado) es una copia local de
<https://github.com/jpospinalo/MachineLearning/blob/main/nlp/dinos.csv>.

## Configuración

```bash
uv sync
cp .env.example .env   # completar OLLAMA_BASE_URL y OLLAMA_MODEL
uv run pytest
```

No se versionan credenciales ni tokens; `.env` está en `.gitignore`.
