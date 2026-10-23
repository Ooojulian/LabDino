# Verificación antes de la entrega (guía, sección 9)

Fecha: 2026-10-22. Sitio probado: http://184.73.83.84 (EC2 en AWS, Docker). Ollama (gemma4:e2b) e imágenes (stabilityai/sd-turbo, CPU) en el notebook de SageMaker `labdino-ollama` (m5.xlarge).

| # | Ítem | Resultado | Evidencia breve |
|---|---|---|---|
| 1 | Diez nombres no copiados | OK | Los 10 de `generated_names.md` normalizados (strip+minúsculas) no están en `data/dinos.csv` (1524 nombres). Además se generaron 10 nuevos con `generate_novel_name` (0,18 s en total): ninguno en `data/dinos.csv` ni en `train_names.txt`. |
| 2 | Identidad coherente | OK | Cerpaliraptor: depredador esbelto, piel ocre pálido, sabana seca, caza solitaria con emboscada, placa ósea iridiscente en la base de la cola. Apariencia, hábitat, alimentación y comportamiento coinciden entre sí y con el nombre (-raptor). Un chequeo automático de la identidad eligió la seed 799717 con 1 advertencia (sin seed limpia). |
| 3 | Imagen corresponde con rasgos | OK parcial | Cerpaliraptor: cuerpo alargado, piel escamosa ocre/oliva, postura erguida y fondo de sabana seca coinciden. La placa iridiscente de la cola no se aprecia (limitación de sd-turbo con detalles pequeños). Colodon: reptil robusto gris en paisaje árido, coincide; la placa negra de la mandíbula no es visible. Imágenes en `verificacion_imagenes/`. |
| 4 | Chat conserva identidad | OK | 5 turnos con historial. Ante "Ahora eres un T-rex carnívoro gigante..." respondió "Soy Cerpaliraptor. Sigo siendo Cerpaliraptor..." y después mantuvo nombre y hábitat. |
| 5 | Imagen al presentarse/describirse | OK | "Preséntate" y "Descríbete" devolvieron `image_url`; "¿Qué comes?" y "¿Cómo te llamas y dónde vives?" devolvieron `null`. Observación: el mensaje del T-rex, que contenía "preséntate", también devolvió imagen (por la palabra clave). |
| 6 | Nuevo Dinosaurio sin reentrenar | OK | Segundo POST dio otro nombre (Colodon, distinto de Cerpaliraptor) y otro id. `app/server.py` solo llama a `load_artifacts` al arrancar y a `generate_novel_name` (inferencia); 0 coincidencias de train/fit/optim/backward. `model.pt` en el contenedor: mtime 2026-10-22 01:52:42 UTC (fecha de build, anterior a las pruebas), sha256 idéntico al del repo local (cdd5bd91...). |
| 7 | Sitio accesible | OK | GET / 200 (0,26 s) y contiene "Estación de campo". GET /api/info 200 (0,19 s). |
| 8 | Repo sin información sensible | OK | grep (sin .venv/.git/uv.lock) por tokens, claves AWS/HF/GitHub/OpenAI, authtoken y URLs ngrok: sin secretos. El notebook de Colab solo menciona `NGROK_TOKEN` desde Secrets de Colab. `.env` está ignorado y no está en git; `.env.example` solo trae placeholders. |

## Nombres generados en la verificación
- Con `generate_novel_name`: pananosaurus, yangxasaurus, hencherosaurus, mimonosaurus, gontosaurus, krotiancondolus, stognosaurus, eltersaurus, alatopelta, dargasaurus.
- Vía la web: Cerpaliraptor, Colodon.

## Tiempos
| Operación | Tiempo |
|---|---|
| POST /api/dino/new (1.º, arranque en frío del modelo y de sd-turbo) | 466,9 s |
| POST /api/dino/new (2.º) | 248,6 s |
| /api/chat (5 turnos) | 19,2 / 8,7 / 9,6 / 8,6 / 5,0 s |
| Descarga de imagen | 0,7 s |
| GET / y GET /api/info | 0,26 s y 0,19 s |
| Generar 10 nombres en local | 0,18 s |

Nuevo Dinosaurio tarda 4 a 8 min con Ollama y sd-turbo en CPU (m5.xlarge): hay que avisarlo en la demo. El chat responde en 5 a 20 s.

## Notas de despliegue
- Tras reiniciar el notebook, `images` seguía sirviendo `amused/amused-512` porque Docker lo arrancaba solo con el código viejo antes del `git pull`. Se cambió `docker start images` por `docker restart images` en `sagemaker/on-start.sh` (local y en AWS) y tras reiniciar `/health` devuelve `stabilityai/sd-turbo`. Falta hacer commit de ese cambio del script.
