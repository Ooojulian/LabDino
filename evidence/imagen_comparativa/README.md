# Comparativa de generadores de imagen (Doractodon, seed 0, CPU)

| Archivo | Modelo | Steps | Tiempo CPU | Observación |
|---|---|---|---|---|
| `1_amused512_prompt_original.png` | amused/amused-512, prompt "paleontological illustration" | 12 | ~55 s | Estilo anterior: caricaturesco/plano, aspecto de ilustración. |
| `2_amused512_prompt_fotorrealista.png` | amused/amused-512, prompt fotográfico | 24 | ~94 s | Más realista pero con textura artificial, el doble de lento y el rasgo distintivo no se ve. |
| `3_sdturbo_4pasos.png` | stabilityai/sd-turbo, prompt fotográfico | 4 | ~37-45 s | Fotografía creíble: marrón terroso, cuerpo robusto, llanura árida, luz natural. |

Tiempos medidos en una CPU de 8 hilos (torch CPU, fp32), sin GPU. En el notebook SageMaker (4 vCPU) se estiman ~50 s para sd-turbo.

## Por qué sd-turbo
- Más realista: amused-512 sale caricaturesco incluso con un prompt fotorrealista; sd-turbo da aspecto de fotografía de vida salvaje.
- Tamaño similar (~1B de parámetros, ~2 GB en fp16), dentro de lo que permite la guía ("otra alternativa de tamaño similar").
- Corre en CPU: modelo destilado, 512x512 con 1-4 pasos y guidance 0 (por eso el prompt negativo solo se registra en el json); más rápido que amused con 12-24 pasos.
- Limitación: ninguna de las imágenes dibuja con claridad la protuberancia cónica del hocico; la imagen es aproximada a la ficha.
- Seeds probadas con sd-turbo: 0 (elegida), 1, 2, 3. La 1 muestra un carnívoro con la boca abierta (no coherente con la dieta herbívora de la ficha).
