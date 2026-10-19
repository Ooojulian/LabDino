# Resumen de entrenamiento

**Criterio de elección (fijado antes de entrenar):** menor pérdida de validación

**Configuración ganadora:** B

| Métrica | A | B |
|---|---|---|
| Arquitectura | GRU | GRU |
| Capas | 1 | 2 |
| Hidden | 128 | 256 |
| Dropout | 0.0 | 0.2 |
| Épocas corridas | 23 | 18 |
| Máx. épocas | 100 | 100 |
| Paciencia | 10 | 10 |
| Pérdida | CrossEntropyLoss(ignore_index=<PAD>) | CrossEntropyLoss(ignore_index=<PAD>) |
| Optimizador | Adam | Adam |
| lr | 0.003 | 0.003 |
| Batch | 64 | 64 |
| Parámetros | 66877 | 625853 |
| Mejor val loss | 1.5716 | 1.5589 |
| Época del mejor | 13 | 8 |

Notas: la pérdida de entrenamiento es el promedio por token durante la época (con dropout activo); la de validación se calcula en modo eval sin gradientes. El checkpoint guardado es el de la mejor época de validación, no el último.

![comparativa](loss_curves_comparison.png)
