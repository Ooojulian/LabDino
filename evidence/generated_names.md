# Nombres generados

Estrategia usada para los 10 nombres: **top-p=0.9 (T=1.0)**. Nombre elegido: **Doractodon**.

## Los 10 nombres (ninguno está en el conjunto de entrenamiento)

1. Alapentosaurus
2. Salimimus
3. Gatarisaurus
4. Doractodon
5. Kematentaceratops
6. Revecterris
7. Mamusaurus
8. Sauroreptor
9. Yadanoceratops
10. Drachyrosuchus

## Comparación de estrategias

Misma semilla para todas, 1000 muestras por estrategia. La novedad se mide contra todo el dataset (train+val).

| Estrategia | Temperatura | top-k | top-p | Copias del dataset | Repetidos | Longitud media | < 3 caracteres |
|---|---|---|---|---|---|---|---|
| T=0.5 | 0.5 | - | - | 7.0% | 14.5% | 11.57 | 0.0% |
| T=1.0 | 1.0 | - | - | 1.0% | 0.2% | 11.97 | 0.2% |
| T=1.5 | 1.5 | - | - | 0.1% | 0.3% | 13.15 | 0.8% |
| top-k=5 (T=1.0) | 1.0 | 5 | - | 4.4% | 7.6% | 11.63 | 0.0% |
| top-k=3 (T=1.0) | 1.0 | 3 | - | 10.7% | 38.4% | 11.50 | 0.0% |
| top-p=0.9 (T=1.0) | 1.0 | - | 0.9 | 1.2% | 1.1% | 11.65 | 0.0% |

## Análisis

- **Temperatura baja (0.5)** concentra la probabilidad en los caracteres más probables: el modelo reproduce con frecuencia nombres reales (7.0 % de copias) y repite mucho (14.5 % de muestras repetidas). Es poco útil para crear especies nuevas.
- **Temperatura 1.0** muestrea la distribución aprendida sin modificarla: casi no copia (1.0 %) ni repite (0.2 %), pero a veces produce nombres de menos de 3 caracteres (0.2 %).
- **Temperatura 1.5** aplana la distribución: la tasa de copias es mínima (0.1 %), pero los nombres se alargan (13.15 caracteres de media frente a ~11.6 de las demás) y aparecen más cortos degenerados (0.8 %) y secuencias menos pronunciables (por ejemplo "pocavclikamidocrechiy").
- **Top-k** recorta la cola pero deja la parte alta intacta: k=5 sube las copias a 4.4 % y los repetidos a 7.6 %, y k=3 llega a 10.7 % y 38.4 %. A menor k, menos diversidad.
- **Top-p = 0.9 (T=1.0)** elimina la cola improbable sin estrechar tanto como top-k: 1.2 % de copias, 1.1 % de repetidos, 0 % de nombres cortos y longitud media 11.65 (el dataset tiene 11.96). Frente a T=1.0 puro, que tiene la longitud media más fiel al dataset (11.97), top-p evita los nombres degenerados de menos de 3 caracteres. Por eso se eligió para generar los 10 nombres.
- Los 10 nombres reutilizan patrones reales del dataset (-saurus, -odon, -ceratops, -raptor) sin copiar ninguno. Se elige **Doractodon**: es pronunciable, de longitud moderada y su terminación -odon es de las que se usarán como ejemplo en la etapa de identidad con Ollama.
- Limitación: la novedad se mide solo por coincidencia exacta con el dataset; un nombre puede ser muy parecido a uno real (p. ej. variantes de "-saurus") y seguir contando como novedoso.
