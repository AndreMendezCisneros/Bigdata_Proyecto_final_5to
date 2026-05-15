# Solo_5to_cohorte

Proyecto **independiente** del directorio `Bigdata_Proyecto_final`: aquí está **todo el código** del motor SATE-SR en un solo archivo [`motor_sate_solo5to.py`](motor_sate_solo5to.py). **No importa** `proyecto_sate_curso.py`.

Las salidas se escriben **solo dentro de esta carpeta** (`salida_ml/`, `salida_reportes/solo5to/`). **No** actualiza archivos en `Bigdata_Proyecto_final/salida_ml/` (por ejemplo `reporte_dashboard_completo.csv` del proyecto grande es otro flujo).

---

## Qué hace el código (resumen ejecutivo)

1. Conecta a **MongoDB** usando `MONGODB_URI` y `MONGODB_DB_NAME` (variables de entorno).
2. Arma la lista de **DNI únicos** de alumnos en **5.° de secundaria** del **año lectivo ancla** (p. ej. 2025).
3. Para **solo esos DNI** y **solo ese año** en las colecciones con `anio_lectivo`, ejecuta un **ETL**: lee asistencia, nómina, tres bimestres de notas, incidentes y encuesta; integra todo en una fila por estudiante.
4. Opcionalmente carga **historial de notas** de los **cuatro años anteriores** al ancla (mismos DNI), incluyendo I–IV bimestre cuando existan datos en `cuarto_bimestre`.
5. Calcula la **nota proyectada al IV bimestre** del año ancla, la **clasificación aprueba/desaprueba** y métricas de **validación temporal** (explicadas abajo).
6. Opcionalmente escribe **CSV**, **PNG** y, con `--integrador`, artefactos extra en `salida_ml/`.

---

## Flujo del programa (paso a paso)

### 1. Carga de configuración y CLI

- Al ejecutar `motor_sate_solo5to.py`, se cargan variables desde `.env` (esta carpeta, luego `Bigdata_Proyecto_final`, luego `escuela_datos_sinteticos`).
- Argumentos útiles: `--anio`, `--plots`, `--plots-dir`, `--export-ml [ruta.csv]`, `--json`, `--integrador`.

### 2. Cohorte: ¿quiénes son los “solo 5.°”?

Función **`dnis_quintos_secundaria_anio`**: obtiene DNIs del año ancla en este **orden de intentos**:

1. **`primer_bimestre`**: `anio_lectivo` + `nivel_educativo: secundaria` y grado numérico **5** en `GRADO` / `Grado`.
2. Si no hay resultados: **`primer_bimestre`** solo con año y grado 5 (sin exigir nivel; se excluye primaria explícita).
3. Si aún no hay: **`asistencia`** con año y grado 5.

Así se adapta a datos reales donde no siempre viene `nivel_educativo` en todas las colecciones.

### 3. Análisis principal: `ejecutar_analisis_sate`

Recibe `mongodb_uri`, `database_name`, `anio_lectivo` (ancla) y `dni_allowlist` (el conjunto de DNIs del paso anterior). Con eso:

**Filtro Mongo:** `anio_lectivo` del ancla + lista de DNI (`$or` sobre distintas claves de documento donde puede venir el DNI). **No** se exige `nivel_educativo` en el find general del ETL para no vaciar nómina/asistencia reales.

**ETL por colecciones (mismo año ancla):**

| Orden | Colección | Qué extrae |
|-------|-----------|------------|
| 1 | `asistencia` | Por DNI (y clave nombre): suma días presente/ausente; **% de faltas**; marca `Analisis_Asistencia` **1** si las faltas son bajo el umbral crítico (`MODEL_CONFIG["umbral_faltas_critico"]`, por defecto 30%), **0** si hay riesgo. |
| 2 | `nomina` | Padre/madre vive, trabaja el estudiante, discapacidad, situación de matrícula; puntaje compuesto → `Analisis_Situacion_Familiar` **1** si puntaje ≥ 4, **0** si no. |
| 3–5 | `primer_bimestre`, `segundo_bimestre`, `tercer_bimestre` | `PROMEDIO_APRENDIZAJE_AUTONOMO` convertido con letras **C, B, A, AD** a números (`MODEL_CONFIG["conversion_notas"]`) → `NotaBim1`, `NotaBim2`, `NotaBim3`. Si falta conversión, se usa **5** como valor por defecto en el año ancla. |
| 6 | `incidente` | Por nombre del estudiante: si existe alguna falta **no leve**, `Analisis_Incidencias` = **0**; si solo leves o sin graves, **1**. |
| 7 | `encuesta` | Texto abierto (varias columnas candidatas) → análisis de sentimiento → `Analisis_Sentimiento_Estudiante` **1** positivo/neutro, **0** negativo. |

**Merge:** se unen todas las fuentes por **DNI** (incidentes enlazan por nombre → DNI). Resultado: diccionario `estudiantes_map` con una entrada por alumno.

**Historial (años ancla − 4 … ancla − 1):** función **`cargar_resumen_notas_historial`**. Para cada año lee `primer_bimestre`, `segundo_bimestre`, `tercer_bimestre` y **`cuarto_bimestre`**. Solo cuenta notas con conversión válida (no rellena con 5 los huecos del historial). Por año, si hay al menos `historial_min_bims_por_anio` bimestres con nota, calcula media anual; resume en `hist_media_global`, `hist_n_anios_utiles`, etc.

**Predicción por fila:**

1. **`proyectar_nota_nucleo_b4`**: con `NotaBim1–3` del ancla, regresión / manejo de outliers y tope de cambio respecto al III bimestre (`max_proyeccion_cambio`).
2. **`aplicar_guia_historial`**: si hay suficientes años útiles y `historial_habilitado`, ajusta suavemente la nota núcleo hacia la media histórica (con límites en `MODEL_CONFIG`).
3. **`_castigo_sate_fila`**: resta hasta **4 puntos** en total según factores en riesgo (cada factor en **0** suma su peso en `pesos_penalizacion`, por defecto 1.0 cada uno).
4. Resultado acotado a la escala **`nota_escala`** (5–20).
5. **`clasificar_resultado`**: compara la nota final con **`umbral_aprobacion`** (por defecto **12**) → binario aprueba/desaprueba.

Columnas expuestas por alumno incluyen `Nota_Nucleo_B4`, `Nota_Proyectada_B4`, campos `Hist_*`, factores y notas bimestrales.

**Validación temporal (métricas en consola y CSV):** no valida el IV bimestre real de 2025. Es un **proxy**: con **Bim1 y Bim2** del mismo año se estima **Bim3** y se compara con el **Bim3 real** (en binario aprueba/desaprueba). De ahí salen precisión, recall, F1, AUC-ROC y matriz de confusión. Sirve como chequeo interno del enfoque “notas + castigos”, no como certificación del error del IV bimestre.

### 4. Salida del `main()`

- Imprime resumen, matriz, factores y lista corta de desapruebas.
- Con **`--export-ml`**: escribe CSV en la carpeta del archivo elegido (por defecto `salida_ml/dataset_ml_solo5to.csv`) y además `dataset_powerbi_solo5to.csv` y `reporte_dashboard_completo_solo5to.csv` en el mismo directorio.
- Con **`--plots`**: PNG en `salida_reportes/solo5to/` (distribución predicción, histograma nota proyectada, heatmap correlaciones, factores apilados, matriz confusión).
- Con **`--json`**: imprime el diccionario `resultado` completo por stdout.
- Con **`--integrador`**: pipeline adicional (JSON/SQLite/sklearn según implementación en el mismo archivo).

**Lectura vs escritura:** el motor **solo lee** Mongo y **escribe archivos locales**. **No** modifica colecciones en Mongo.

---

## `MODEL_CONFIG` (qué puedes tunear sin tocar lógica dispersa)

En [`motor_sate_solo5to.py`](motor_sate_solo5to.py): conversión de letras a notas, umbral de aprobación, umbral de faltas para asistencia, pesos del castigo SATE, tope de proyección, escala de nota, y bloque **`historial_*`** (activar/desactivar, peso máximo, mínimos de años/bimestres, dispersión, delta máximo del ajuste).

---

## Requisitos

- Python 3.10+
- MongoDB accesible con las colecciones que usa el ETL (nombres indicados arriba).

## Instalación

```powershell
cd Solo_5to_cohorte
py -m pip install -r requirements.txt
```

## Configuración

Copia [`.env.example`](.env.example) a **`.env`** con `MONGODB_URI` (y opcionalmente `MONGODB_DB_NAME`, `ANIO_LECTIVO`).

Si no hay `.env` aquí, el motor intenta cargar (en orden):

1. `Solo_5to_cohorte/.env`
2. `../Bigdata_Proyecto_final/.env`
3. `../escuela_datos_sinteticos/.env`

## Ejecución

```powershell
cd Solo_5to_cohorte
py motor_sate_solo5to.py --plots --export-ml
py motor_sate_solo5to.py --plots --export-ml --integrador
```

Archivos típicos generados en **esta** carpeta:

- `salida_ml/dataset_ml_solo5to.csv` — columnas tipo ML (`snake_case`).
- `salida_ml/dataset_powerbi_solo5to.csv` — mismo contenido enriquecido para Power BI (BOM UTF-8).
- `salida_ml/reporte_dashboard_completo_solo5to.csv` — una fila por alumno + columnas KPI repetidas para tableros.

## Nota sobre el proyecto principal

En `Bigdata_Proyecto_final` sigue existiendo **`proyecto_sate_curso.py`** para el análisis de **todos** los alumnos del año y exporta, entre otros, `salida_ml/reporte_dashboard_completo.csv` **en esa ruta**. Esta carpeta es la variante **solo 5.° secundaria** con código autocontenido a propósito.
