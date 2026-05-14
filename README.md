# Solo_5to_cohorte

Proyecto **independiente** del directorio `Bigdata_Proyecto_final`: aquí está **todo el código** del motor SATE-SR (copia en un solo archivo `motor_sate_solo5to.py`). **No importa** `proyecto_sate_curso.py`.

## Qué hace

1. Lee en MongoDB la colección **`primer_bimestre`** para el **año lectivo ancla** (por defecto `2025`, configurable con `--anio` o `ANIO_LECTIVO` en `.env`).
2. Obtiene los **DNI** de alumnos en **5.° de secundaria** (`nivel_educativo: "secundaria"` y grado numérico **5** en `GRADO` / `Grado`).
3. Ejecuta el **mismo pipeline SATE** (ETL, merge, predicción, métricas) **solo** para esos DNI en ese año.
4. Opcionalmente genera **PNG**, **CSV** ML/Power BI y el **`--integrador`** (Mongo JSON, SQLite, sklearn).

Las salidas se escriben dentro de **esta carpeta** (`salida_ml/`, `salida_reportes/solo5to/`), para no mezclarlas con el proyecto principal.

## Requisitos

- Python 3.10+
- Misma base Mongo que uses en el curso. La lista de alumnos de **5.°** se obtiene en este orden: `primer_bimestre` (idealmente con `nivel_educativo: secundaria`), si no hay coincidencias **solo por año y grado 5** en `primer_bimestre`, y si aún no hay, en **`asistencia`** por año y grado 5. Se ignoran filas marcadas explícitamente como **primaria**. El ETL con lista de DNI **no** exige `nivel_educativo` en todas las colecciones (datos reales a menudo no lo tienen).

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

Archivos típicos generados:

- `salida_ml/dataset_ml_solo5to.csv`
- `salida_ml/dataset_powerbi_solo5to.csv`
- `salida_ml/reporte_dashboard_completo_solo5to.csv`

## Nota sobre el proyecto principal

En `Bigdata_Proyecto_final` sigue existiendo **`proyecto_sate_curso.py`** para el análisis de **todos** los alumnos del año. Esta carpeta es solo la variante **5.° secundaria** con código duplicado a propósito, como pediste.
