# -*- coding: utf-8 -*-
"""
Solo_5to_cohorte — motor SATE-SR v3.0 (copia autocontenida)
===========================================================
Paquete independiente: NO importa `proyecto_sate_curso.py` del proyecto principal.

Criterio de alumnos: quienes figuran en **5.° de secundaria** en el año lectivo
ancla (por defecto 2025), detectados en `primer_bimestre` (DNI + grado 5 +
`nivel_educativo: secundaria`). El análisis ETL y predicción usa **solo esos DNIs**
en ese año.

Ejecución (desde esta carpeta):
  py motor_sate_solo5to.py --plots --export-ml

Variables: MONGODB_URI, MONGODB_DB_NAME, ANIO_LECTIVO (archivo `.env` aquí o en carpetas vecinas).
"""
from __future__ import annotations

from typing import Dict, List, Any, Optional, Set
from pymongo import MongoClient
from datetime import datetime
import re
import math
import logging
# Configurar codificación UTF-8 para Windows
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configurar logging para que Flask muestre los mensajes
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Importaciones opcionales - si scikit-learn no está disponible, usar implementación manual
try:
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, 
        roc_auc_score, confusion_matrix
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[INFO] scikit-learn no disponible, usando implementacion manual de metricas")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("[INFO] numpy no disponible, usando implementacion manual")

# Importación opcional de pysentimiento para análisis de sentimiento avanzado
HAS_PYSENTIMIENTO = False
sentiment_analyzer = None
try:
    from pysentimiento import create_analyzer
    HAS_PYSENTIMIENTO = True
    print("[INFO] pysentimiento disponible, inicializando analizador de sentimientos...")
    try:
        sentiment_analyzer = create_analyzer(task="sentiment", lang="es")
        print("[OK] Analizador de sentimientos pysentimiento inicializado correctamente")
    except Exception as e:
        print(f"[ADVERTENCIA] Error inicializando pysentimiento: {e}")
        print("[INFO] Usando analizador manual de sentimientos como fallback")
        HAS_PYSENTIMIENTO = False
        sentiment_analyzer = None
except ImportError:
    print("[INFO] pysentimiento no disponible, usando analizador manual de sentimientos")
    print("[INFO] Para mejor precisión, instala: py -m pip install pysentimiento torch transformers")

MODEL_CONFIG = {
    "version": "3.1.0",
    "conversion_notas": {
        'C': 5,   # En Inicio
        'B': 13,  # En Proceso
        'A': 16,  # Logro Esperado
        'AD': 19  # Logro Destacado
    },
    "umbral_aprobacion": 12,
    "umbral_faltas_critico": 30,  # Porcentaje
    "pesos_penalizacion": {
        "asistencia": 1.0,
        "incidencias": 1.0,
        "sentimiento": 1.0,
        "familia": 1.0
    },
    "max_proyeccion_cambio": 4,
    "nota_escala": [5, 20],
    # Guía longitudinal: años anteriores al ancla (solo refina; núcleo = B1–B3 del anio_lectivo)
    "historial_habilitado": True,
    "historial_peso_max": 0.35,
    "historial_min_anios_utiles": 2,
    "historial_min_bims_por_anio": 2,
    "historial_umbral_std_anual": 4.0,
    "historial_max_delta": 1.0,
}

CONFIG = MODEL_CONFIG


def convertir_calificacion(valor: Any) -> Optional[int]:
    """Convierte calificación cualitativa a numérica"""
    if not valor or valor is None:
        return None
    
    val_upper = str(valor).strip().upper()
    return MODEL_CONFIG["conversion_notas"].get(val_upper)


def normalizar_dni(doc: Dict) -> Optional[str]:
    """Normaliza columnas DNI con diferentes nombres"""
    for key in ('DNI', 'Nº', 'dni', 'Numero de documento', 'Número de documento', 'Documento'):
        if key in doc and doc[key] not in (None, ''):
            s = str(doc[key]).strip()
            if s:
                return s
    return None


def extraer_texto_abierto_encuesta(doc: Dict) -> Optional[str]:
    """Texto libre de encuesta (Google Forms): pregunta 17 / sugerencia a directiva."""
    for k in (
        'sugerencia_sentimientos',
        'sugerencia_sentimiento',
        'sentimiento',
        'sugerencia',
        'comentario',
        'texto',
    ):
        v = doc.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    for k in doc.keys():
        if k == '_id':
            continue
        kl = str(k).lower()
        if any(
            x in kl
            for x in (
                'sugerir',
                'directiva',
                'por qué',
                'por que',
                'experiencia (ej',
                'mejorar tu experiencia',
            )
        ):
            v = doc.get(k)
            if v is not None and str(v).strip():
                return str(v).strip()
    return None


def normalizar_nombres(doc: Dict) -> str:
    """Normaliza columnas de nombres de estudiantes"""
    posibles_nombres = [
        'Apellidos_Nombres',
        'APELLIDOS_Y_NOMBRES',
        'ALUMNOS/AS',
        'Nombre y Apellido',
        'nombre_completo',
        'Apellidos Nombres',
        'NOMBRE_COMPLETO'
    ]
    
    for nombre_col in posibles_nombres:
        if nombre_col in doc and doc[nombre_col]:
            valor = str(doc[nombre_col]).strip()
            if valor:
                return valor
    
    # Buscar columnas que contengan 'apellido' o 'nombre'
    for key in doc.keys():
        key_lower = key.lower()
        if ('apellido' in key_lower or 'nombre' in key_lower) and \
           key.upper() != 'DNI' and key != 'Nº' and key != '_id':
            valor = str(doc[key]).strip()
            if valor:
                return valor
    
    return ''


def analizar_sentimiento_espanol(texto: Any) -> int:
    """
    Analiza sentimiento en español usando pysentimiento si está disponible,
    sino usa análisis manual basado en palabras clave.
    
    Retorna:
        1 = Sentimiento positivo o neutro (sin riesgo)
        0 = Sentimiento negativo (con riesgo)
    """
    if not texto or str(texto).strip() == '':
        return 1  # Ausencia = Positivo por defecto
    
    texto_limpio = str(texto).strip()
    
    # Casos especiales que son neutrales/positivos
    casos_neutros = [
        'nada', '.', '', 'ninguno', 'ninguna', 'n/a', 
        'sin comentarios', 'sin comentario', 'no hay', 
        'ningún', 'ninguna observación'
    ]
    if texto_limpio.lower() in casos_neutros:
        return 1  # Neutro = sin riesgo
    
    # Usar pysentimiento si está disponible (más preciso)
    if HAS_PYSENTIMIENTO and sentiment_analyzer is not None:
        try:
            resultado = sentiment_analyzer.predict(texto_limpio)
            sentimiento = resultado.output
            # pysentimiento retorna: 'POS', 'NEU', 'NEG'
            return 1 if sentimiento in ['POS', 'NEU'] else 0
        except Exception as e:
            logger.warning(f'Error usando pysentimiento, usando método manual: {e}')
            # Continuar con método manual si falla
    
    # Método manual (fallback o si pysentimiento no está disponible)
    texto_limpio_lower = texto_limpio.lower()
    
    # Palabras negativas comunes en español (expandida)
    # Palabras negativas FUERTES (peso 2) - indican claramente sentimiento negativo
    palabras_negativas_fuertes = [
        'no me gusta', 'no me gustó', 'odio', 'odiar', 'terrible', 'horrible',
        'aburrido', 'aburrida', 'aburran', 'aburren', 'monótona', 'monótono',
        'triste', 'tristeza', 'enojado', 'enojada', 'preocupado', 'preocupada',
        'cansado', 'cansada', 'estresado', 'estresada', 'molesto', 'molesta',
        'frustrado', 'frustrada', 'desanimado', 'desanimada', 'preocupante',
        'injusto', 'injusta', 'maltrato', 'violencia', 'peleas', 'pelea',
        'conflicto', 'conflictos', 'agresión', 'agresiones', 'miedo', 'temor',
        'ansiedad', 'nervioso', 'nerviosa', 'inseguro', 'insegura', 'solo', 'sola',
        'solitario', 'solitario', 'abandonado', 'abandonada', 'discriminar',
        'discriminación', 'bullying', 'acoso', 'burla', 'burlas', 'desfasados',
        'desfasadas', 'desactualizado', 'desactualizada'
    ]
    
    # Palabras negativas REGULARES (peso 1)
    palabras_negativas = [
        'mal', 'malo', 'mala', 'problema', 'problemas', 'difícil', 'dificil',
        'preocupación'
    ]
    
    # Palabras positivas comunes en español (expandida)
    palabras_positivas = [
        'bien', 'bueno', 'buena', 'excelente', 'genial', 'me gusta', 'me gustó',
        'feliz', 'contento', 'contenta', 'satisfecho', 'satisfecha', 'agradecido',
        'agradecida', 'perfecto', 'perfecta', 'maravilloso', 'maravillosa',
        'mejor', 'mejora', 'mejorado', 'mejorada', 'progreso', 'avance', 'avances',
        'apoyo', 'ayuda', 'compañerismo', 'amistad', 'respeto', 'tranquilo', 'tranquila',
        'motivado', 'motivada', 'entusiasmado', 'entusiasmada', 'orgulloso', 'orgullosa',
        'alegre', 'alegría', 'divertido', 'divertida', 'emocionado', 'emocionada',
        'esperanza', 'optimista', 'confianza', 'seguro', 'segura', 'cómodo', 'cómoda'
    ]
    
    # Contar ocurrencias (usar regex para palabras completas)
    # Palabras negativas fuertes tienen peso 2
    negativas_fuertes = sum(len(re.findall(rf'\b{re.escape(palabra)}\b', texto_limpio_lower)) 
                           for palabra in palabras_negativas_fuertes) * 2
    # Palabras negativas regulares tienen peso 1
    negativas_regulares = sum(len(re.findall(rf'\b{re.escape(palabra)}\b', texto_limpio_lower)) 
                             for palabra in palabras_negativas)
    
    negativas = negativas_fuertes + negativas_regulares
    positivas = sum(len(re.findall(rf'\b{re.escape(palabra)}\b', texto_limpio_lower)) 
                    for palabra in palabras_positivas)
    
    # Debug: si encontramos palabras negativas, loguear
    if negativas > 0:
        logger.debug(f'Texto analizado manualmente ({negativas} neg, {positivas} pos): {texto_limpio[:100]}')
    
    # Lógica mejorada:
    # 1. Si hay palabras negativas fuertes (peso 2), es más probable que sea negativo
    # 2. Si hay más palabras negativas que positivas (considerando pesos), es negativo
    # 3. Si hay al menos una palabra negativa fuerte y ninguna positiva, es negativo
    if negativas_fuertes > 0 and positivas == 0:
        return 0  # Claramente negativo (tiene palabras negativas fuertes y ninguna positiva)
    
    # Si hay palabras negativas fuertes, ser más estricto
    if negativas_fuertes > 0:
        # Si las negativas (con peso) superan a las positivas, es negativo
        if negativas > positivas:
            return 0
        # Si hay palabras negativas fuertes pero también positivas, considerar el contexto
        # Si hay más negativas fuertes que positivas, es negativo
        if negativas_fuertes / 2 > positivas:
            return 0
    
    return 0 if negativas > positivas else 1


def proyectar_nota_nucleo_b4(fila: Dict, config: Dict = MODEL_CONFIG) -> float:
    """
    Proyección solo a partir de NotaBim1–3 del año ancla (sin SATE ni historial).
    Incluye detección de outliers y tope de cambio respecto a Bim3.
    """
    notas = [fila.get('NotaBim1', 5), fila.get('NotaBim2', 5), fila.get('NotaBim3', 5)]

    nota_min, nota_max = config["nota_escala"]
    notas_validadas = [max(nota_min, min(nota_max, n)) for n in notas]

    if HAS_NUMPY:
        media = float(np.mean(notas_validadas))
        desviacion = float(np.std(notas_validadas)) if len(notas_validadas) > 1 else 1.0
    else:
        media = sum(notas_validadas) / len(notas_validadas)
        variance = sum((x - media) ** 2 for x in notas_validadas) / len(notas_validadas)
        desviacion = math.sqrt(variance) if variance > 0 else 1.0

    z_scores = [abs((n - media) / (desviacion if desviacion > 0 else 1)) for n in notas_validadas]
    tiene_outlier = any(z > 2 for z in z_scores)

    if tiene_outlier:
        cambio = (notas_validadas[2] - notas_validadas[0]) / 2
        proyeccion_b4 = notas_validadas[2] + cambio
    else:
        n = 3
        sum_x = 6
        sum_y = sum(notas_validadas)
        sum_xy = notas_validadas[0] * 1 + notas_validadas[1] * 2 + notas_validadas[2] * 3
        sum_x2 = 14

        denom = (n * sum_x2 - sum_x * sum_x)
        if denom == 0:
            proyeccion_b4 = notas_validadas[2]
        else:
            m = (n * sum_xy - sum_x * sum_y) / denom
            b = (sum_y - m * sum_x) / n
            proyeccion_b4 = m * 4 + b

    max_cambio = config["max_proyeccion_cambio"]
    proyeccion_b4 = max(
        notas_validadas[2] - max_cambio,
        min(notas_validadas[2] + max_cambio, proyeccion_b4),
    )
    return float(proyeccion_b4)


def _castigo_sate_fila(fila: Dict, config: Dict = MODEL_CONFIG) -> float:
    pesos = config["pesos_penalizacion"]
    return (
        (1 - fila.get('Analisis_Asistencia', 1)) * pesos["asistencia"] +
        (1 - fila.get('Analisis_Incidencias', 1)) * pesos["incidencias"] +
        (1 - fila.get('Analisis_Sentimiento_Estudiante', 1)) * pesos["sentimiento"] +
        (1 - fila.get('Analisis_Situacion_Familiar', 1)) * pesos["familia"]
    )


def _std_muestral(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return math.sqrt(var) if var > 0 else 0.0


def aplicar_guia_historial(
    nota_nucleo: float,
    resumen: Optional[Dict[str, Any]],
    config: Dict = MODEL_CONFIG,
) -> float:
    """
    Acerca suavemente la proyección núcleo hacia la media histórica anual del alumno
    cuando hay suficientes años con datos; acota el movimiento y penaliza dispersión alta.
    """
    if not resumen or not config.get("historial_habilitado", True):
        return nota_nucleo
    mu = resumen.get("hist_media_global")
    if mu is None:
        return nota_nucleo
    n = int(resumen.get("hist_n_anios_utiles") or 0)
    min_anios = int(config.get("historial_min_anios_utiles", 2))
    if n < min_anios:
        return nota_nucleo
    alpha = float(config.get("historial_peso_max", 0.35)) * min(1.0, n / 4.0)
    std_a = float(resumen.get("hist_std_medias_anuales") or 0.0)
    if std_a > float(config.get("historial_umbral_std_anual", 4.0)):
        alpha *= 0.5
    nota_adj = nota_nucleo + alpha * (float(mu) - nota_nucleo)
    md = float(config.get("historial_max_delta", 1.0))
    nota_adj = max(nota_nucleo - md, min(nota_nucleo + md, nota_adj))
    lo, hi = config["nota_escala"]
    return max(lo, min(hi, nota_adj))


def proyectar_nota_robusta(
    fila: Dict,
    config: Dict = MODEL_CONFIG,
    resumen_historial: Optional[Dict[str, Any]] = None,
) -> float:
    """Proyección robusta: núcleo año ancla, guía opcional por historial, penalización SATE."""
    nota_min, nota_max = config["nota_escala"]
    nucleo = proyectar_nota_nucleo_b4(fila, config)
    nucleo = aplicar_guia_historial(nucleo, resumen_historial, config)
    castigo = _castigo_sate_fila(fila, config)
    nota_final = nucleo - castigo
    return max(nota_min, min(nota_max, nota_final))


def clasificar_resultado(nota: float, umbral: float = MODEL_CONFIG["umbral_aprobacion"]) -> int:
    """Clasifica resultado: APRUEBA (1) vs DESAPRUEBA (0)"""
    return 1 if nota >= umbral else 0


def calcular_metricas(y_true: List[int], y_pred: List[int], y_scores: Optional[List[float]] = None) -> Dict:
    """
    Calcula métricas de validación del modelo
    
    Args:
        y_true: Valores reales (binarios: 0 o 1)
        y_pred: Predicciones binarias (0 o 1)
        y_scores: Scores continuos para calcular AUC-ROC (opcional, si no se proporciona usa y_pred)
                  Usar notas proyectadas como scores mejora significativamente el AUC-ROC
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "auc_roc": 0.5,
            "matriz_confusion": {
                "verdaderos_positivos": 0,
                "falsos_positivos": 0,
                "verdaderos_negativos": 0,
                "falsos_negativos": 0
            }
        }
    
    # Calcular matriz de confusión manualmente
    tp = fp = tn = fn = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            tn += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
    
    # Calcular métricas manualmente
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # AUC-ROC: Usar scores continuos si están disponibles (MUCHO MÁS PRECISO)
    if y_scores is not None and len(y_scores) == len(y_true):
        # Usar scores continuos para un cálculo mucho más preciso del AUC-ROC
        if HAS_SKLEARN:
            try:
                auc_roc = roc_auc_score(y_true, y_scores)
                logger.info(f'AUC-ROC calculado con scores continuos (sklearn): {auc_roc:.4f}')
            except (ValueError, Exception) as e:
                logger.warning(f'Error calculando AUC-ROC con sklearn, usando método manual: {e}')
                auc_roc = calcular_auc_roc_manual_con_scores(y_true, y_scores)
        else:
            auc_roc = calcular_auc_roc_manual_con_scores(y_true, y_scores)
            logger.info(f'AUC-ROC calculado con scores continuos (manual): {auc_roc:.4f}')
    else:
        # Fallback: usar predicciones binarias (menos preciso, pero funciona)
        if HAS_SKLEARN:
            try:
                # sklearn puede calcular AUC-ROC con predicciones binarias, pero es menos preciso
                auc_roc = roc_auc_score(y_true, y_pred)
                logger.warning('AUC-ROC calculado con predicciones binarias (menos preciso). Usa y_scores para mejor precisión.')
            except ValueError:
                auc_roc = calcular_auc_roc_manual(y_true, y_pred)
        else:
            auc_roc = calcular_auc_roc_manual(y_true, y_pred)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc_roc": float(auc_roc),
        "matriz_confusion": {
            "verdaderos_positivos": int(tp),
            "falsos_positivos": int(fp),
            "verdaderos_negativos": int(tn),
            "falsos_negativos": int(fn)
        }
    }


def calcular_auc_roc_manual_con_scores(y_true: List[int], y_scores: List[float]) -> float:
    """
    Calcula AUC-ROC manualmente usando scores continuos (MÁS PRECISO)
    
    Este método es mucho más preciso que usar predicciones binarias porque
    considera la "confianza" del modelo (nota proyectada) en lugar de solo
    la clasificación final (aprueba/desaprueba).
    """
    if len(y_true) == 0 or len(y_scores) == 0:
        return 0.5
    
    positivos_reales = sum(1 for y in y_true if y == 1)
    negativos_reales = sum(1 for y in y_true if y == 0)
    
    if positivos_reales == 0 or negativos_reales == 0:
        return 0.5
    
    # Crear pares (real, score)
    pares = [(y_true[i], y_scores[i]) for i in range(len(y_true))]
    
    # Contar pares correctamente ordenados
    pares_correctos = 0
    total_pares = 0
    
    for i in range(len(pares)):
        for j in range(i + 1, len(pares)):
            # Solo comparar pares donde las clases reales son diferentes
            if pares[i][0] != pares[j][0]:
                total_pares += 1
                # Verificar si el orden predicho es correcto
                # Si el positivo real tiene score mayor que el negativo real, está bien ordenado
                if ((pares[i][0] > pares[j][0] and pares[i][1] > pares[j][1]) or
                    (pares[i][0] < pares[j][0] and pares[i][1] < pares[j][1])):
                    pares_correctos += 1
                # Si tienen el mismo score, contar como medio correcto (tie)
                elif pares[i][1] == pares[j][1]:
                    pares_correctos += 0.5
    
    return pares_correctos / total_pares if total_pares > 0 else 0.5


def calcular_auc_roc_manual(y_true: List[int], y_pred: List[int]) -> float:
    """Calcula AUC-ROC manualmente usando el método de pares (con predicciones binarias)"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.5
    
    positivos_reales = sum(1 for y in y_true if y == 1)
    negativos_reales = sum(1 for y in y_true if y == 0)
    
    if positivos_reales == 0 or negativos_reales == 0:
        return 0.5
    
    # Crear pares (real, predicho)
    pares = [(y_true[i], y_pred[i]) for i in range(len(y_true))]
    
    # Contar pares correctamente ordenados
    pares_correctos = 0
    total_pares = 0
    
    for i in range(len(pares)):
        for j in range(i + 1, len(pares)):
            # Solo comparar pares donde las clases reales son diferentes
            if pares[i][0] != pares[j][0]:
                total_pares += 1
                # Verificar si el orden predicho es correcto
                if ((pares[i][0] > pares[j][0] and pares[i][1] >= pares[j][1]) or
                    (pares[i][0] < pares[j][0] and pares[i][1] <= pares[j][1])):
                    pares_correctos += 1
    
    return pares_correctos / total_pares if total_pares > 0 else 0.5


# Columnas que no son días de asistencia (incluye metadatos del import sintético)
ASISTENCIA_METADATA_KEYS = frozenset({
    '_id', 'DNI', 'dni', 'Nº', 'Apellidos_Nombres', 'APELLIDOS_Y_NOMBRES', 'ALUMNOS/AS',
    'SECCIÓN', 'GRADO', 'Seccion', 'Grado', 'anio_lectivo', 'origen_datos', 'nivel_educativo',
})


def _norm_clave_nombre_completo(s: Any) -> str:
    """Clave estable para cruzar nombres (mayúsculas, espacios colapsados)."""
    return " ".join(str(s or "").strip().upper().split())


def _valor_dia_asistencia(val: Any) -> Optional[int]:
    """
    Celda de asistencia: 1 presente, 0/2 falta. Mongo/CSV suelen traer str o float.
    Retorna None si no es un día marcado (vacío / no reconocido).
    """
    if val is None:
        return None
    if isinstance(val, float):
        if math.isnan(val):
            return None
        vi = int(val)
        return vi if vi in (0, 1, 2) else None
    if isinstance(val, int) and not isinstance(val, bool):
        return val if val in (0, 1, 2) else None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "-", ""):
        return None
    try:
        vi = int(float(s.replace(",", ".")))
    except ValueError:
        return None
    return vi if vi in (0, 1, 2) else None


def _columnas_dias_asistencia(doc: Dict[str, Any]) -> list[str]:
    """Solo columnas tipo Día_N (evita contar _id u otras claves)."""
    out: list[str] = []
    for k in doc.keys():
        if k in ASISTENCIA_METADATA_KEYS:
            continue
        ks = str(k).strip()
        if ks.startswith("Día_") or ks.lower().startswith("día_"):
            out.append(k)
    return out


# Claves de DNI habituales en documentos Mongo (alineado con normalizar_dni)
DNI_MONGO_KEYS = ("DNI", "dni", "Nº", "Numero de documento", "Número de documento", "Documento")


def grado_desde_doc(doc: Dict[str, Any]) -> Optional[int]:
    """Grado numérico inicial (p. ej. 5 en '5' o '5°') desde GRADO/Grado; None si no parseable."""
    raw = doc.get("GRADO") if doc.get("GRADO") not in (None, "") else doc.get("Grado")
    if raw is None or str(raw).strip() == "":
        return None
    m = re.match(r"^\s*(\d+)", str(raw).strip())
    return int(m.group(1)) if m else None


def _dnis_grado5_desde_documentos(docs: List[Dict[str, Any]]) -> Set[str]:
    """Un DNI por grado 5; excluye filas con nivel_educativo explícitamente primaria."""
    por_dni: Dict[str, Optional[int]] = {}
    for doc in docs:
        nv = str(doc.get("nivel_educativo") or "").strip().lower()
        if nv == "primaria":
            continue
        dni = normalizar_dni(doc)
        if not dni:
            continue
        g = grado_desde_doc(doc)
        if g is None:
            continue
        prev = por_dni.get(dni)
        if prev is None:
            por_dni[dni] = g
        elif prev != g:
            por_dni[dni] = None
    return {d for d, g in por_dni.items() if g == 5}


def dnis_quintos_secundaria_anio(
    mongodb_uri: str, database_name: str, anio_lectivo: int
) -> Set[str]:
    """
    DNIs en 5.° (grado numérico 5) en el año ancla.

    Orden de intentos (datos reales a menudo sin `nivel_educativo` en bimestres):
    1) `primer_bimestre` con nivel secundaria + año
    2) `primer_bimestre` solo año (excluye documentos marcados como primaria)
    3) `asistencia` solo año (mismo criterio de exclusión)
    """
    proy: Dict[str, int] = {k: 1 for k in ("anio_lectivo", "GRADO", "Grado", "nivel_educativo")}
    for k in DNI_MONGO_KEYS:
        proy[k] = 1

    client = MongoClient(mongodb_uri)
    try:
        db = client[database_name]
        pb = db["primer_bimestre"]

        docs = list(pb.find({"anio_lectivo": anio_lectivo, "nivel_educativo": "secundaria"}, proy))
        out = _dnis_grado5_desde_documentos(docs)
        if out:
            print("[INFO] Lista 5.°: primer_bimestre con nivel_educativo=secundaria.")
            return out

        docs = list(pb.find({"anio_lectivo": anio_lectivo}, proy))
        out = _dnis_grado5_desde_documentos(docs)
        if out:
            print("[INFO] Lista 5.°: primer_bimestre por año (sin exigir nivel; se excluye primaria explícita).")
            return out

        docs = list(db["asistencia"].find({"anio_lectivo": anio_lectivo}, proy))
        out = _dnis_grado5_desde_documentos(docs)
        if out:
            print("[INFO] Lista 5.°: asistencia por año y grado 5 (sin exigir nivel en Mongo).")
        return out
    finally:
        client.close()


def _mongo_filter_con_dnies(
    anio_lectivo: Optional[int],
    dni_allowlist: Optional[Set[str]],
    restringir_nivel_secundaria: bool,
) -> Dict[str, Any]:
    """Filtro find(): año opcional + opcional $or DNI + opcional nivel secundaria."""
    partes: List[Dict[str, Any]] = []
    if anio_lectivo is not None:
        partes.append({"anio_lectivo": anio_lectivo})
    if dni_allowlist is not None and len(dni_allowlist) > 0:
        ids = list(dni_allowlist)
        partes.append({"$or": [{k: {"$in": ids}} for k in DNI_MONGO_KEYS]})
    if restringir_nivel_secundaria:
        partes.append({"nivel_educativo": "secundaria"})
    if not partes:
        return {}
    if len(partes) == 1:
        return partes[0]
    return {"$and": partes}


def cargar_resumen_notas_historial(
    db: Any,
    dnis: Set[str],
    anio_lectivo: int,
    dni_allowlist: Optional[Set[str]],
) -> Dict[str, Dict[str, Any]]:
    """
    Lee primer a cuarto bimestre en los cuatro años anteriores a ``anio_lectivo``.
    Colección IV: ``cuarto_bimestre`` (mismo esquema que los demás; si no existe en la BD, se ignora).
    Por año lectivo cuenta como útil si hay al menos ``historial_min_bims_por_anio`` notas válidas
    (cualitativas convertidas); no usa 5 como relleno para faltantes.
    """
    cfg = MODEL_CONFIG
    if not dnis or not cfg.get("historial_habilitado", True):
        return {}
    min_bims = int(cfg.get("historial_min_bims_por_anio", 2))
    anios_previos = [anio_lectivo - k for k in (4, 3, 2, 1)]
    # dni -> anio -> [Bim1..Bim4] último doc por celda (orden _id)
    cubo: Dict[str, Dict[int, List[Optional[int]]]] = {}
    colecciones = (
        (1, "primer_bimestre"),
        (2, "segundo_bimestre"),
        (3, "tercer_bimestre"),
        (4, "cuarto_bimestre"),
    )

    for anio in anios_previos:
        allow = dni_allowlist if dni_allowlist is not None and len(dni_allowlist) > 0 else dnis
        if not allow:
            continue
        mf = _mongo_filter_con_dnies(anio, allow, restringir_nivel_secundaria=False)
        for numero_bim, nombre_col in colecciones:
            col = db[nombre_col]
            for doc in col.find(mf).sort("_id", 1):
                dni = normalizar_dni(doc)
                if not dni or dni not in dnis:
                    continue
                nota_numerica = convertir_calificacion(doc.get("PROMEDIO_APRENDIZAJE_AUTONOMO"))
                if dni not in cubo:
                    cubo[dni] = {}
                if anio not in cubo[dni]:
                    cubo[dni][anio] = [None, None, None, None]
                if nota_numerica is not None:
                    cubo[dni][anio][numero_bim - 1] = int(nota_numerica)

    out: Dict[str, Dict[str, Any]] = {}
    for dni in dnis:
        por_anio = cubo.get(dni) or {}
        medias_anuales: List[float] = []
        anios_usados: List[int] = []
        for anio in anios_previos:
            notas_anio = por_anio.get(anio)
            if not notas_anio:
                continue
            presentes = [x for x in notas_anio if x is not None]
            if len(presentes) >= min_bims:
                medias_anuales.append(sum(presentes) / len(presentes))
                anios_usados.append(anio)
        n_utiles = len(medias_anuales)
        if n_utiles == 0:
            out[dni] = {
                "hist_n_anios_utiles": 0,
                "hist_media_global": None,
                "hist_std_medias_anuales": 0.0,
                "hist_anios_muestra": "",
            }
            continue
        media_g = sum(medias_anuales) / n_utiles
        std_g = _std_muestral(medias_anuales)
        out[dni] = {
            "hist_n_anios_utiles": n_utiles,
            "hist_media_global": round(media_g, 4),
            "hist_std_medias_anuales": round(std_g, 4),
            "hist_anios_muestra": ",".join(str(a) for a in sorted(anios_usados)),
        }
    return out


def ejecutar_analisis_sate(
    mongodb_uri: str,
    database_name: str,
    anio_lectivo: Optional[int] = None,
    dni_allowlist: Optional[Set[str]] = None,
) -> Dict:
    """
    Función principal: Ejecuta el análisis SATE-SR completo.

    Si anio_lectivo está definido, solo se leen documentos con ese año (colecciones con campo anio_lectivo).
    Si es None, se mantiene el comportamiento anterior (todos los documentos).
    Si dni_allowlist está definido, solo entran esos DNI en el find por año;
    no se exige nivel_educativo en Mongo (muchos datos reales no lo traen en todas las colecciones).
    incidentes: solo filtro por año en modo lista DNI.
    """
    print('[INFO] Iniciando analisis SATE-SR v3.0 (Python)...')
    if dni_allowlist is not None and len(dni_allowlist) == 0:
        raise ValueError("dni_allowlist vacío: ningún DNI cumple la cohorte o el filtro.")
    cohorte = dni_allowlist is not None
    if cohorte:
        print(f'[INFO] Modo solo 5.° secundaria: {len(dni_allowlist)} DNI (lista desde ancla grado 5)')
    # No añadir nivel_educativo al find: nómina/asistencia reales suelen no tenerlo y quedarían vacías.
    mongo_filter = _mongo_filter_con_dnies(anio_lectivo, dni_allowlist, restringir_nivel_secundaria=False)
    if anio_lectivo is not None:
        mongo_filter_incidente = {"anio_lectivo": anio_lectivo}
    else:
        mongo_filter_incidente = {}
    if anio_lectivo is not None:
        print(f'[INFO] Filtro MongoDB: anio_lectivo = {anio_lectivo}' + (" + DNI allowlist" if cohorte else ""))
    
    # Conectar a MongoDB
    client = MongoClient(mongodb_uri)
    db = client[database_name]
    
    try:
        # ============================================
        # FASE ETL: EXTRACCIÓN Y TRANSFORMACIÓN
        # ============================================
        
        # 1. ASISTENCIAS
        print('[1/6] Procesando datos de Asistencia...')
        col_asistencias = db['asistencia']
        docs_asistencias = list(col_asistencias.find(mongo_filter).sort('_id', 1))
        
        asistencia_map = {}
        for doc in docs_asistencias:
            dni = normalizar_dni(doc)
            nombres = normalizar_nombres(doc)
            if not dni or not dni.strip():
                continue
            
            day_cols = _columnas_dias_asistencia(doc)
            asistencias = sum(1 for col in day_cols if _valor_dia_asistencia(doc.get(col)) == 1)
            faltas = sum(1 for col in day_cols if _valor_dia_asistencia(doc.get(col)) in (0, 2))
            
            key = f"{dni}_{nombres}"
            if key not in asistencia_map:
                asistencia_map[key] = {
                    'DNI': dni,
                    'Apellidos_Nombres': nombres,
                    'Seccion': doc.get('SECCIÓN') or doc.get('Seccion', ''),
                    'Grado': doc.get('GRADO') or doc.get('Grado', ''),
                    'cantidad_asistencias': 0,
                    'cantidad_faltas': 0
                }
            
            asistencia_map[key]['cantidad_asistencias'] += asistencias
            asistencia_map[key]['cantidad_faltas'] += faltas
        
        df_asistencias_final = []
        for reg in asistencia_map.values():
            total_dias = reg['cantidad_asistencias'] + reg['cantidad_faltas']
            porcentaje_faltas = (reg['cantidad_faltas'] / total_dias * 100) if total_dias > 0 else 0
            umbral_faltas = MODEL_CONFIG["umbral_faltas_critico"]
            
            df_asistencias_final.append({
                'DNI': reg['DNI'],
                'Apellidos_Nombres': reg['Apellidos_Nombres'],
                'Seccion': reg['Seccion'],
                'Grado': reg['Grado'],
                'Analisis_Asistencia': 0 if porcentaje_faltas >= umbral_faltas else 1
            })
        
        print(f'   [OK] Asistencias procesadas: {len(df_asistencias_final)} registros')
        
        # 2. NÓMINA (Situación Familiar)
        print('[2/6] Procesando datos de Nómina...')
        col_nomina = db['nomina']
        docs_nomina = list(col_nomina.find(mongo_filter).sort('_id', 1))
        
        df_nomina_final = []
        for doc in docs_nomina:
            dni = normalizar_dni(doc)
            nombres = normalizar_nombres(doc)
            if not dni:
                continue
            
            analisis_padre_vive = 1 if str(doc.get('padre_vive', '')).strip().upper() == 'SI' else -1
            analisis_madre_vive = 1 if str(doc.get('madre_vive', '')).strip().upper() == 'SI' else -1
            analisis_trabaja_estudiante = -1 if str(doc.get('trabaja_estudiante', '')).strip().upper() == 'SI' else 1
            analisis_tipo_discapacidad = 1 if not doc.get('tipo_discapacidad') or str(doc.get('tipo_discapacidad', '')).strip() == '' else -2
            
            situacion_mat = str(doc.get('situacion_matricula', '')).strip().upper()
            analisis_situacion_matricula = 0
            if situacion_mat == 'P':
                analisis_situacion_matricula = 1
            elif situacion_mat == 'PG':
                analisis_situacion_matricula = -1
            
            puntaje_total = (analisis_padre_vive + analisis_madre_vive + 
                           analisis_trabaja_estudiante + analisis_tipo_discapacidad + 
                           analisis_situacion_matricula)
            
            df_nomina_final.append({
                'DNI': dni,
                'Apellidos_Nombres': nombres,
                'Genero': doc.get('sexo', ''),
                'Analisis_Situacion_Familiar': 1 if puntaje_total >= 4 else 0
            })
        
        print(f'   [OK] Nomina procesada: {len(df_nomina_final)} registros')
        
        # 3, 4, 5. BIMESTRES
        def procesar_bimestre(numero_bim: int, nombre_coleccion: str) -> List[Dict]:
            print(f'[{numero_bim + 2}/6] Procesando Bimestre {numero_bim}...')
            col_bim = db[nombre_coleccion]
            docs_bim = list(col_bim.find(mongo_filter).sort('_id', 1))
            
            resultados = []
            for doc in docs_bim:
                dni = normalizar_dni(doc)
                nombres = normalizar_nombres(doc)
                if not dni or not dni.strip():
                    continue
                
                nota_numerica = convertir_calificacion(doc.get('PROMEDIO_APRENDIZAJE_AUTONOMO'))
                
                resultados.append({
                    'DNI': dni,
                    'Apellidos_Nombres': nombres,
                    f'NotaBim{numero_bim}': nota_numerica if nota_numerica else 5
                })
            
            return resultados
        
        df_bim1_final = procesar_bimestre(1, 'primer_bimestre')
        df_bim2_final = procesar_bimestre(2, 'segundo_bimestre')
        df_bim3_final = procesar_bimestre(3, 'tercer_bimestre')
        
        # 6. INCIDENTES
        print('[6/6] Procesando datos de Incidencias...')
        col_incidente = db['incidente']
        docs_incidente = list(col_incidente.find(mongo_filter_incidente))
        
        incidente_map: Dict[str, Dict[str, Any]] = {}
        for doc in docs_incidente:
            nombre = doc.get('Nombre y Apellido') or normalizar_nombres(doc)
            if not nombre:
                continue
            nombre_key = _norm_clave_nombre_completo(nombre)
            tipo_falta = str(doc.get('Tipo de Falta', '')).strip()
            es_leve = tipo_falta.lower() == 'leve'
            
            if nombre_key not in incidente_map:
                incidente_map[nombre_key] = {'Analisis_Incidencias': 1 if es_leve else 0}
            else:
                if not es_leve:
                    incidente_map[nombre_key]['Analisis_Incidencias'] = 0
        
        df_incidente_grouped = [
            {'Apellidos_Nombres': nombre, **datos}
            for nombre, datos in incidente_map.items()
        ]
        
        print(f'   [OK] Incidentes procesados: {len(df_incidente_grouped)} registros')
        
        # 7. ENCUESTA (Análisis de Sentimiento)
        print('[INFO] Analizando sentimientos de estudiantes...')
        col_encuesta = db['encuesta']
        docs_encuesta = list(col_encuesta.find(mongo_filter).sort('_id', 1))
        
        encuesta_map: Dict[str, Dict] = {}
        sin_dni_encuesta = 0
        sentimientos_positivos = 0
        sentimientos_negativos = 0
        textos_vacios = 0
        
        for doc in docs_encuesta:
            dni = normalizar_dni(doc)
            if not dni:
                sin_dni_encuesta += 1
                continue
            
            if dni not in encuesta_map:
                texto_sentimiento = extraer_texto_abierto_encuesta(doc)
                
                if not texto_sentimiento or str(texto_sentimiento).strip() == '':
                    textos_vacios += 1
                    sentimiento = 1
                else:
                    texto_str = str(texto_sentimiento).strip()
                    sentimiento = analizar_sentimiento_espanol(texto_str)
                
                if sentimiento == 1:
                    sentimientos_positivos += 1
                else:
                    sentimientos_negativos += 1
                
                encuesta_map[dni] = {
                    'DNI': dni,
                    'Analisis_Sentimiento_Estudiante': sentimiento
                }
        
        df_encuesta_final = list(encuesta_map.values())
        if sin_dni_encuesta:
            print(
                f'   [ADVERTENCIA] Encuesta: {sin_dni_encuesta} filas sin DNI '
                f'(no se enlazan a alumno). Regenera CSV con columna DNI o vuelve a importar encuesta.'
            )
        print(f'   [OK] Encuesta procesada: {len(docs_encuesta)} respuestas leídas, {len(encuesta_map)} con DNI enlazable')
        print(f'   [INFO] Sentimientos: {sentimientos_positivos} positivos, {sentimientos_negativos} negativos, {textos_vacios} vacíos')
        
        # ============================================
        # INTEGRACIÓN DE DATOS (Merge)
        # ============================================
        print('[INFO] Integrando datos de todas las fuentes...')
        
        estudiantes_map = {}
        
        # Agregar datos de nómina (base)
        for reg in df_nomina_final:
            estudiantes_map[reg['DNI']] = reg.copy()
        
        # Merge con asistencias
        for reg in df_asistencias_final:
            if not reg.get('DNI'):
                continue
            dni = reg['DNI']
            estudiante = estudiantes_map.get(dni, {'DNI': dni, 'Apellidos_Nombres': reg.get('Apellidos_Nombres', '')})
            estudiante.update({
                'Apellidos_Nombres': reg.get('Apellidos_Nombres') or estudiante.get('Apellidos_Nombres', ''),
                'Seccion': reg.get('Seccion') or estudiante.get('Seccion', ''),
                'Grado': reg.get('Grado') or estudiante.get('Grado', ''),
                'Analisis_Asistencia': reg.get('Analisis_Asistencia', 1)
            })
            estudiantes_map[dni] = estudiante
        
        # Merge con bimestres
        for idx, df_bim in enumerate([df_bim1_final, df_bim2_final, df_bim3_final], 1):
            for reg in df_bim:
                if not reg.get('DNI'):
                    continue
                dni = reg['DNI']
                estudiante = estudiantes_map.get(dni, {'DNI': dni, 'Apellidos_Nombres': reg.get('Apellidos_Nombres', '')})
                estudiante.update({
                    'Apellidos_Nombres': reg.get('Apellidos_Nombres') or estudiante.get('Apellidos_Nombres', ''),
                    f'NotaBim{idx}': reg.get(f'NotaBim{idx}', 5)
                })
                estudiantes_map[dni] = estudiante
        
        # Merge con incidentes (por nombre exacto o normalizado)
        nombre_to_dni: Dict[str, str] = {}
        for dni, est in estudiantes_map.items():
            n = (est.get('Apellidos_Nombres') or '').strip()
            if not n:
                continue
            nombre_to_dni[n] = dni
            nombre_to_dni[_norm_clave_nombre_completo(n)] = dni
        for reg in df_incidente_grouped:
            raw = (reg.get('Apellidos_Nombres') or '').strip()
            dni = nombre_to_dni.get(raw) or nombre_to_dni.get(_norm_clave_nombre_completo(raw))
            if dni:
                estudiante = estudiantes_map[dni]
                estudiante['Analisis_Incidencias'] = reg.get('Analisis_Incidencias', 1)
                estudiantes_map[dni] = estudiante
        
        # Merge con sentimientos
        # Crear un set de DNIs que tienen datos de encuesta
        dnis_con_encuesta = set()
        
        for reg in df_encuesta_final:
            dni = reg.get('DNI')
            if dni and dni in estudiantes_map:
                estudiante = estudiantes_map[dni]
                estudiante['Analisis_Sentimiento_Estudiante'] = reg.get('Analisis_Sentimiento_Estudiante', 1)
                estudiantes_map[dni] = estudiante
                dnis_con_encuesta.add(dni)
        
        # Para estudiantes sin datos de encuesta, marcar como desconocido (neutral)
        # Usaremos 1 (sin riesgo) solo si realmente tienen datos positivos
        # Si no tienen datos, deberíamos marcarlos de manera diferente o usar un valor neutral
        # Por ahora, mantenemos 1 pero agregamos un log para debugging
        estudiantes_sin_encuesta = len(estudiantes_map) - len(dnis_con_encuesta)
        if estudiantes_sin_encuesta > 0:
            print(f'   [ADVERTENCIA] {estudiantes_sin_encuesta} estudiantes sin datos de encuesta (marcados como sin riesgo por defecto)')
        
        # Convertir a lista y aplicar valores por defecto
        df_final = []
        for dni, est in estudiantes_map.items():
            if not dni or not str(dni).strip():
                continue
            dni_s = str(dni).strip()
            df_final.append({
                'DNI': dni_s,
                'Apellidos_Nombres': est.get('Apellidos_Nombres', ''),
                'Genero': est.get('Genero', ''),
                'Seccion': est.get('Seccion', ''),
                'Grado': est.get('Grado', ''),
                'NotaBim1': est.get('NotaBim1', 5),
                'NotaBim2': est.get('NotaBim2', 5),
                'NotaBim3': est.get('NotaBim3', 5),
                'Analisis_Asistencia': est.get('Analisis_Asistencia', 1),
                'Analisis_Incidencias': est.get('Analisis_Incidencias', 1),
                'Analisis_Sentimiento_Estudiante': est.get('Analisis_Sentimiento_Estudiante', 1),
                'Analisis_Situacion_Familiar': est.get('Analisis_Situacion_Familiar', 1),
            })
        
        # Eliminar duplicados por DNI
        dni_set = set()
        df_final = [est for est in df_final if est['DNI'] not in dni_set and not dni_set.add(est['DNI'])]
        
        if dni_allowlist is not None:
            antes = len(df_final)
            allow = {str(x).strip() for x in dni_allowlist}
            df_final = [est for est in df_final if str(est.get("DNI", "")).strip() in allow]
            if antes != len(df_final):
                print(f'   [INFO] Post-filtro por DNI permitidos: {antes} -> {len(df_final)} estudiantes')
        
        print(f'[OK] Tabla integrada: {len(df_final)} estudiantes unicos')
        
        if len(df_final) == 0:
            raise ValueError('No se encontraron estudiantes para analizar.')
        
        hist_map: Dict[str, Dict[str, Any]] = {}
        if anio_lectivo is not None and MODEL_CONFIG.get("historial_habilitado", True):
            dnis_f: Set[str] = {str(e["DNI"]).strip() for e in df_final if str(e.get("DNI", "")).strip()}
            if dnis_f:
                hist_map = cargar_resumen_notas_historial(db, dnis_f, anio_lectivo, dni_allowlist)
                min_u = int(MODEL_CONFIG.get("historial_min_anios_utiles", 2))
                n_ok = sum(
                    1 for v in hist_map.values()
                    if int(v.get("hist_n_anios_utiles", 0) or 0) >= min_u
                )
                a_ini, a_fin = anio_lectivo - 4, anio_lectivo - 1
                min_b = int(MODEL_CONFIG.get("historial_min_bims_por_anio", 2))
                print(
                    f'   [INFO] Historial notas ({a_ini}..{a_fin}): {n_ok}/{len(dnis_f)} DNIs '
                    f'con >= {min_u} años útiles (media anual con >= {min_b} bimestres con nota)'
                )
        for est in df_final:
            hrow = hist_map.get(str(est["DNI"]).strip(), {})
            est["Hist_N_Anios_Utiles"] = int(hrow.get("hist_n_anios_utiles", 0) or 0)
            est["Hist_Media_Global"] = hrow.get("hist_media_global")
            est["Hist_Std_Medias_Anual"] = hrow.get("hist_std_medias_anuales", 0.0)
            est["Hist_Anios_Muestra"] = hrow.get("hist_anios_muestra", "") or ""
        
        # ============================================
        # MODELO PREDICTIVO
        # ============================================
        print('[INFO] Ejecutando predicciones...')
        
        for est in df_final:
            dni_k = str(est.get('DNI', '')).strip()
            hraw = hist_map.get(dni_k)
            min_u = int(MODEL_CONFIG.get("historial_min_anios_utiles", 2))
            h_use = None
            if (
                hraw
                and hraw.get("hist_media_global") is not None
                and int(hraw.get("hist_n_anios_utiles", 0) or 0) >= min_u
            ):
                h_use = hraw
            est['Nota_Nucleo_B4'] = round(proyectar_nota_nucleo_b4(est), 4)
            est['Nota_Proyectada_B4'] = proyectar_nota_robusta(est, resumen_historial=h_use)
            est['Prediccion_Final_Binaria'] = clasificar_resultado(est['Nota_Proyectada_B4'])
            est['Estado'] = '[OK] APRUEBA' if est['Prediccion_Final_Binaria'] == 1 else '[X] DESAPRUEBA'
        
        print('[OK] Predicciones completadas')
        
        # ============================================
        # VALIDACIÓN DEL MODELO (Temporal - más realista)
        # ============================================
        print('[INFO] Validando modelo con validación temporal...')
        print('[INFO] Usando Bim1 y Bim2 para predecir Bim3, y validando con Bim3 real')
        
        # Validación temporal: usar Bim1 y Bim2 para predecir Bim3
        # Esto es más realista porque simula predecir el futuro
        y_true_temporal = []
        y_pred_temporal = []
        y_scores_temporal = []  # Scores continuos para calcular AUC-ROC con mayor precisión
        
        for est in df_final:
            # Solo validar estudiantes que tienen al menos Bim1 y Bim2
            if est.get('NotaBim1') and est.get('NotaBim2') and est.get('NotaBim3'):
                # Realidad: clasificar Bim3 real
                realidad_bim3 = clasificar_resultado(est['NotaBim3'])
                
                # Predicción: usar solo Bim1 y Bim2 para predecir Bim3
                # Simular proyección usando solo los primeros dos bimestres
                notas_para_validacion = [est.get('NotaBim1', 5), est.get('NotaBim2', 5)]
                nota_min, nota_max = MODEL_CONFIG["nota_escala"]
                notas_validadas = [max(nota_min, min(nota_max, n)) for n in notas_para_validacion]
                
                # Regresión lineal simple con solo 2 puntos
                if len(notas_validadas) == 2:
                    # Proyección simple: continuar la tendencia
                    cambio = notas_validadas[1] - notas_validadas[0]
                    proyeccion_bim3 = notas_validadas[1] + cambio
                    
                    # Aplicar límite de cambio máximo
                    max_cambio = MODEL_CONFIG["max_proyeccion_cambio"]
                    proyeccion_bim3 = max(
                        notas_validadas[1] - max_cambio,
                        min(notas_validadas[1] + max_cambio, proyeccion_bim3)
                    )
                    
                    # Aplicar penalización por factores de riesgo (igual que en el modelo real)
                    pesos = MODEL_CONFIG["pesos_penalizacion"]
                    castigo = (
                        (1 - est.get('Analisis_Asistencia', 1)) * pesos["asistencia"] +
                        (1 - est.get('Analisis_Incidencias', 1)) * pesos["incidencias"] +
                        (1 - est.get('Analisis_Sentimiento_Estudiante', 1)) * pesos["sentimiento"] +
                        (1 - est.get('Analisis_Situacion_Familiar', 1)) * pesos["familia"]
                    )
                    
                    nota_final_validacion = max(nota_min, min(nota_max, proyeccion_bim3 - castigo))
                    prediccion_bim3 = clasificar_resultado(nota_final_validacion)
                    
                    y_true_temporal.append(realidad_bim3)
                    y_pred_temporal.append(prediccion_bim3)
                    y_scores_temporal.append(nota_final_validacion)  # Guardar score continuo para AUC-ROC
        
        # Calcular métricas con validación temporal usando scores continuos (MEJORA SIGNIFICATIVA)
        if len(y_true_temporal) > 0:
            print(f'[INFO] Validación temporal: {len(y_true_temporal)} estudiantes con datos completos')
            print(f'[INFO] Calculando AUC-ROC con scores continuos (notas proyectadas) para mayor precisión...')
            metricas = calcular_metricas(y_true_temporal, y_pred_temporal, y_scores_temporal)
            
            # Log de métricas temporales para debugging
            logger.info(f'VALIDACION TEMPORAL - AUC-ROC: {metricas["auc_roc"]:.4f} (usando scores continuos)')
            logger.info(f'VALIDACION TEMPORAL - Precision: {metricas["precision"]:.4f}, Recall: {metricas["recall"]:.4f}')
            print(f'[OK] AUC-ROC mejorado usando scores continuos: {metricas["auc_roc"]:.4f}')
        else:
            print('[ADVERTENCIA] No hay suficientes datos para validación temporal, usando validación estándar')
            # Fallback: validación estándar usando notas proyectadas como scores
            y_true = []
            y_pred = []
            y_scores = []
            for est in df_final:
                realidad_bim3 = clasificar_resultado(est.get('NotaBim3', 5))
                y_true.append(realidad_bim3)
                y_pred.append(est['Prediccion_Final_Binaria'])
                y_scores.append(est['Nota_Proyectada_B4'])  # Usar nota proyectada como score
            
            print('[INFO] Calculando AUC-ROC con scores continuos (notas proyectadas) para mayor precisión...')
            metricas = calcular_metricas(y_true, y_pred, y_scores)
            logger.info(f'VALIDACION ESTANDAR - AUC-ROC: {metricas["auc_roc"]:.4f} (usando scores continuos)')
        
        # ============================================
        # PREPARAR RESULTADOS FINALES
        # ============================================
        
        # Ordenar por Sección y Apellidos
        df_final.sort(key=lambda x: (x.get('Seccion', ''), x.get('Apellidos_Nombres', '')))
        
        total_estudiantes = len(df_final)
        aprueba_count = sum(1 for e in df_final if e['Prediccion_Final_Binaria'] == 1)
        desaprueba_count = total_estudiantes - aprueba_count
        promedio_nota_proyectada = sum(e['Nota_Proyectada_B4'] for e in df_final) / total_estudiantes
        min_hist_u = int(MODEL_CONFIG.get("historial_min_anios_utiles", 2))
        n_hist_ok = sum(
            1 for e in df_final if int(e.get("Hist_N_Anios_Utiles", 0) or 0) >= min_hist_u
        )
        pct_hist = (n_hist_ok / total_estudiantes * 100.0) if total_estudiantes else 0.0
        
        factores_riesgo = {
            'asistencia': {
                'sin_riesgo': sum(1 for e in df_final if e['Analisis_Asistencia'] == 1),
                'con_riesgo': sum(1 for e in df_final if e['Analisis_Asistencia'] == 0),
            },
            'incidencias': {
                'sin_riesgo': sum(1 for e in df_final if e['Analisis_Incidencias'] == 1),
                'con_riesgo': sum(1 for e in df_final if e['Analisis_Incidencias'] == 0),
            },
            'sentimiento': {
                'sin_riesgo': sum(1 for e in df_final if e.get('Analisis_Sentimiento_Estudiante', 1) == 1),
                'con_riesgo': sum(1 for e in df_final if e.get('Analisis_Sentimiento_Estudiante', 1) == 0),
            },
            'situacion_familiar': {
                'sin_riesgo': sum(1 for e in df_final if e['Analisis_Situacion_Familiar'] == 1),
                'con_riesgo': sum(1 for e in df_final if e['Analisis_Situacion_Familiar'] == 0),
            },
        }
        
        resultado = {
            'success': True,
            'version': MODEL_CONFIG['version'],
            'fecha_analisis': datetime.now().isoformat(),
            'anio_lectivo': anio_lectivo,
            'database_name': database_name,
            'total_estudiantes': total_estudiantes,
            'solo_quintos_secundaria': cohorte,
            'metricas': {
                'aprueba': aprueba_count,
                'desaprueba': desaprueba_count,
                'porcentaje_aprueba': (aprueba_count / total_estudiantes) * 100,
                'porcentaje_desaprueba': (desaprueba_count / total_estudiantes) * 100,
                'promedio_nota_proyectada': promedio_nota_proyectada,
                'historial_dnis_utiles': n_hist_ok,
                'historial_pct_cobertura': round(pct_hist, 2),
                **metricas
            },
            'factores_riesgo': factores_riesgo,
            'resultados': [
                {
                    'DNI': est['DNI'],
                    'Apellidos_Nombres': est['Apellidos_Nombres'],
                    'Genero': est['Genero'],
                    'Seccion': est['Seccion'],
                    'Grado': est['Grado'],
                    'NotaBim1': est['NotaBim1'],
                    'NotaBim2': est['NotaBim2'],
                    'NotaBim3': est['NotaBim3'],
                    'Analisis_Asistencia': est['Analisis_Asistencia'],
                    'Analisis_Incidencias': est['Analisis_Incidencias'],
                    'Analisis_Sentimiento_Estudiante': est['Analisis_Sentimiento_Estudiante'],
                    'Analisis_Situacion_Familiar': est['Analisis_Situacion_Familiar'],
                    'Hist_N_Anios_Utiles': est.get('Hist_N_Anios_Utiles', 0),
                    'Hist_Media_Global': est.get('Hist_Media_Global'),
                    'Hist_Std_Medias_Anual': est.get('Hist_Std_Medias_Anual'),
                    'Hist_Anios_Muestra': est.get('Hist_Anios_Muestra', ''),
                    'Nota_Nucleo_B4': round(float(est['Nota_Nucleo_B4']), 2),
                    'Nota_Proyectada_B4': round(est['Nota_Proyectada_B4'], 2),
                    'Prediccion_Final_Binaria': est['Prediccion_Final_Binaria'],
                    'Estado': est['Estado']
                }
                for est in df_final
            ]
        }
        if cohorte and dni_allowlist is not None:
            resultado["dnis_quintos_secundaria"] = len(dni_allowlist)
        
        print('[OK] Analisis SATE-SR completado exitosamente')
        return resultado
        
    finally:
        client.close()



# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# REPORTING (export ML, Power BI, gráficos base)
# ---------------------------------------------------------------------------

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _grado_a_numero(grado: Any) -> float:
    if grado is None or str(grado).strip() == "":
        return float("nan")
    m = re.match(r"^\s*(\d+)", str(grado).strip())
    return float(m.group(1)) if m else float("nan")


def resultado_a_dataframe_ml(resultado: dict[str, Any]) -> pd.DataFrame:
    """Tabla lista para entrenar modelos: features numéricas + objetivos."""
    rows = resultado.get("resultados") or []
    if not rows:
        return pd.DataFrame()

    anio = resultado.get("anio_lectivo")

    records = []
    seen: set[str] = set()
    for r in rows:
        dni = str(r.get("DNI", "")).strip()
        if dni and dni in seen:
            continue
        if dni:
            seen.add(dni)

        records.append(
            {
                "anio_lectivo": anio,
                "dni": dni,
                "genero": r.get("Genero", ""),
                "grado_txt": r.get("Grado", ""),
                "seccion": str(r.get("Seccion", "")).strip(),
                "grado_num": _grado_a_numero(r.get("Grado")),
                "nota_bim1": r.get("NotaBim1"),
                "nota_bim2": r.get("NotaBim2"),
                "nota_bim3": r.get("NotaBim3"),
                "analisis_asistencia": r.get("Analisis_Asistencia"),
                "analisis_incidencias": r.get("Analisis_Incidencias"),
                "analisis_sentimiento": r.get("Analisis_Sentimiento_Estudiante"),
                "analisis_situacion_familiar": r.get("Analisis_Situacion_Familiar"),
                "hist_n_anios_utiles": r.get("Hist_N_Anios_Utiles"),
                "hist_media_global": r.get("Hist_Media_Global"),
                "hist_std_medias_anual": r.get("Hist_Std_Medias_Anual"),
                "hist_anios_muestra": r.get("Hist_Anios_Muestra"),
                "nota_nucleo_b4": r.get("Nota_Nucleo_B4"),
                "nota_proyectada_b4": r.get("Nota_Proyectada_B4"),
                "target_aprueba_binario": r.get("Prediccion_Final_Binaria"),
            }
        )

    return pd.DataFrame.from_records(records)


RISK_FEATURE_COLS = [
    "analisis_asistencia",
    "analisis_incidencias",
    "analisis_sentimiento",
    "analisis_situacion_familiar",
]


def enriquecer_para_powerbi(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas legibles y métricas derivadas para Power BI / Excel."""
    out = df.copy()
    if "target_aprueba_binario" in out.columns:
        out["estado_prediccion_txt"] = out["target_aprueba_binario"].map(
            {0: "Desaprueba", 1: "Aprueba"}
        )
    rc = [c for c in RISK_FEATURE_COLS if c in out.columns]
    if rc:
        out["num_factores_en_riesgo"] = (out[rc] == 0).sum(axis=1).astype(int)
    if "grado_txt" in out.columns and "seccion" in out.columns:
        out["clave_grado_seccion"] = (
            out["grado_txt"].astype(str).str.strip()
            + "-"
            + out["seccion"].astype(str).str.strip()
        )
    return out


def export_ml_csv(df: pd.DataFrame, path: Path, sep: str = ";") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=sep, index=False, encoding="utf-8")
    return path.resolve()


def export_para_powerbi(df: pd.DataFrame, path: Path, sep: str = ";") -> Path:
    """CSV con BOM UTF-8 para Excel/Power BI en Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    enriquecer_para_powerbi(df).to_csv(path, sep=sep, index=False, encoding="utf-8-sig")
    return path.resolve()


def _kpis_y_resumen_factores_planos(resultado: dict[str, Any]) -> dict[str, Any]:
    """KPI de validación + conteos de factores (antes en metricas_resumen / factores_riesgo) como columnas fijas."""
    m = resultado.get("metricas") or {}
    fr = resultado.get("factores_riesgo") or {}
    mc = m.get("matriz_confusion") if isinstance(m.get("matriz_confusion"), dict) else {}

    def fr_pair(key: str) -> tuple[Any, Any]:
        b = fr.get(key)
        if not isinstance(b, dict):
            return None, None
        return b.get("sin_riesgo"), b.get("con_riesgo")

    a_sr, a_cr = fr_pair("asistencia")
    i_sr, i_cr = fr_pair("incidencias")
    s_sr, s_cr = fr_pair("sentimiento")
    f_sr, f_cr = fr_pair("situacion_familiar")

    return {
        "ejecucion_fecha": resultado.get("fecha_analisis"),
        "ejecucion_anio_lectivo": resultado.get("anio_lectivo"),
        "ejecucion_database": resultado.get("database_name"),
        "version_modelo": resultado.get("version"),
        "total_estudiantes": resultado.get("total_estudiantes"),
        "prediccion_aprueba_n": m.get("aprueba"),
        "prediccion_desaprueba_n": m.get("desaprueba"),
        "prediccion_pct_aprueba": round(float(m.get("porcentaje_aprueba") or 0), 4),
        "prediccion_pct_desaprueba": round(float(m.get("porcentaje_desaprueba") or 0), 4),
        "prediccion_promedio_nota_b4": m.get("promedio_nota_proyectada"),
        "validacion_precision": m.get("precision"),
        "validacion_recall": m.get("recall"),
        "validacion_f1": m.get("f1_score"),
        "validacion_auc_roc": m.get("auc_roc"),
        "validacion_matriz_tp": mc.get("verdaderos_positivos"),
        "validacion_matriz_fp": mc.get("falsos_positivos"),
        "validacion_matriz_fn": mc.get("falsos_negativos"),
        "validacion_matriz_tn": mc.get("verdaderos_negativos"),
        "resumen_factor_asistencia_sin_riesgo_n": a_sr,
        "resumen_factor_asistencia_con_riesgo_n": a_cr,
        "resumen_factor_conducta_sin_riesgo_n": i_sr,
        "resumen_factor_conducta_con_riesgo_n": i_cr,
        "resumen_factor_sentimiento_sin_riesgo_n": s_sr,
        "resumen_factor_sentimiento_con_riesgo_n": s_cr,
        "resumen_factor_familia_sin_riesgo_n": f_sr,
        "resumen_factor_familia_con_riesgo_n": f_cr,
    }


def resultado_a_dataframe_dashboard_completo(resultado: dict[str, Any]) -> pd.DataFrame:
    """
    Tabla única para Power BI: una fila por estudiante (plantilla tipo informe) y columnas de resumen
    (métricas globales + conteos por factor) repetidas en cada fila para tarjetas sin modelo estrella.
    """
    rows_src = resultado.get("resultados") or []
    if not rows_src:
        return pd.DataFrame()
    extras = _kpis_y_resumen_factores_planos(resultado)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in rows_src:
        dni = str(r.get("DNI", "")).strip()
        if dni and dni in seen:
            continue
        if dni:
            seen.add(dni)
        pred = r.get("Prediccion_Final_Binaria")
        estado = "APRUEBA" if pred == 1 else "DESAPRUEBA"
        fam_raw = r.get("Analisis_Situacion_Familiar")
        try:
            fam = float(fam_raw) if fam_raw is not None else float("nan")
        except (TypeError, ValueError):
            fam = float("nan")
        row = {
            "DNI": dni,
            "Apellidos y Nombres": str(r.get("Apellidos_Nombres", "")).strip().upper(),
            "Género": str(r.get("Genero", "")).strip(),
            "Sección": str(r.get("Seccion", "")).strip(),
            "Grado": r.get("Grado", ""),
            "Bim 1": r.get("NotaBim1"),
            "Bim 2": r.get("NotaBim2"),
            "Bim 3": r.get("NotaBim3"),
            "Asistencia": r.get("Analisis_Asistencia"),
            "Conducta": r.get("Analisis_Incidencias"),
            "Sentimiento": r.get("Analisis_Sentimiento_Estudiante"),
            "Fam. Estable": fam,
            "Nota núcleo B4": r.get("Nota_Nucleo_B4"),
            "Hist años útiles": r.get("Hist_N_Anios_Utiles"),
            "Hist media global": r.get("Hist_Media_Global"),
            "Hist años muestra": r.get("Hist_Anios_Muestra"),
            "Nota Proy. B4": r.get("Nota_Proyectada_B4"),
            "Estado": estado,
        }
        row.update(extras)
        records.append(row)
    return pd.DataFrame.from_records(records)


def export_reporte_dashboard_completo_csv(resultado: dict[str, Any], path: Path, sep: str = ";") -> Path:
    """Un solo CSV (UTF-8-BOM, `;`) con detalle por alumno y bloque de resumen predictivo para Power BI."""
    path.parent.mkdir(parents=True, exist_ok=True)
    resultado_a_dataframe_dashboard_completo(resultado).to_csv(
        path, sep=sep, index=False, encoding="utf-8-sig"
    )
    return path.resolve()


def generar_graficos(resultado: dict[str, Any], directorio: Path) -> list[Path]:
    """Genera PNG en directorio. Devuelve rutas creadas."""
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    guardados: list[Path] = []

    df = resultado_a_dataframe_ml(resultado)
    if df.empty:
        return guardados

    # 1) Distribución objetivo clasificación
    fig, ax = plt.subplots(figsize=(7, 4))
    vc = df["target_aprueba_binario"].value_counts().sort_index()
    labels = ["Desaprueba (0)", "Aprueba (1)"]
    ax.bar(labels, [vc.get(0, 0), vc.get(1, 0)], color=["#c0392b", "#27ae60"])
    ax.set_title("Predicción binaria SATE-SR (aprueba / desaprueba)")
    ax.set_ylabel("Estudiantes")
    p1 = directorio / "01_distribucion_prediccion.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    guardados.append(p1)

    # 2) Histograma nota proyectada coloreado por clase
    fig, ax = plt.subplots(figsize=(8, 4))
    for val, color, nombre in [(0, "#c0392b", "Desaprueba"), (1, "#27ae60", "Aprueba")]:
        sub = df[df["target_aprueba_binario"] == val]["nota_proyectada_b4"].dropna()
        if len(sub):
            sns.histplot(sub, kde=True, ax=ax, color=color, label=nombre, alpha=0.45)
    ax.set_title("Distribución de nota proyectada IV bimestre por clase")
    ax.set_xlabel("Nota proyectada")
    ax.legend()
    p2 = directorio / "02_histograma_nota_proyectada.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    guardados.append(p2)

    # 3) Heatmap correlación features numéricas
    cols_num = [
        "nota_bim1",
        "nota_bim2",
        "nota_bim3",
        "analisis_asistencia",
        "analisis_incidencias",
        "analisis_sentimiento",
        "analisis_situacion_familiar",
        "nota_proyectada_b4",
        "target_aprueba_binario",
    ]
    sub = df[[c for c in cols_num if c in df.columns]]
    if sub.shape[1] >= 2:
        corr = sub.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
        ax.set_title("Correlación entre variables numéricas")
        p3 = directorio / "03_correlacion_heatmap.png"
        fig.tight_layout()
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        guardados.append(p3)

    # 4) Factores de riesgo (barras apiladas simuladas desde conteos)
    fr = resultado.get("factores_riesgo") or {}
    if fr:
        factores = []
        sin_r = []
        con_r = []
        for nombre in ("asistencia", "incidencias", "sentimiento", "situacion_familiar"):
            bloque = fr.get(nombre) or {}
            factores.append(nombre)
            sin_r.append(bloque.get("sin_riesgo", 0))
            con_r.append(bloque.get("con_riesgo", 0))
        fig, ax = plt.subplots(figsize=(8, 4))
        x = range(len(factores))
        ax.bar(x, sin_r, label="Sin riesgo (factor)", color="#2ecc71")
        ax.bar(x, con_r, bottom=sin_r, label="Con riesgo", color="#e74c3c")
        ax.set_xticks(list(x))
        ax.set_xticklabels(factores, rotation=15, ha="right")
        ax.set_title("Factores de riesgo (conteos)")
        ax.legend()
        ax.set_ylabel("Estudiantes")
        p4 = directorio / "04_factores_riesgo_apilados.png"
        fig.tight_layout()
        fig.savefig(p4, dpi=150)
        plt.close(fig)
        guardados.append(p4)

    # 5) Matriz de confusión (métricas validación temporal)
    mc = (resultado.get("metricas") or {}).get("matriz_confusion")
    if isinstance(mc, dict) and mc:
        tp = mc.get("verdaderos_positivos", 0)
        fp = mc.get("falsos_positivos", 0)
        fn = mc.get("falsos_negativos", 0)
        tn = mc.get("verdaderos_negativos", 0)
        # Filas: real aprueba / desaprueba; columnas: pred aprueba / desaprueba
        mat = [[tp, fn], [fp, tn]]
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            mat,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Pred. aprueba", "Pred. desaprueba"],
            yticklabels=["Real aprueba", "Real desaprueba"],
        )
        ax.set_title("Matriz de confusión (validación temporal)")
        p5 = directorio / "05_matriz_confusion.png"
        fig.tight_layout()
        fig.savefig(p5, dpi=150)
        plt.close(fig)
        guardados.append(p5)

    return guardados


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PIPELINE INTEGRADOR (SQLite, JSON Mongo, ML extendido)
# ---------------------------------------------------------------------------

import gc
import json
import logging
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bson import ObjectId
from pymongo import MongoClient
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


class MongoJSONEncoder(json.JSONEncoder):
    """Serializa ObjectId (y fechas) para exportar colecciones a JSON."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def export_encuesta_json(
    uri: str,
    db_name: str,
    anio_lectivo: int,
    path_out: Path,
    limit: int = 5000,
    dni_allowlist: Optional[Set[str]] = None,
) -> Path:
    """MongoDB → JSON (solo lectura)."""
    path_out = Path(path_out)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    # No exigir nivel_educativo en el find (muchas encuestas no lo tienen).
    filtro = _mongo_filter_con_dnies(anio_lectivo, dni_allowlist, False)
    with MongoClient(uri) as client:
        coll = client[db_name]["encuesta"]
        docs = list(coll.find(filtro, {"_id": 0}).limit(limit))
    with path_out.open("w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2, cls=MongoJSONEncoder)
    logger.info("Export JSON encuesta: %s (%s registros)", path_out, len(docs))
    return path_out.resolve()


def agregacion_asistencia_por_grado(
    uri: str,
    db_name: str,
    anio_lectivo: int,
    dni_allowlist: Optional[Set[str]] = None,
) -> list[dict[str, Any]]:
    """Pipeline $match + $group (solo lectura), alineado a GRADO/Grado y SECCIÓN/Seccion del import."""
    match = _mongo_filter_con_dnies(anio_lectivo, dni_allowlist, False)
    with MongoClient(uri) as client:
        coll = client[db_name]["asistencia"]
        pipeline = [
            {"$match": match},
            {
                "$addFields": {
                    "_grado": {"$ifNull": ["$Grado", {"$ifNull": ["$GRADO", None]}]},
                    "_secc": {
                        "$ifNull": [
                            "$Seccion",
                            {"$ifNull": ["$SECCIÓN", {"$ifNull": ["$SECCION", None]}]},
                        ]
                    },
                }
            },
            {
                "$group": {
                    "_id": "$_grado",
                    "total_estudiantes": {"$sum": 1},
                    "secciones": {"$addToSet": "$_secc"},
                }
            },
            {"$sort": {"total_estudiantes": -1}},
        ]
        return list(coll.aggregate(pipeline))


def enriquecer_nombres_desde_resultado(
    resultado: dict[str, Any], df_ml: pd.DataFrame
) -> pd.DataFrame:
    """Añade `nombre_completo` para SQLite a partir de filas del motor SATE."""
    m: dict[str, str] = {}
    for r in resultado.get("resultados") or []:
        d = str(r.get("DNI", "")).strip()
        if not d:
            continue
        nom = str(r.get("Apellidos_Nombres", "")).strip()
        m[d] = nom or d
    out = df_ml.copy()
    out["nombre_completo"] = out["dni"].astype(str).str.strip().map(lambda d: m.get(d, d))
    return out


def sqlite_export_desde_df_ml(df: pd.DataFrame, db_path: Path) -> Path:
    """
    DDL + DML en SQLite (sqlite3): secciones, estudiantes, predicciones.
    Columnas esperadas: snake_case del `resultado_a_dataframe_ml`.
    """
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()

    def _f(x: Any, default: float = 0.0) -> float:
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return default
            return float(x)
        except (TypeError, ValueError):
            return default

    def _i(x: Any, default: int = 1) -> int:
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return default
            return int(float(x))
        except (TypeError, ValueError):
            return default

    df = df.copy()
    if df.empty:
        raise ValueError("DataFrame ML vacío: no se puede crear SQLite.")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE secciones (
            id_seccion   INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre       TEXT NOT NULL,
            grado        TEXT NOT NULL,
            UNIQUE(nombre, grado)
        );
        CREATE TABLE estudiantes (
            DNI          TEXT PRIMARY KEY,
            nombre       TEXT NOT NULL,
            genero       TEXT,
            id_seccion   INTEGER,
            FOREIGN KEY (id_seccion) REFERENCES secciones(id_seccion)
        );
        CREATE TABLE predicciones (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            DNI          TEXT NOT NULL,
            nota_bim1    REAL,
            nota_bim2    REAL,
            nota_bim3    REAL,
            nota_proy    REAL,
            estado       TEXT,
            fecha_pred   TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (DNI) REFERENCES estudiantes(DNI)
        );
        """
    )

    secciones = (
        df[["seccion", "grado_txt"]]
        .drop_duplicates()
        .rename(columns={"seccion": "Seccion", "grado_txt": "Grado"})
    )
    for _, row in secciones.iterrows():
        cur.execute(
            "INSERT OR IGNORE INTO secciones (nombre, grado) VALUES (?, ?)",
            (str(row["Seccion"]).strip(), str(row["Grado"]).strip()),
        )

    col_nombre = "nombre_completo" if "nombre_completo" in df.columns else "dni"

    for _, row in df.iterrows():
        dni = str(row.get("dni", "")).strip()
        if not dni:
            continue
        nombre = str(row.get(col_nombre, dni)).strip() or dni
        cur.execute(
            "SELECT id_seccion FROM secciones WHERE nombre = ? AND grado = ?",
            (str(row.get("seccion", "")).strip(), str(row.get("grado_txt", "")).strip()),
        )
        r = cur.fetchone()
        id_sec = r[0] if r else None
        gen = str(row.get("genero", "") or "M").strip()[:1] or "M"
        cur.execute(
            "INSERT OR REPLACE INTO estudiantes (DNI, nombre, genero, id_seccion) VALUES (?, ?, ?, ?)",
            (dni, nombre, gen, id_sec),
        )
        estado_txt = "Aprueba" if _i(row.get("target_aprueba_binario"), 1) == 1 else "Desaprueba"
        cur.execute(
            """INSERT INTO predicciones (DNI, nota_bim1, nota_bim2, nota_bim3, nota_proy, estado)
               VALUES (?,?,?,?,?,?)""",
            (
                dni,
                _f(row.get("nota_bim1"), 5),
                _f(row.get("nota_bim2"), 5),
                _f(row.get("nota_bim3"), 5),
                _f(row.get("nota_proyectada_b4"), 5),
                estado_txt,
            ),
        )

    cur.execute(
        "UPDATE predicciones SET estado = 'Desaprueba' WHERE nota_proy < 12 AND estado = 'Aprueba'"
    )
    cur.execute("DELETE FROM predicciones WHERE nota_proy IS NULL")
    conn.commit()
    conn.close()
    logger.info("SQLite escrito: %s", db_path)
    return db_path.resolve()


def _normalizar_genero(s: Any) -> str:
    if pd.isna(s):
        return "M"
    v = str(s).strip().upper()
    if v in ("F", "FEMENINO", "MUJER", "FEMALE"):
        return "F"
    return "M"


def sqlalchemy_orm_opcional(df: pd.DataFrame, db_path: Path) -> Path | None:
    """ORM opcional (si sqlalchemy no está instalado, se omite)."""
    try:
        from sqlalchemy import Column, Float, ForeignKey, Integer, String, create_engine
        from sqlalchemy.orm import Session, declarative_base, relationship

        Base = declarative_base()
    except ImportError:
        logger.warning("sqlalchemy no instalado; omite ORM (py -m pip install sqlalchemy).")
        return None

    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()

    class SeccionORM(Base):
        __tablename__ = "secciones_orm"
        id = Column(Integer, primary_key=True, autoincrement=True)
        nombre = Column(String(32), nullable=False)
        grado = Column(String(64), nullable=False)
        alumnos = relationship("EstudianteORM", back_populates="seccion")

    class EstudianteORM(Base):
        __tablename__ = "estudiantes_orm"
        DNI = Column(String(32), primary_key=True)
        nombre = Column(String(512), nullable=False)
        nota_proy = Column(Float)
        estado = Column(String(64))
        id_seccion = Column(Integer, ForeignKey("secciones_orm.id"))
        seccion = relationship("SeccionORM", back_populates="alumnos")

    engine = create_engine(f"sqlite:///{db_path.as_posix()}", echo=False)
    Base.metadata.create_all(engine)

    secciones_df = df[["seccion", "grado_txt"]].drop_duplicates()
    with Session(engine) as session:
        for _, row in secciones_df.iterrows():
            s = (
                session.query(SeccionORM)
                .filter(
                    SeccionORM.nombre == str(row["seccion"]).strip(),
                    SeccionORM.grado == str(row["grado_txt"]).strip(),
                )
                .first()
            )
            if not s:
                session.add(
                    SeccionORM(
                        nombre=str(row["seccion"]).strip(),
                        grado=str(row["grado_txt"]).strip(),
                    )
                )
        session.commit()

        for _, row in df.iterrows():
            dni = str(row.get("dni", "")).strip()
            if not dni:
                continue
            sec = (
                session.query(SeccionORM)
                .filter(
                    SeccionORM.nombre == str(row.get("seccion", "")).strip(),
                    SeccionORM.grado == str(row.get("grado_txt", "")).strip(),
                )
                .first()
            )
            est = session.get(EstudianteORM, dni)
            if est is None:
                est = EstudianteORM(DNI=dni)
                session.add(est)
            est.nombre = str(row.get("nombre_completo", dni)).strip() or dni
            try:
                est.nota_proy = float(row["nota_proyectada_b4"])
            except (TypeError, ValueError):
                est.nota_proy = None
            est.estado = "Aprueba" if int(float(row.get("target_aprueba_binario") or 0)) == 1 else "Desaprueba"
            est.id_seccion = sec.id if sec else None
        session.commit()
    engine.dispose()
    logger.info("SQLAlchemy ORM: %s", db_path)
    return db_path.resolve()


def graficos_curriculum(df: pd.DataFrame, plots_dir: Path) -> list[Path]:
    """Figuras adicionales tipo informe (Seaborn + Matplotlib)."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    out: list[Path] = []

    dfp = enriquecer_para_powerbi(df.copy())
    if "estado_prediccion_txt" not in dfp.columns and "target_aprueba_binario" in dfp.columns:
        dfp["estado_prediccion_txt"] = dfp["target_aprueba_binario"].map(
            {0: "Desaprueba", 1: "Aprueba"}
        )

    # Barras por sección
    if "seccion" in dfp.columns and "estado_prediccion_txt" in dfp.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        conteo = dfp.groupby(["seccion", "estado_prediccion_txt"], observed=False).size().unstack(fill_value=0)
        conteo.plot(kind="bar", ax=ax, color=["#e74c3c", "#2ecc71"], edgecolor="black")
        ax.set_title("Predicción por sección", fontweight="bold")
        ax.set_xlabel("Sección")
        ax.set_ylabel("Estudiantes")
        ax.legend(["Desaprueba", "Aprueba"])
        fig.tight_layout()
        p = plots_dir / "06_barras_seccion_prediccion.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        out.append(p)

    # Torta global
    if "estado_prediccion_txt" in dfp.columns:
        fig, ax = plt.subplots(figsize=(6, 6))
        valores = dfp["estado_prediccion_txt"].value_counts()
        ax.pie(
            valores,
            labels=valores.index,
            autopct="%1.1f%%",
            colors=["#2ecc71", "#e74c3c"],
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        ax.set_title("Distribución predicción SATE-SR", fontweight="bold")
        fig.tight_layout()
        p = plots_dir / "07_torta_prediccion.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        out.append(p)

    # Línea evolución promedios
    cols = ["nota_bim1", "nota_bim2", "nota_bim3", "nota_proyectada_b4"]
    if all(c in dfp.columns for c in cols):
        fig, ax = plt.subplots(figsize=(8, 5))
        promedios = [dfp[c].astype(float).mean() for c in cols]
        bimestres = ["Bim 1", "Bim 2", "Bim 3", "Proy. B4"]
        ax.plot(bimestres, promedios, marker="o", color="steelblue", linewidth=2.5, markersize=8)
        ax.axhline(12, color="red", linestyle="--", linewidth=1.5, label="Umbral 12")
        ax.set_title("Evolución promedio de notas", fontweight="bold")
        ax.legend()
        fig.tight_layout()
        p = plots_dir / "08_linea_evolucion_notas.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        out.append(p)

    # Dashboard 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("SATE-SR — Dashboard integrador", fontweight="bold", fontsize=14)
    if "estado_prediccion_txt" in dfp.columns and "genero" in dfp.columns:
        dfp["_gen"] = dfp["genero"].map(_normalizar_genero)
        sns.countplot(data=dfp, x="estado_prediccion_txt", hue="_gen", ax=axes[0, 0], palette="Set2")
        axes[0, 0].set_title("Predicción por género")
    melt_cols = [c for c in ("nota_bim1", "nota_bim2", "nota_bim3", "nota_proyectada_b4") if c in dfp.columns]
    if len(melt_cols) >= 2:
        df_melt = dfp[melt_cols].melt(var_name="Bimestre", value_name="Nota")
        sns.boxplot(data=df_melt, x="Bimestre", y="Nota", ax=axes[0, 1], palette="Blues", hue="Bimestre", legend=False)
        axes[0, 1].axhline(12, color="red", linestyle="--", linewidth=1.0)
        axes[0, 1].set_title("Distribución de notas")
    corr_cols = [
        c
        for c in (
            "nota_bim1",
            "nota_bim2",
            "nota_bim3",
            "analisis_asistencia",
            "analisis_incidencias",
            "analisis_sentimiento",
            "analisis_situacion_familiar",
            "nota_proyectada_b4",
        )
        if c in dfp.columns
    ]
    if len(corr_cols) >= 2:
        sns.heatmap(
            dfp[corr_cols].corr(numeric_only=True),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=axes[1, 0],
            linewidths=0.5,
        )
        axes[1, 0].set_title("Correlaciones")
    if "nota_proyectada_b4" in dfp.columns:
        sns.histplot(dfp["nota_proyectada_b4"].astype(float), kde=True, ax=axes[1, 1], bins=18, color="steelblue")
        axes[1, 1].axvline(12, color="red", linestyle="--", linewidth=1.2)
        axes[1, 1].set_title("Histograma nota proyectada B4")
    fig.tight_layout()
    p = plots_dir / "09_dashboard_integrador.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    out.append(p)

    return out


def entrenamiento_modelos_comparados(df: pd.DataFrame, plots_dir: Path) -> dict[str, Any]:
    """Regresión logística vs Random Forest + regresión lineal nota B4 + KMeans (bonus)."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    p_reg: Path | None = None
    features = [
        "nota_bim1",
        "nota_bim2",
        "nota_bim3",
        "analisis_asistencia",
        "analisis_incidencias",
        "analisis_sentimiento",
        "analisis_situacion_familiar",
    ]
    target = "target_aprueba_binario"
    cols = [c for c in features if c in df.columns]
    if target not in df.columns or len(cols) < 3:
        logger.warning("ML extendido: faltan columnas; se omite.")
        return {}

    dfx = df[cols].copy()
    for c in cols:
        dfx[c] = pd.to_numeric(dfx[c], errors="coerce")
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int).values

    le = LabelEncoder()
    gen = df["genero"].fillna("M").astype(str).str.strip().str.upper() if "genero" in df.columns else pd.Series(["M"] * len(df))
    gen = gen.replace(
        {"MASCULINO": "M", "HOMBRE": "M", "MALE": "M", "FEMENINO": "F", "MUJER": "F", "FEMALE": "F"}
    )
    X_extra = le.fit_transform(gen)
    dfx["genero_enc"] = X_extra

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(dfx.values)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if len(np.unique(y)) < 2:
        logger.warning("ML extendido: una sola clase en target; se omite train/test.")
        return {}

    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=0.2, random_state=42, stratify=y
    )
    lr = LogisticRegression(random_state=42, max_iter=800)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    rf = RandomForestClassifier(n_estimators=120, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    # ROC sobre conjunto completo con probabilidades RF (más estable que binarios sueltos)
    y_proba = rf.predict_proba(Xs)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf), display_labels=["Desaprueba", "Aprueba"]).plot(
        cmap="Blues", ax=ax1
    )
    ax1.set_title("Matriz confusión — Random Forest (holdout)")
    ax2.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax2.plot([0, 1], [0, 1], "navy", linestyle="--", lw=1)
    ax2.set(xlabel="FPR", ylabel="TPR", title="Curva ROC (prob. RF, todo el set)")
    ax2.legend(loc="lower right")
    fig.tight_layout()
    p_roc = plots_dir / "10_validacion_rf_roc.png"
    fig.savefig(p_roc, dpi=200)
    plt.close(fig)

    reg_cols = [c for c in ("nota_bim1", "nota_bim2", "nota_bim3", "analisis_asistencia", "analisis_situacion_familiar") if c in df.columns]
    res_reg: dict[str, float] = {}
    if reg_cols and "nota_proyectada_b4" in df.columns:
        Xr = SimpleImputer(strategy="median").fit_transform(df[reg_cols].apply(pd.to_numeric, errors="coerce"))
        yr = pd.to_numeric(df["nota_proyectada_b4"], errors="coerce")
        yr = yr.fillna(yr.median()).values
        lin = LinearRegression()
        lin.fit(Xr, yr)
        pred = lin.predict(Xr)
        res_reg = {
            "mse": float(mean_squared_error(yr, pred)),
            "rmse": float(np.sqrt(mean_squared_error(yr, pred))),
            "r2": float(r2_score(yr, pred)),
        }
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(yr, pred, alpha=0.5, s=25)
        ax.plot([5, 20], [5, 20], "r--", lw=1)
        ax.set_title(f"Regresión lineal nota B4 — R²={res_reg['r2']:.3f}")
        ax.set_xlabel("Observado")
        ax.set_ylabel("Predicho")
        fig.tight_layout()
        p_reg = plots_dir / "11_regresion_lineal_nota_b4.png"
        fig.savefig(p_reg, dpi=200)
        plt.close(fig)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(Xs)
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in range(3):
        m = clusters == c
        ax.scatter(X2[m, 0], X2[m, 1], s=22, alpha=0.65, label=f"Cluster {c}")
    ax.set_title("KMeans + PCA (2D)")
    ax.legend()
    fig.tight_layout()
    p_km = plots_dir / "12_kmeans_pca.png"
    fig.savefig(p_km, dpi=200)
    plt.close(fig)

    importancias = pd.Series(rf.feature_importances_, index=list(dfx.columns)).sort_values(ascending=False)
    print("\n--- ML extendido (sklearn) ---")
    print(f"  Accuracy Regresión logística: {acc_lr:.4f}")
    print(f"  Accuracy Random Forest:       {acc_rf:.4f}")
    print(f"  AUC-ROC (RF proba, full):    {roc_auc:.4f}")
    if res_reg:
        print(f"  Regresión nota B4 — R²: {res_reg['r2']:.4f}  RMSE: {res_reg['rmse']:.4f}")
    print("\n  Importancia features (RF):")
    print(importancias.round(4).to_string())
    print(classification_report(y_test, y_pred_rf, target_names=["Desaprueba", "Aprueba"]))

    graficos_extra: list[Path] = [p_roc, p_km]
    if p_reg is not None:
        graficos_extra.append(p_reg)
    return {
        "accuracy_logistic": acc_lr,
        "accuracy_random_forest": acc_rf,
        "auc_roc_rf_full": roc_auc,
        "regresion_nota_b4": res_reg,
        "graficos": graficos_extra,
    }


def ejecutar_pipeline_integrador(
    uri: str,
    db_name: str,
    anio_lectivo: int,
    resultado: dict[str, Any],
    df_ml: pd.DataFrame,
    root: Path,
    plots_dir: Path,
    dni_allowlist: Optional[Set[str]] = None,
) -> dict[str, Any]:
    """
    Orquesta: JSON encuesta, agregación Mongo, SQLite, ORM opcional, figuras extra, ML comparativo.
    """
    root = Path(root)
    plots_dir = Path(plots_dir)
    salida_ml = root / "salida_ml"
    salida_ml.mkdir(parents=True, exist_ok=True)

    resumen: dict[str, Any] = {"rutas": []}
    df_ml = enriquecer_nombres_desde_resultado(resultado, df_ml)
    if dni_allowlist is not None and len(dni_allowlist) > 0 and "dni" in df_ml.columns:
        allow = {str(x).strip() for x in dni_allowlist}
        df_ml = df_ml[df_ml["dni"].astype(str).str.strip().isin(allow)].copy()

    # --- NoSQL: muestras solo lectura + export JSON ---
    try:
        agg = agregacion_asistencia_por_grado(uri, db_name, anio_lectivo, dni_allowlist=dni_allowlist)
        print("\n--- MongoDB: agregación asistencia por grado ---")
        for r in agg[:12]:
            print(f"  Grado {r.get('_id')}: n={r.get('total_estudiantes')} secciones={r.get('secciones')}")
        jpath = salida_ml / "encuesta_export_integrador.json"
        export_encuesta_json(uri, db_name, anio_lectivo, jpath, dni_allowlist=dni_allowlist)
        resumen["rutas"].append(str(jpath))
    except Exception as e:
        logger.exception("Fallo bloque Mongo integrador: %s", e)
        print(f"[AVISO] Bloque Mongo integrador omitido: {e}")

    # --- SQLite + ORM ---
    try:
        db_sql = salida_ml / "sate_sr.db"
        sqlite_export_desde_df_ml(df_ml, db_sql)
        resumen["rutas"].append(str(db_sql))
        gc.collect()
        orm_path = sqlalchemy_orm_opcional(df_ml, salida_ml / "sate_sr_orm.db")
        if orm_path:
            resumen["rutas"].append(str(orm_path))
    except Exception as e:
        logger.exception("SQLite/ORM: %s", e)
        print(f"[AVISO] SQLite/ORM: {e}")

    # --- Gráficos + ML ---
    try:
        for p in graficos_curriculum(df_ml, plots_dir):
            resumen["rutas"].append(str(p))
        ml_info = entrenamiento_modelos_comparados(df_ml, plots_dir)
        resumen["ml"] = ml_info
        for p in ml_info.get("graficos") or []:
            ps = str(p)
            if ps not in resumen["rutas"]:
                resumen["rutas"].append(ps)
    except Exception as e:
        logger.exception("Gráficos/ML extendido: %s", e)
        print(f"[AVISO] Gráficos/ML extendido: {e}")

    # --- Tabla resumen interpretación (consola) ---
    m = resultado.get("metricas") or {}
    auc_m = m.get("auc_roc")
    print(
        "\n--- Interpretación rápida (motor SATE + extensión sklearn) ---\n"
        f"  AUC-ROC motor (valid. temporal): {auc_m}\n"
        "  MongoDB: datos semiestructurados y agregaciones flexibles.\n"
        "  SQLite: resultado tabular con JOIN para informes locales.\n"
    )

    return resumen


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
if not os.getenv("MONGODB_URI"):
    load_dotenv(ROOT.parent / "Bigdata_Proyecto_final" / ".env")
if not os.getenv("MONGODB_URI"):
    load_dotenv(ROOT.parent / "escuela_datos_sinteticos" / ".env")

def main() -> None:
    default_ml = ROOT / "salida_ml" / "dataset_ml_solo5to.csv"
    parser = argparse.ArgumentParser(
        description="SATE-SR solo alumnos de 5.° secundaria (ancla por año en primer_bimestre)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprime solo el JSON completo del resultado",
    )
    parser.add_argument(
        "--anio",
        type=int,
        default=None,
        help="Año lectivo ancla (default: ANIO_LECTIVO del .env o 2025)",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Genera PNG en --plots-dir",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help=f"Carpeta para gráficos (default: {ROOT / 'salida_reportes' / 'solo5to'})",
    )
    parser.add_argument(
        "--export-ml",
        nargs="?",
        const=str(default_ml),
        default=None,
        metavar="CSV",
        help=f"Exporta CSV ML (default: {default_ml})",
    )
    parser.add_argument(
        "--integrador",
        action="store_true",
        help="Pipeline integrador (Mongo JSON, SQLite, sklearn) alineado a los mismos DNIs",
    )
    args = parser.parse_args()

    uri = os.getenv("MONGODB_URI", "").strip()
    if not uri:
        sys.stderr.write(
            "Falta MONGODB_URI. Crea .env en esta carpeta o usa el de Bigdata_Proyecto_final / escuela_datos_sinteticos.\n"
        )
        sys.exit(1)

    db_name = os.getenv("MONGODB_DB_NAME", "bigData_San_ramon").strip()
    anio_env = os.getenv("ANIO_LECTIVO", "2025").strip()
    anio = args.anio if args.anio is not None else int(anio_env)

    print("=" * 60)
    print("  Solo_5to_cohorte — SATE-SR (solo 5.° secundaria, ancla)")
    print("=" * 60)
    print(f"  Base de datos: {db_name}")
    print(f"  Año ancla:     {anio}")
    print(f"  Motor:         {ROOT / 'motor_sate_solo5to.py'}")
    print("=" * 60)

    dnis = dnis_quintos_secundaria_anio(uri, db_name, anio)
    if not dnis:
        print(
            "\n[ERROR] No hay DNIs con grado 5 en el año ancla.\n"
            "  Se probó: primer_bimestre (con y sin nivel), luego asistencia.\n"
            "  Revisa anio_lectivo y columnas GRADO/Grado en Mongo.\n"
        )
        sys.exit(2)

    print(f"\n[INFO] 5.° secundaria en {anio}: {len(dnis)} estudiantes (DNI únicos).")

    resultado = ejecutar_analisis_sate(uri, db_name, anio_lectivo=anio, dni_allowlist=dnis)

    if args.json:
        print(json.dumps(resultado, ensure_ascii=False, indent=2))
        return

    if not resultado.get("success"):
        print("Error:", resultado.get("error", resultado))
        sys.exit(1)

    m = resultado.get("metricas", {})
    fr = resultado.get("factores_riesgo", {})
    print("\n--- Resumen (solo 5.° secundaria) ---")
    print(f"Total estudiantes:     {resultado.get('total_estudiantes')}")
    print(f"Aprueban (pred.):      {m.get('aprueba')} ({m.get('porcentaje_aprueba', 0):.1f}%)")
    print(f"Desaprueban (pred.):   {m.get('desaprueba')} ({m.get('porcentaje_desaprueba', 0):.1f}%)")
    print(f"Nota proyectada (avg): {m.get('promedio_nota_proyectada', 0):.2f}")
    for k in ("precision", "recall", "f1_score", "auc_roc"):
        if k in m:
            label = "F1" if k == "f1_score" else k.upper()
            print(f"{label:<12} {m[k]:.4f}")

    mc = m.get("matriz_confusion")
    if isinstance(mc, dict):
        print("\n--- Matriz de confusión (validación temporal) ---")
        for clave, val in mc.items():
            print(f"  {clave}: {val}")

    print("\n--- Factores de riesgo (conteos) ---")
    for nombre, bloque in fr.items():
        if isinstance(bloque, dict):
            print(f"  {nombre}: sin_riesgo={bloque.get('sin_riesgo')}  con_riesgo={bloque.get('con_riesgo')}")

    rows = resultado.get("resultados") or []
    _vistos: set[str] = set()
    en_riesgo: list = []
    for r in rows:
        if r.get("Prediccion_Final_Binaria") != 0:
            continue
        dni_k = str(r.get("DNI", "")).strip()
        if dni_k and dni_k in _vistos:
            continue
        if dni_k:
            _vistos.add(dni_k)
        en_riesgo.append(r)
    print(f"\n--- Estudiantes con predicción desaprueba ({len(en_riesgo)}) — máx. 25 ---")
    for r in en_riesgo[:25]:
        nom = (r.get("Apellidos_Nombres") or "")[:40]
        print(
            f"  {r.get('DNI')} | {r.get('Grado')} {r.get('Seccion')} | "
            f"NotaB4≈{r.get('Nota_Proyectada_B4')} | {nom}"
        )
    if len(en_riesgo) > 25:
        print(f"  ... y {len(en_riesgo) - 25} más. Usa --json para la lista completa.")

    plots_dir = args.plots_dir if args.plots_dir is not None else ROOT / "salida_reportes" / "solo5to"
    if args.plots:
        paths = generar_graficos(resultado, plots_dir)
        print(f"\n--- Gráficos ({len(paths)}) en {plots_dir} ---")
        for p in paths:
            print(f"  {p}")

    if args.export_ml is not None:
        df_ml = resultado_a_dataframe_ml(resultado)
        out_csv = Path(args.export_ml)
        export_ml_csv(df_ml, out_csv)
        salida = out_csv.parent
        export_para_powerbi(df_ml, salida / "dataset_powerbi_solo5to.csv")
        dash_path = salida / "reporte_dashboard_completo_solo5to.csv"
        export_reporte_dashboard_completo_csv(resultado, dash_path)
        print(f"\n--- CSV exportados ({salida.resolve()}) ---")
        print(f"  ML estándar:     {out_csv.name} ({len(df_ml)} filas)")
        print("  Power BI:        dataset_powerbi_solo5to.csv")
        print("  Dashboard:       reporte_dashboard_completo_solo5to.csv")

    if args.integrador:
        plots_integrador = args.plots_dir if args.plots_dir is not None else ROOT / "salida_reportes" / "solo5to"
        df_int = resultado_a_dataframe_ml(resultado)
        if df_int.empty:
            print("\n[AVISO] --integrador omitido: dataset ML vacío.")
        else:
            print("\n--- Pipeline integrador ---")
            res_int = ejecutar_pipeline_integrador(
                uri, db_name, anio, resultado, df_int, ROOT, Path(plots_integrador), dni_allowlist=dnis
            )
            print(f"  Artefactos generados: {len(res_int.get('rutas', []))}")
            for ruta in res_int.get("rutas", [])[:20]:
                print(f"    {ruta}")
            if len(res_int.get("rutas", [])) > 20:
                print(f"    ... y {len(res_int['rutas']) - 20} más.")

    print("\n[Listo] Solo_5to_cohorte.")


if __name__ == "__main__":
    main()
