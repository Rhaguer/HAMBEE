# HAMBEE | Implementación y optimización

Informe técnico defensivo - Evaluación Sumativa N.º 4 de Machine Learning

| Identificación | Datos |
| --- | --- |
| Autores | Ricardo Haguer y Yerko Maldonado |
| Carrera / sede | Ingeniería en Informática / INACAP Puente Alto |
| Asignatura / sección | Machine Learning, TIEL26 / 101-8V |
| Docente | Francisco Valdivieso Tapia |
| Fecha / modelo | 18 de noviembre de 2026 / HAMBEE 1.2.0 |

## Resumen ejecutivo

Se implementó un servicio local de clasificación de abejas con polen (P) y sin polen (NP), basado en MobileNetV3-Small exportado a ONNX. FastAPI expone inferencia y un panel de demostración. El contenedor se construyó, se probó por HTTP y se liberó mediante un pipeline que conserva evidencia y registra los artefactos en MLflow.

| Indicador | Resultado de la evidencia |
| --- | --- |
| Prueba reservada | 67/71 aciertos; exactitud 94,37 %; F1 macro 0,9437 |
| p95 del runtime en CPU | 0.86 ms; 355 mediciones con tensores precargados |
| p95 HTTP en Docker | 3.18 ms; 142 solicitudes secuenciales |
| Datos | 714 originales; 571 entrenamiento, 72 validación y 71 prueba |
| Liberación | Pipeline aprobado; dos jobs; tag v1.2.0 |

El resultado de prueba describe el corpus disponible. No demuestra generalización a otras colmenas, cámaras o condiciones de luz. El despliegue automatizado de aceptación es temporal; la demostración persistente se ejecuta en el equipo local.

# 1. Correspondencia con la pauta

| Criterio | Implementación y evidencia |
| --- | --- |
| 4.1.1 Optimización y recursos | ONNX optimizado, comparación FP32/INT8, medición de latencia y RAM, lotes y análisis de alternativas GPU. Secciones 3 y 4. |
| 4.1.2 Plan de acción | Fases secuenciales con entregables y criterios de salida, desde datos hasta operación y reversión. Sección 5. |
| 4.1.3 Integración continua | GitHub Actions, pruebas automáticas de calidad/latencia, MLflow, versionado y despliegue independiente. Sección 6. |
| 4.1.4 Implementación práctica | FastAPI y Docker probados por HTTP, contrato de entrada/salida, excepciones y tiempos medidos. Sección 7. |

## Arquitectura de inferencia

JPEG/PNG → validación de tamaño y formato → RGB, redimensionamiento y normalización → sesión ONNX Runtime en CPU → logits y softmax → clase NP/P y respuesta JSON. El mismo servicio entrega el panel HTML y la documentación OpenAPI.

## Separación de responsabilidades

train.py ajusta y exporta el modelo; dataset.py controla manifiesto, particiones y huellas; preprocess.py concentra el preprocesamiento; app.py implementa el contrato REST; smoke.py prueba el servicio; benchmark.py y benchmark_lotes.py miden inferencia; validar_docker.py comprueba la imagen y su funcionamiento; registrar_mlflow.py conserva la trazabilidad de la validación.

## Criterio de aceptación

La liberación exige integridad de datos y modelo, contrato HTTP correcto, predicciones idénticas al optimizar, F1 macro de prueba >= 0,94 y p95 ONNX <= 25 ms. El umbral de 25 ms pertenece al runtime y no representa una garantía de latencia HTTP bajo concurrencia. Docker y el registro MLflow también deben completar sus verificaciones.

# 2. Datos, ajuste y selección

El conjunto contiene 714 JPEG originales, sin aumentos precomputados. El manifiesto registra ruta, partición, clase, grupo de origen, nombre original y SHA-256. Se verifican hashes únicos, contenido exacto del ZIP y ausencia de grupos compartidos entre particiones.

| Partición | Imágenes | Grupos de origen | Propósito |
| --- | --- | --- | --- |
| Entrenamiento | 571 | 319 | Ajuste de parámetros y calibración INT8 |
| Validación | 72 | 40 | Selección de época y comparación FP32/INT8 |
| Prueba | 71 | 40 | Evaluación reservada del modelo elegido |

MobileNetV3-Small parte de pesos ImageNet y se ajusta durante 40 épocas en CPU con AdamW, tasa 0,0003, lote 32, entropía cruzada, semilla 20261117 y seis hilos. El preprocesamiento produce tensores N×3×96×160, RGB y normalización ImageNet.

La selección maximiza exactitud de validación; ante empate minimiza pérdida y conserva la primera época. Se eligió la época 33: 72/72 aciertos y pérdida 0,02634. La exportación usa opset 17 y eje de lote variable; la paridad PyTorch/ONNX registrada en 72 imágenes tiene error máximo de logits 6,29×10^-5.

| Clase real | Predicción NP | Predicción P | Recall |
| --- | --- | --- | --- |
| NP | 34 | 0 | 100,00 % |
| P | 4 | 33 | 89,19 % |

Los cuatro errores de prueba son falsos negativos de P. Cada imagen equivale a 1,41 puntos porcentuales de exactitud. Se conserva FP32 por la comparación en validación; no se utiliza el test para elegir la cuantización. Las comprobaciones repetidas en CI son controles de regresión sobre el mismo conjunto, no nuevas estimaciones independientes de generalización.

# 3. Optimización y uso de recursos

Comparación de grafo desactivado y ORT_ENABLE_ALL en AMD EPYC 7763 64-Core Processor, ONNX Runtime 1.31.1, CPU y un hilo intraoperador e interoperador. Cada configuración se ejecuta en un proceso separado, con 20 calentamientos y 355 llamadas medidas a session.run. La carga de imagen, el preprocesamiento y HTTP quedan fuera de esta medición.

Figura 1. Latencia del runtime y matriz de confusión sobre 71 imágenes. Fuente: benchmark-ci.json de la liberación.

| Métrica | Sin optimizar | Optimizado |
| --- | --- | --- |
| p50 / p95 (ms) | 3.16 / 3.35 | 0.83 / 0.86 |
| RSS final (MiB) | 126.44 | 130.43 |
| Carga de sesión (ms) | 18.26 | 24.17 |

La optimización reduce el p95 un 74.3 % y conserva las 71 clases predichas. El RSS final varía de 126.44 a 130.43 MiB. RSS incluye Python, bibliotecas y tensores; es una lectura residente final, no el máximo de memoria. No se utiliza GPU ni se mide VRAM.

## Cuantización estática INT8: ensayo y decisión

| Validación: 72 imágenes | FP32 | INT8 QDQ S8S8 |
| --- | --- | --- |
| Tamaño del modelo (MB decimales) | 6.09 | 1.85 |
| p95 (ms) | 0.87 | 1.16 |
| F1 macro | 1.0000 | 0.4996 |
| RSS final (MiB) | 112.68 | 107.14 |

Se calibró por canal con 128 originales de entrenamiento. El ensayo INT8 reduce almacenamiento y RSS, pero pierde calidad y empeora la latencia en AMD EPYC 9V74 80-Core Processor. Se adopta FP32 optimizado. Estos resultados provienen de un ensayo distinto al runner de la figura 1; las comparaciones válidas se realizan dentro de cada ensayo.

# 4. Procesamiento por lotes y alternativas

Figura 2. Historial de validación y rendimiento por lotes. Los lotes se midieron en CPU AMD EPYC 9V74; el historial corresponde al ajuste previo del modelo.

| Lote | p95 del lote (ms) | Imágenes/s |
| --- | --- | --- |
| 1 | 0.91 | 1418.3 |
| 2 | 1.58 | 1602.1 |
| 4 | 3.08 | 1604.6 |
| 8 | 5.53 | 1698.9 |
| 16 | 11.59 | 1666.6 |

Se midieron 25 recorridos intercalados de las 71 imágenes, con tensores precargados y lotes fijos de 1, 2, 4, 8 y 16. Los cinco tamaños conservaron las predicciones del lote unitario. El p95 corresponde a lotes completos; el rendimiento incorpora el último lote parcial. El lote 8 alcanzó el mayor rendimiento del ensayo, sin ser una garantía para otro equipo.

El contenedor liberado también procesó los 714 JPEG, incluyendo lectura del ZIP, SHA-256, preprocesamiento, inferencia y CSV: 0.91 s y 783.1 imágenes/s. Esta medición completa tiene un alcance distinto al benchmark del runtime.

## Batching dinámico y TensorRT

La API implementada recibe una imagen por solicitud. Para aumentar capacidad con solicitudes concurrentes se evaluaría Triton con cola, tamaño máximo de lote y demora máxima de agrupamiento. La ganancia de rendimiento debe contrastarse con la espera añadida al p95 HTTP. Las mediciones de lotes fijos no se presentan como batching dinámico implementado.

TensorRT FP16/INT8 requiere GPU NVIDIA compatible, perfiles de lote y validación de operadores. Se medirían VRAM, transferencias CPU/GPU, latencia extremo a extremo y paridad con FP32. El pruning exigiría evaluar soporte de esparsidad y posiblemente reajustar la red; reducir parámetros no asegura aceleración en un runtime denso. No se atribuyen resultados de estas alternativas al servicio CPU.

# 5. Plan de acción por fases

| Fase | Actividad y entregable | Condición de salida |
| --- | --- | --- |
| 1. Datos y ajuste | Manifiesto, partición por grupos, historial y modelo seleccionado por validación. | 714 originales verificados, grupos sin cruce, selección documentada. |
| 2. Exportación | ONNX opset 17, preprocesamiento único y comparación numérica. | Artefacto válido y paridad PyTorch/ONNX registrada. |
| 3. Optimización | Comparar grafo, INT8 y lotes; medir calidad, RAM y latencia. | Elegir FP32 optimizado y documentar compromisos. |
| 4. Integración REST | FastAPI, OpenAPI, panel, límites y manejo de errores. | Contrato probado, 67/71 y CSV de 714 filas reproducidos. |
| 5. Empaquetado | Dockerfile, dependencias fijadas, usuario 10001 y healthcheck. | Contenedor healthy y pruebas HTTP aprobadas con hashes correctos. |
| 6. CI/MLOps | GitHub Actions, quality gate, registro MLflow y artefactos. | Validación aprobada y registro persistido FINISHED. |
| 7. Liberación | Tag, imagen empaquetada, SHA-256 y despliegue en runner independiente. | Carga y prueba del mismo artefacto liberado aprobadas. |
| 8. Operación | Seguimiento de calidad, latencia, errores, RAM y distribución P/NP. | Investigar degradación y revertir a una imagen previamente validada. |

## Gestión de fallos y reversión

Un fallo de integridad o calidad bloquea la liberación. Un fallo de despliegue exige revisar logs y healthcheck antes de servir solicitudes. Se conserva la imagen anterior con su huella; para revertir se carga, se comprueba /health y una predicción conocida, y se cambia el enrutamiento o puerto hacia esa instancia. El modelo no se modifica dentro de un contenedor en ejecución.

## Continuidad del monitoreo

En una operación prolongada se deben instrumentar p95 HTTP, tasa 4xx/5xx, RAM y distribución de predicciones. Las nuevas etiquetas permiten medir calidad por clase. Un cambio de distribución inicia una investigación; no demuestra por sí solo pérdida de exactitud. Este monitoreo continuo es un plan operativo, mientras que las pruebas automatizadas de liberación sí están ejecutadas y archivadas.

# 6. Integración continua y MLOps

El workflow se ubica en .github/workflows/ci.yml y responde a cambios en main, ramas de validación, pull requests, ejecución manual y tags v*. El job validate instala dependencias, verifica datos y API, mide calidad y lotes, construye y prueba Docker, y registra la validación en MLflow. Cualquier paso fallido detiene la liberación.

| Control | Implementación |
| --- | --- |
| Versionado | Git y tag enlazan código, ONNX y datos; SHA-256 identifica modelo y manifiesto; el manifiesto identifica cada JPEG. |
| Calidad y velocidad | F1 macro >= 0,94; p95 ONNX <= 25 ms; 355 mediciones y paridad de clases. |
| Contrato | Estado, inferencia, errores 413/415/422, panel, galería y CSV completo. |
| MLflow | Parámetros, métricas, commit, hashes, modelo ONNX, manifiesto y JSON de validación. SQLite y artefactos archivados. |
| Despliegue continuo | Un tag válido empaqueta la imagen con SHA-256; deploy la descarga, verifica, carga y prueba en otro runner. |

El modelo y el conjunto de datos son pequeños y quedan preservados en Git. MLflow agrega trazabilidad por ejecución. No se requiere DVC para esta estrategia; en conjuntos de mayor tamaño se evaluaría almacenamiento externo versionado. El registro actual importa el historial anterior con una etiqueta explícita, sin representar esa importación como un nuevo entrenamiento.

## Evidencia ejecutada

Pipeline de liberación: https://github.com/Rhaguer/HAMBEE/actions/runs/44284173203

Commit liberado: 9a6237ff3888ce631e7b0e71e232a1cfafbc9d72

Registro MLflow: 4e6e1ebff39641dc8f5fbadfc361ca2d | estado: FINISHED

Se conservan evidencia-eva4, evidencia-mlops, eva4-image y evidencia-despliegue-eva4. El despliegue de aceptación es efímero y no equivale a un servicio público permanente. El backend SQLite permite inspeccionar parámetros y métricas; los artefactos conservan también los archivos originales en la descarga.

# 7. API y contenedor probados

| Ruta | Contrato y resultado |
| --- | --- |
| GET /health | 200: status ready, versión 1.2.0 y CPUExecutionProvider. |
| POST /predict | Multipart file JPEG/PNG; clase NP/P, confidence, probabilities, model_version e inference_ms. |
| GET /docs | Documentación OpenAPI con entrada, respuesta y ejemplos de error. |
| GET / | Panel de clasificación, evaluación, galería y descarga del CSV. |
| Errores | 422 sin file; 415 con contenido inválido; 413 sobre 5 MiB o 12 megapíxeles. |

Los puntajes softmax suman aproximadamente uno y describen la salida del clasificador; no se presentan como probabilidades calibradas. inference_ms mide exclusivamente session.run. La latencia HTTP incluye envío local, carga y preprocesamiento, por lo que no debe confundirse con la del runtime.

| Prueba de liberación | Resultado |
| --- | --- |
| Docker Engine / plataforma | 29.4.2 / linux/amd64 |
| Usuario / estado | UID 10001 / healthy |
| HTTP secuencial | 142 solicitudes; p50 2.80 ms; p95 3.18 ms |
| Panel / CSV | 714 imágenes y 714 filas verificadas |
| Prueba del clasificador | 67/71; matriz [[34,0],[4,33]] |

El job deploy ejecuta la imagen recuperada del artefacto de liberación, no una reconstrucción independiente. Valida su huella de archivo, healthcheck, usuario sin privilegios, modelo, datos y respuestas HTTP. api-release-ci.json registra el identificador real de imagen y el enlace al pipeline.

Las solicitudes HTTP medidas son secuenciales sobre loopback en el runner. La carga concurrente, los fallos de red y el dimensionamiento para clientes externos requieren otra prueba de capacidad. La implementación local y el contenedor probado satisfacen el alcance local permitido por la pauta.

# 8. Reproducción y demostración

## Ejecución local

Descomprimir la entrega y abrir EVA4. Con Python 3.12 instalado, ejecutar el archivo Ejecutar HAMBEE.cmd o el comando: py -3.12 ejecutar.py. El lanzador crea .venv, instala dependencias y abre el panel; requiere internet durante la primera instalación.

## Construcción y prueba Docker

Desde EVA4, con Docker Engine disponible: docker build -t hambee:1.2.0 . Luego: docker run --rm -p 127.0.0.1:8000:8000 hambee:1.2.0. Abrir http://127.0.0.1:8000. Para la verificación completa instalar requirements-test.txt y ejecutar python validar_docker.py.

## Pruebas reproducibles

| Comando desde EVA4 | Verificación |
| --- | --- |
| python verificar_dataset.py data/Dataset.zip | 714 originales, hashes y particiones. |
| python smoke.py --panel | API, evaluación, galería y CSV. |
| python benchmark.py data/Dataset.zip --output benchmark-ci.json | Latencia, RAM y calidad del runtime. |
| python quality_gate.py benchmark-ci.json | Umbrales y paridad de clases. |
| python benchmark_lotes.py data/Dataset.zip --output lotes-ci.json | Capacidad por tamaño de lote. |
| python registrar_mlflow.py --salida resultados-mlops | Requiere requirements-mlops.txt y evidencia Docker vigente. |

## Secuencia de defensa

Mostrar /health, clasificar un archivo y abrir /docs. Ejecutar la evaluación reservada y descargar las 714 predicciones, diferenciando los dos alcances. Explicar las cuatro equivocaciones P, la elección de FP32 y los compromisos de RAM/latencia. Mostrar el pipeline aprobado, MLflow y la prueba del contenedor liberado. La guía DEFENSA_EVA4.md contiene preguntas y respuestas de apoyo.

Colab ofrece una demostración complementaria en CPU utilizando el mismo ONNX. No se necesita volver a entrenar para defender la entrega. El reentrenamiento completo es opcional y puede variar numéricamente entre entornos.

# 9. Trazabilidad y referencias

## Artefactos de la entrega

Repositorio: https://github.com/Rhaguer/HAMBEE
Versión: https://github.com/Rhaguer/HAMBEE/releases/tag/v1.2.0

La carpeta evidencia/ci-liberacion conserva los resultados descargados del pipeline. liberacion.json enlaza commit, tag, jobs y estado final. El paquete académico agrega este informe y la guía de defensa al código liberado; no modifica los pesos ni los datos de la imagen validada.

Modelo SHA-256: 0269cdc030412fbc0ca55592c5fd3d91bad56ff01058fefec1a7f4a1ead0b203

Manifiesto SHA-256: b47431a5f1a82093081e9bc2a23e049c94e0192ba7f7ce430fda391752b91a83

Imagen Docker ID: sha256:b04f0835b8c99a0416230319e3127e674b5951da146f468e1c3a15da35f74ed0

## Fuentes

[1] INACAP. Evaluación Sumativa N.º 4 de Machine Learning. Unidad 4: Implementación modelo Deep learning. Documento de pauta proporcionado; criterios 4.1.1 a 4.1.4, máximo 24 puntos.

[2] ONNX Runtime. Graph optimizations. https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html

[3] ONNX Runtime. Quantize ONNX models. https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

[4] MLflow. ML Experiment Tracking. https://mlflow.org/docs/latest/ml/tracking/

[5] NVIDIA Triton. Dynamic Batcher. https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html

[6] GitHub. Documentación de artefactos de GitHub Actions. https://docs.github.com/en/actions/using-workflows/storing-workflow-data-as-artifacts

## Alcance y conclusión técnica

La entrega implementa y verifica una solución local de inferencia con API, optimización, empaquetado Docker, integración continua y trazabilidad MLOps. Las evidencias permiten reproducir el funcionamiento y fundamentar la decisión técnica. La evaluación del corpus es limitada y la calificación académica depende de la pauta y de la defensa de los autores.
