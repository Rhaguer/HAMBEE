# Guía de defensa técnica de la EVA4

Autores: Ricardo Haguer y Yerko Maldonado. Modelo HAMBEE 1.2.0.

## Correspondencia con la pauta

| Criterio | Evidencia para mostrar |
|---|---|
| 4.1.1 Optimización y recursos | Comparación FP32 sin optimización/optimizado, ensayo INT8 en validación, RAM RSS, lotes 1/2/4/8/16 y justificación de la elección. |
| 4.1.2 Plan de acción | Fases con entradas, entregables, condición de salida y reversión, desarrolladas en el informe. |
| 4.1.3 Integración continua | Workflow aprobado, umbrales automáticos, registro MLflow, SHA-256 de modelo/datos, tag y artefacto de imagen verificado. |
| 4.1.4 Implementación práctica | FastAPI funcionando, documentación OpenAPI, carga de imagen, rechazo de entradas inválidas y contenedor probado por HTTP. |

## Secuencia sugerida de demostración

1. Abrir `Ejecutar HAMBEE.cmd` en Windows con Python 3.12 instalado. Esperar a que se muestre el panel; el primer inicio descarga dependencias.
2. Ejecutar la evaluación y mostrar los 67 aciertos de las 71 imágenes reservadas. Diferenciar ese resultado del procesamiento de las 714 imágenes, que incluye entrenamiento y validación.
3. Mostrar los cuatro errores de la clase P en la matriz de confusión y explicar el recall de P: 33/37 = 89,19 %.
4. Cargar una imagen JPEG/PNG y mostrar clase, puntajes y tiempo del runtime. Explicar que el softmax no garantiza una probabilidad calibrada.
5. Abrir `/docs` y `/health`. Explicar 422 sin archivo, 415 con contenido inválido y 413 con exceso de tamaño o píxeles.
6. Mostrar el pipeline aprobado y sus dos jobs de validación y despliegue de liberación, el registro MLflow y `api-release-ci.json`.
7. Mostrar cómo se descargan la imagen y su SHA-256, cómo se verifica y carga, y cómo se ejecuta localmente. El runner de aceptación es temporal.

## Preguntas y respuestas de apoyo

**¿Por qué ONNX Runtime en CPU?** La red exportada permite inferencia sin instalar PyTorch en el servicio. La optimización del grafo reduce la latencia y mantiene las clases predichas. La CPU disponible basta para el alcance medido.

**¿Por qué no se usa INT8?** El ensayo reduce el tamaño del archivo y el RSS, pero empeora considerablemente el F1 macro y la latencia en la plataforma medida. Se conserva FP32 optimizado. Cuantizar no garantiza mayor velocidad ni calidad equivalente.

**¿Se implementó batching dinámico?** Se midieron lotes fijos. La API atiende una imagen por solicitud. El batching dinámico requiere una cola de solicitudes, tamaño máximo y tiempo de espera; se explica como evolución con Triton y exige medir p95 HTTP bajo concurrencia. No se presenta como una funcionalidad implementada.

**¿Qué aportaría TensorRT?** En una GPU NVIDIA compatible se evaluaría FP16/INT8 con perfiles de lote, uso de VRAM y paridad numérica. No se atribuyen resultados GPU a las mediciones CPU.

**¿Qué impide la fuga de datos?** El manifiesto registra SHA-256 y grupo de origen; las 714 imágenes son distintas y los grupos no se comparten entre entrenamiento, validación y prueba. La selección de época utiliza validación. El test no se usa para elegir entre FP32 e INT8.

**¿Qué bloquea una liberación defectuosa?** Fallos de integridad, contrato HTTP, discrepancias de predicción al optimizar, F1 macro inferior a 0,94 o p95 del runtime superior a 25 ms. También bloquea una prueba Docker o el registro de evidencia que falle.

**¿MLflow reentrena el modelo?** No. Registra la validación actual y los artefactos versionados. Las curvas anteriores se importan con una etiqueta explícita de historial.

**¿El 94,37 % garantiza rendimiento en cualquier colmena?** No. Es el resultado sobre 71 imágenes del corpus disponible. Se necesitan datos independientes de otras colmenas, cámaras y condiciones de iluminación para estimar generalización.

## Operación y reversión

Registrar p95 HTTP, errores 4xx/5xx, uso de RAM, distribución P/NP y calidad de nuevas imágenes etiquetadas. Comparar datos nuevos con una referencia antes de reentrenar; el drift es una señal para investigar y no demuestra por sí solo pérdida de calidad. Conservar imagen y huella de la versión anterior. Para revertir, cargar esa imagen, comprobar su huella y `/health`, ejecutar una predicción conocida y cambiar el puerto o el enrutamiento hacia la instancia recuperada. No sustituir el modelo dentro de un contenedor en ejecución.
