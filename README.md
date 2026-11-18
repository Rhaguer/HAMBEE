# HAMBEE · EVA4 de Machine Learning

Clasificación P/NP en CPU con MobileNetV3-Small, modelo ONNX **1.1.0** de la **época 37**. El panel ejecuta la prueba reservada y procesa el dataset completo. La API documentada permanece en `/docs`.

[Descargar entrega completa v1.1.0](https://github.com/Rhaguer/HAMBEE/releases/tag/v1.1.0) · [Informe PDF](docs/EVA4_Informe_Haguer_Maldonado.pdf) · [Guía de Colab](colab/README.md)

## Ejecutar en VS Code

1. Descomprime la entrega completa. Abre la carpeta `EVA4` en VS Code.
2. Selecciona **Python 3.12** con la extensión oficial de Python. Pulsa **F5** y elige **HAMBEE · Ejecutar evaluación completa** si se muestra el selector.
3. El lanzador prepara las dependencias, abre el panel en el navegador y ejecuta la demostración automáticamente. Al finalizar, el panel habilita la descarga del CSV con 7.140 predicciones.

En Windows también puedes hacer doble clic en `Ejecutar HAMBEE.cmd`. La primera instalación requiere acceso a internet. Para abrir el panel manualmente desde una terminal:

```powershell
py -3.12 ejecutar.py
```

El botón **Analizar dataset** procesa las imágenes precargadas. La galería muestra la etiqueta real y la predicción de cada imagen y permite filtrar por partición o nombre. La carga de una imagen propia es opcional. El ZIP de entrega coloca `Dataset.zip` en `EVA4/data/`. En una copia descargada de Git, coloca el dataset en esa ruta o configura su ubicación mediante la variable `HAMBEE_DATASET`. La prueba de 71 originales está incluida en `validacion/`.

## Docker

```powershell
docker build -t hambee:1.1.0 .
docker run --rm -p 127.0.0.1:8000:8000 hambee:1.1.0
```

Abre `http://127.0.0.1:8000`. Docker incluye el panel y la prueba reservada. Para habilitar el procesamiento de las 7.140 imágenes, monta el archivo del paquete en modo lectura:

```powershell
$dataset = (Resolve-Path .\data\Dataset.zip).Path
docker run --rm -p 127.0.0.1:8000:8000 -e HAMBEE_DATASET=/data/Dataset.zip --mount "type=bind,source=$dataset,target=/data/Dataset.zip,readonly" hambee:1.1.0
```

La imagen se ejecuta con un usuario sin privilegios y comprueba la salud de FastAPI. El flujo de CI valida la API, los umbrales de calidad y latencia, los tamaños de lote, y la imagen por HTTP. Los paquetes de versiones etiquetadas pueden descargarse como artefactos del flujo correspondiente.

## Colab

Abre el notebook de la carpeta separada y presiona **▶** en la única celda. Cuando aparezca el selector, carga el ZIP de entrega sin descomprimir. Colab instala las dependencias y genera la prueba de la API, la evaluación, los gráficos y el CSV. El dataset completo viaja dentro del ZIP.

## Evidencia y reproducción

- `models/metadata.json` registra 40 épocas, selección por exactitud y pérdida de validación, y huellas SHA-256.
- `evidencia/benchmark.json`, `lotes-cpu.json` y `procesamiento-dataset.json` separan latencia de inferencia, capacidad de lote y tiempo total. La exactitud se calcula sobre las 71 imágenes de prueba.
- `evidencia/api-local.json` y `api-docker.json` describen las pruebas de la versión 1.1.0.
- `evidencia/comparacion-epocas.json` compara los modelos de las épocas 2 y 37.

Para reproducir las métricas desde una terminal dentro de la carpeta, usando un entorno Python 3.12 con `requirements-test.txt`:

```powershell
python smoke.py validacion/test/P/P10109-127r.jpg --panel
python benchmark.py validacion --output benchmark-local.json
python quality_gate.py benchmark-local.json
python benchmark_lotes.py validacion --output lotes-local.json
python procesar_dataset.py data/Dataset.zip --output resultados-dataset
```

Para repetir el ajuste, extrae el dataset original e instala `requirements-train.txt` en un entorno independiente. Ejecuta `train.py <carpeta_dataset> --epochs 40` en una copia de trabajo, ya que vuelve a generar los archivos de `models/`. El uso normal del panel y del notebook reutiliza el modelo exportado.

