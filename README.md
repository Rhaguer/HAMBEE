# HAMBEE EVA4

Clasificador de abejas con polen (P) y sin polen (NP), MobileNetV3-Small v1.2.0, época 33 de 40. El ZIP de datos tiene 714 JPEG originales y su manifiesto. Las particiones por grupo de origen tienen 571 imágenes de entrenamiento, 72 de validación y 71 de prueba.

## Demostración local

Descomprime la entrega, abre `EVA4` en VS Code, selecciona Python 3.12 y pulsa F5. En Windows también puedes abrir `Ejecutar HAMBEE.cmd`. El lanzador crea un entorno virtual en el equipo, instala las dependencias y abre el panel. Imprime la URL con el puerto de la sesión. El primer arranque requiere acceso a PyPI. Para hacerlo desde terminal:

```bash
python ejecutar.py
# Para mostrar una URL fija en la terminal
python ejecutar.py --puerto 8000 --sin-navegador
```

Si Windows bloquea el acceso directo por el publicador, abre PowerShell en EVA4 y ejecuta `py -3.12 ejecutar.py` con tu instalación local de Python.

El panel muestra métricas de las 71 imágenes reservadas, analiza el conjunto de 714 originales, permite explorar la galería por partición y descarga un CSV de 714 filas. También admite una imagen externa JPEG o PNG. El contrato REST está en `/docs`, con `GET /health` y `POST /predict`.

## Contenedor Docker

Ejecuta desde `EVA4`:

```bash
docker build -t hambee:1.2.0 .
docker run --rm -p 127.0.0.1:8000:8000 hambee:1.2.0
```

Abre http://127.0.0.1:8000. La imagen incluye los datos originales y ejecuta la aplicación con un usuario sin privilegios. Para automatizar la construcción, el despliegue local y las pruebas HTTP, ejecuta desde EVA4:

```bash
python -m pip install -r requirements-test.txt
python validar_docker.py
```

El script verifica estado `healthy`, usuario, huellas del modelo y el manifiesto, códigos HTTP, latencia, galería y las 714 filas. Genera `resultados-docker/api-docker.json` al completar todas las pruebas. Con `--solo-probar --imagen hambee:1.2.0` verifica una imagen existente.

## Google Colab

Sube `HAMBEE_EVA4_Colab/HAMBEE_EVA4_Colab.ipynb` a Colab. Ejecuta la celda y selecciona el ZIP completo. La celda instala dependencias en un entorno aislado y descarga un ZIP con resultados de la sesión. El notebook usa la exportación ONNX incluida.

## Reproducir las pruebas

```bash
python -m pip install -r requirements-test.txt
python verificar_dataset.py data/Dataset.zip
python procesar_dataset.py data/Dataset.zip --output resultados
python benchmark.py data/Dataset.zip --output resultados/benchmark.json
python quality_gate.py resultados/benchmark.json
python benchmark_lotes.py data/Dataset.zip --output resultados/lotes-cpu.json
python smoke.py --panel
```

Para comparar FP32 e INT8 en la partición de validación, instala `onnx==1.23.2` y ejecuta `python benchmark_int8.py data/Dataset.zip`. La calibración INT8 usa 128 originales de entrenamiento. El archivo de la prueba experimental se crea de forma temporal.

Para repetir las 40 épocas en CPU en un entorno independiente, instala primero las versiones de CPU desde el índice oficial de PyTorch. El ajuste descarga los pesos iniciales de ImageNet y puede generar diferencias numéricas según el dispositivo y la versión del entorno.

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu torchvision==0.29.1+cpu
python -m pip install -r requirements-train.txt
python train.py data/Dataset.zip --epochs 40 --threads 6 --output modelo-nuevo
```

## Trazabilidad y CI/CD

`models/metadata.json` vincula la versión del modelo, el SHA-256 de ONNX, el SHA-256 del manifiesto, la semilla, la época elegida y el historial. El manifiesto incluye el SHA-256 de cada JPEG. `evidencia/` contiene los resultados de esta versión. `api-http.json` corresponde a Uvicorn local.

Para activar GitHub Actions, sube el contenido de EVA4 a la raíz del repositorio. Si mantienes EVA4 como subcarpeta, coloca `.github/workflows/ci.yml` en la raíz. Valida primero el commit de entrega y crea el tag `v1.2.0` en ese commit:

```bash
git tag v1.2.0
git push origin v1.2.0
```

El workflow valida integridad, API, paridad, F1 macro ≥ 0,94 y p95 ONNX ≤ 25 ms. La ejecución del tag empaqueta la imagen con SHA-256. Un segundo job descarga el artefacto, comprueba su huella, carga la imagen y prueba el despliegue en un runner independiente. Los artefactos se llaman `evidencia-eva4`, `eva4-image` y `evidencia-despliegue-eva4`. Para revertir, conserva la imagen anterior, verifica su huella y vuelve a desplegarla.
