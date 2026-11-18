# HAMBEE · EVA4 de Machine Learning

Servicio de inferencia binaria P/NP con MobileNetV3-Small y ONNX Runtime en CPU.
Consulte el [informe técnico en PDF](docs/EVA4_Informe_Haguer_Maldonado.pdf), el [Word editable](docs/EVA4_Informe_Haguer_Maldonado.docx) y `evidencia/` para los resultados medidos.

## Inicio local

Desde la carpeta del proyecto, con Python 3.12:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m uvicorn app:app --host 127.0.0.1 --port 8000 --limit-concurrency 16
```

Abra http://127.0.0.1:8000/docs para cargar una imagen en `POST /predict`.
También puede probar con una muestra original incluida:

```powershell
$muestra = (Get-ChildItem validacion/test/P/*.jpg | Select-Object -First 1).FullName
curl.exe -fsS http://127.0.0.1:8000/health
curl.exe -fsS -F "file=@$muestra" http://127.0.0.1:8000/predict
```

`POST /predict` recibe el campo multipart `file` (JPEG o PNG; hasta 5 MiB de
archivo y 12 megapíxeles). Devuelve clase, probabilidades, confianza, versión y
tiempo de ejecución del modelo en milisegundos. La confianza es un puntaje softmax.
Los códigos 413, 415 y 422 indican, respectivamente, exceso de tamaño, imagen
inválida y campo obligatorio ausente. Un proxy de producción debería limitar
también el cuerpo HTTP completo antes de procesar multipart.

## Docker

```sh
docker build -t eva4:1.0.0 .
docker run --rm -p 127.0.0.1:8000:8000 eva4:1.0.0
```

La imagen usa un usuario sin privilegios y expone una comprobación de salud.
Se construyó y probó en Docker Engine sobre Debian 12 en una máquina virtual
local. El contenedor alcanzó el estado `healthy` y superó las pruebas HTTP de
inferencia y validación. Véase `evidencia/api-docker.json`. La ejecución nativa
en Windows se documenta en `evidencia/api-local.json`.

GitHub Actions incluye la misma prueba por HTTP y queda disponible al situar
el contenido de esta carpeta en la raíz
del repositorio. Una etiqueta `v1.0.0` genera una imagen exportable tras superar
las verificaciones. Para desplegar esa imagen en otro host, usar `docker load -i
eva4-image.tar.gz` y ejecutar el comando anterior. Para revertir, conservar y
volver a ejecutar la imagen de la versión aprobada anterior.

## Validación y reproducción

```powershell
.venv/Scripts/python -m pip install -r requirements-test.txt
.venv/Scripts/python smoke.py $muestra --url http://127.0.0.1:8000
.venv/Scripts/python benchmark.py validacion --output benchmark-local.json
.venv/Scripts/python quality_gate.py benchmark-local.json
```

Omita `--url` para ejecutar la prueba contractual sin levantar el servidor.
La carpeta `validacion/` incluye los 71 originales de prueba y su manifiesto,
obtenidos del RAR de la EVA3. Las métricas evalúan la concordancia con las
etiquetas heredadas. Para repetir el entrenamiento, extraiga el RAR original
junto a la carpeta del proyecto e instale las dependencias de entrenamiento en
un entorno independiente con Python 3.12.

```powershell
python -m pip install -r requirements-train.txt
python train.py ../Dataset_Abejas_10x --epochs 12
```

El entrenamiento verifica los SHA-256 del manifiesto y descarga los pesos
iniciales de ImageNet desde PyTorch. Conserva la primera época con mejor
exactitud en validación y genera el ONNX y sus metadatos. La semilla fija reduce
la variabilidad, con posibles diferencias por hardware o backend. Si se publica
un nuevo modelo, actualizar su versión en los metadatos, la versión de la API y
la etiqueta de la imagen. La procedencia del modelo queda registrada mediante
el SHA-256 del ONNX, el SHA-256 del manifiesto original, las versiones de
dependencias, la semilla y el historial de entrenamiento.

