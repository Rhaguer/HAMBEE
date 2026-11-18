"""Genera el informe PDF y su fuente Markdown usando la evidencia de la entrega."""
import json
from io import BytesIO
from pathlib import Path
import sys
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymupdf
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdfcanvas
from PIL import Image as PILImage
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle,
                               PageBreak, Image, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EVIDENCE = ROOT / "evidencia"
OUT = DOCS / "EVA4_Informe_Haguer_Maldonado.pdf"
TEMPLATE = DOCS / "plantilla-original"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    release = load(EVIDENCE / "liberacion.json")
    ci = EVIDENCE / "ci-liberacion"
    benchmark = load(next(ci.rglob("benchmark-ci.json")))
    docker = load(next(ci.rglob("api-release-ci.json")))
    mlops = load(next(ci.rglob("registro-mlflow.json")))
    batches = load(EVIDENCE / "lotes-cpu.json")
    quant = load(EVIDENCE / "benchmark-int8.json")
    metadata = load(ROOT / "models/metadata.json")
    opt, base = benchmark["optimizado"], benchmark["sin_optimizacion"]
    percent = 100 * (1 - opt["p95_ms"] / base["p95_ms"])
    run_url = release["github_run"]
    figures = DOCS / "figuras"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.1), layout="constrained")
    for offset, name, stats, color in [(-.16, "Sin optimizar", base, "#94a3b8"), (.16, "Optimizado", opt, "#177561")]:
        bars = axes[0].bar([offset, 1+offset], [stats["p50_ms"], stats["p95_ms"]], width=.30, label=name, color=color)
        axes[0].bar_label(bars, fmt="%.2f", padding=3)
    axes[0].set_xticks([0, 1], ["p50", "p95"])
    axes[0].set_ylabel("Milisegundos")
    axes[0].legend(frameon=False)
    axes[0].set_ylim(0, base["p95_ms"] * 1.3)
    matrix = opt["confusion_np_p"]
    axes[1].imshow(matrix, cmap="Greens")
    axes[1].set_xticks([0, 1], ["NP", "P"])
    axes[1].set_yticks([0, 1], ["NP", "P"])
    axes[1].set_xlabel("Clase predicha")
    axes[1].set_ylabel("Clase real")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, matrix[i][j], ha="center", va="center", color="white" if matrix[i][j]>20 else "#172b26", fontsize=15)
    fig.savefig(figures / "latencia-matriz.png", dpi=180);plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.1), layout="constrained")
    history = metadata["history"]
    axes[0].semilogy([x["epoch"] for x in history], [x["val_loss"] for x in history], color="#177561")
    best = history[metadata["best_epoch"]-1]
    axes[0].scatter(best["epoch"], best["val_loss"], color="#b87112", zorder=4, label="Época seleccionada: 33")
    axes[0].set(xlabel="Época", ylabel="Pérdida de validación");axes[0].legend(frameon=False)
    batch_metrics = batches["batch_metrics"]
    axes[1].plot([x["batch_size"] for x in batch_metrics], [x["images_per_second"] for x in batch_metrics], "o-", color="#177561")
    axes[1].set(xlabel="Imágenes por lote", ylabel="Imágenes/s", xticks=[1, 2, 4, 8, 16])
    fig.savefig(figures / "historial-lotes.png", dpi=180);plt.close(fig)

    fonts = Path("C:/Windows/Fonts")
    regular_font = fonts / "arial.ttf" if (fonts / "arial.ttf").exists() else TEMPLATE / "NotoSans-Regular.ttf"
    bold_font = fonts / "arialbd.ttf" if (fonts / "arialbd.ttf").exists() else TEMPLATE / "NotoSans-Bold.ttf"
    pdfmetrics.registerFont(TTFont("Informe", str(regular_font)))
    pdfmetrics.registerFont(TTFont("InformeBold", str(bold_font)))
    pdfmetrics.registerFontFamily("Informe", normal="Informe", bold="InformeBold", italic="Informe", boldItalic="InformeBold")
    normal, bold = "Informe", "InformeBold"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TextoInforme", fontName=normal, fontSize=9.9, leading=13.5, spaceAfter=8, textColor=colors.black))
    styles.add(ParagraphStyle(name="TituloInforme", fontName=bold, fontSize=15.5, leading=21, spaceAfter=12, textColor=colors.HexColor("#c8202d"), keepWithNext=True))
    styles.add(ParagraphStyle(name="SubInforme", fontName=bold, fontSize=11.5, leading=15, spaceBefore=10, spaceAfter=7, textColor=colors.HexColor("#c8202d"), keepWithNext=True))
    styles.add(ParagraphStyle(name="CeldaInforme", fontName=normal, fontSize=8.8, leading=11.5))
    styles.add(ParagraphStyle(name="CabeceraTabla", fontName=bold, fontSize=8.8, leading=11.5, textColor=colors.white))
    styles.add(ParagraphStyle(name="PequenoInforme", fontName=normal, fontSize=8.8, leading=11.7, spaceAfter=6, textColor=colors.HexColor("#52606d")))
    story, markdown = [], []
    width = 526.8

    def title(text):
        story.append(Paragraph(escape(text), styles["TituloInforme"]));markdown.append("# " + text + "\n")
    def sub(text):
        story.append(Paragraph(escape(text), styles["SubInforme"]));markdown.append("## " + text + "\n")
    def p(text, small=False):
        story.append(Paragraph(escape(text), styles["PequenoInforme" if small else "TextoInforme"]));markdown.append(text + "\n")
    def web_link(label, url):
        story.append(Paragraph(f'<link href="{escape(url)}" color="#164d8f"><u>{escape(label)}</u></link><br/>{escape(url)}', styles["PequenoInforme"]))
        markdown.append(f"[{label}]({url})\n")
    def table(headers, rows, widths):
        cells = [[Paragraph(escape(str(x)), styles["CabeceraTabla" if i == 0 else "CeldaInforme"]) for x in row] for i, row in enumerate([headers, *rows])]
        t = Table(cells, colWidths=[width * f for f in widths], hAlign="LEFT", repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#26343f")),
                              ("VALIGN", (0,0), (-1,-1), "TOP"),
                              ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#eff3f5"), colors.white]),
                              ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#dddddd")),
                              ("LEFTPADDING", (0,0), (-1,-1), 5.4), ("RIGHTPADDING", (0,0), (-1,-1), 5.4),
                              ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4)]))
        story.extend([t, Spacer(1, 9)])
        markdown.append("| " + " | ".join(headers) + " |\n| " + " | ".join("---" for x in headers) + " |")
        markdown.extend("| " + " | ".join(map(str,row)) + " |" for row in rows);markdown.append("")
    def page():
        story.append(PageBreak())
    def figure(name, caption):
        story.append(Image(str(figures / name), width=width, height=width * 3.1/11.8))
        p(caption, True)

    # La portada usa el membrete de Carta y la composición del informe de referencia.
    title("Resumen ejecutivo")
    p("Informe técnico defensivo - Evaluación Sumativa N.º 4 de Machine Learning")
    p("Se implementó un servicio local de clasificación de abejas con polen (P) y sin polen (NP), basado en MobileNetV3-Small exportado a ONNX. FastAPI expone inferencia y un panel de demostración. El contenedor se construyó, se probó por HTTP y se liberó mediante un pipeline que conserva evidencia y registra los artefactos en MLflow.")
    table(["Indicador", "Resultado de la evidencia"], [
        ["Prueba reservada", "67/71 aciertos; exactitud 94,37 %; F1 macro 0,9437"],
        ["p95 del runtime en CPU", f"{opt['p95_ms']:.2f} ms; 355 mediciones con tensores precargados"],
        ["p95 HTTP en Docker", f"{docker['http_sequential']['p95_ms']:.2f} ms; 142 solicitudes secuenciales"],
        ["Datos", "714 originales; 571 entrenamiento, 72 validación y 71 prueba"],
        ["Liberación", f"Pipeline aprobado; dos jobs; tag {release['tag']}"],
    ], [.36,.64])
    p("El resultado de prueba describe el corpus disponible. No demuestra generalización a otras colmenas, cámaras o condiciones de luz. El despliegue automatizado de aceptación es temporal; la demostración persistente se ejecuta en el equipo local.", True)
    page()

    title("1. Correspondencia con la pauta")
    table(["Criterio", "Implementación y evidencia"], [
        ["4.1.1 Optimización y recursos", "ONNX optimizado, comparación FP32/INT8, medición de latencia y RAM, lotes y análisis de alternativas GPU. Secciones 3 y 4."],
        ["4.1.2 Plan de acción", "Fases secuenciales con entregables y criterios de salida, desde datos hasta operación y reversión. Sección 5."],
        ["4.1.3 Integración continua", "GitHub Actions, pruebas automáticas de calidad/latencia, MLflow, versionado y despliegue independiente. Sección 6."],
        ["4.1.4 Implementación práctica", "FastAPI y Docker probados por HTTP, contrato de entrada/salida, excepciones y tiempos medidos. Sección 7."],
    ], [.32,.68])
    sub("Arquitectura de inferencia")
    p("JPEG/PNG → validación de tamaño y formato → RGB, redimensionamiento y normalización → sesión ONNX Runtime en CPU → logits y softmax → clase NP/P y respuesta JSON. El mismo servicio entrega el panel HTML y la documentación OpenAPI.")
    sub("Separación de responsabilidades")
    p("train.py ajusta y exporta el modelo; dataset.py controla manifiesto, particiones y huellas; preprocess.py concentra el preprocesamiento; app.py implementa el contrato REST; smoke.py prueba el servicio; benchmark.py y benchmark_lotes.py miden inferencia; validar_docker.py comprueba la imagen y su funcionamiento; registrar_mlflow.py conserva la trazabilidad de la validación.")
    sub("Criterio de aceptación")
    p("La liberación exige integridad de datos y modelo, contrato HTTP correcto, predicciones idénticas al optimizar, F1 macro de prueba >= 0,94 y p95 ONNX <= 25 ms. El umbral de 25 ms pertenece al runtime y no representa una garantía de latencia HTTP bajo concurrencia. Docker y el registro MLflow también deben completar sus verificaciones.")
    page()

    title("2. Datos, ajuste y selección")
    p("El conjunto contiene 714 JPEG originales, sin aumentos precomputados. El manifiesto registra ruta, partición, clase, grupo de origen, nombre original y SHA-256. Se verifican hashes únicos, contenido exacto del ZIP y ausencia de grupos compartidos entre particiones.")
    table(["Partición", "Imágenes", "Grupos de origen", "Propósito"], [
        ["Entrenamiento", 571, 319, "Ajuste de parámetros y calibración INT8"],
        ["Validación", 72, 40, "Selección de época y comparación FP32/INT8"],
        ["Prueba", 71, 40, "Evaluación reservada del modelo elegido"],
    ], [.21,.17,.24,.38])
    p("MobileNetV3-Small parte de pesos ImageNet y se ajusta durante 40 épocas en CPU con AdamW, tasa 0,0003, lote 32, entropía cruzada, semilla 20261117 y seis hilos. El preprocesamiento produce tensores N×3×96×160, RGB y normalización ImageNet.")
    p("La selección maximiza exactitud de validación; ante empate minimiza pérdida y conserva la primera época. Se eligió la época 33: 72/72 aciertos y pérdida 0,02634. La exportación usa opset 17 y eje de lote variable; la paridad PyTorch/ONNX registrada en 72 imágenes tiene error máximo de logits 6,29×10^-5.")
    table(["Clase real", "Predicción NP", "Predicción P", "Recall"], [
        ["NP",34,0,"100,00 %"],["P",4,33,"89,19 %"]
    ], [.28,.23,.23,.26])
    p("Los cuatro errores de prueba son falsos negativos de P. Cada imagen equivale a 1,41 puntos porcentuales de exactitud. Se conserva FP32 por la comparación en validación; no se utiliza el test para elegir la cuantización. Las comprobaciones repetidas en CI son controles de regresión sobre el mismo conjunto, no nuevas estimaciones independientes de generalización.")
    page()

    title("3. Optimización y uso de recursos")
    p(f"Comparación de grafo desactivado y ORT_ENABLE_ALL en {benchmark['cpu']}, ONNX Runtime {benchmark['ort']}, CPU y un hilo intraoperador e interoperador. Cada configuración se ejecuta en un proceso separado, con 20 calentamientos y 355 llamadas medidas a session.run. La carga de imagen, el preprocesamiento y HTTP quedan fuera de esta medición.")
    figure("latencia-matriz.png", "Figura 1. Latencia del runtime y matriz de confusión sobre 71 imágenes. Fuente: benchmark-ci.json de la liberación.")
    table(["Métrica", "Sin optimizar", "Optimizado"], [
        ["p50 / p95 (ms)", f"{base['p50_ms']:.2f} / {base['p95_ms']:.2f}",f"{opt['p50_ms']:.2f} / {opt['p95_ms']:.2f}"],
        ["RSS final (MiB)",f"{base['rss_after_mib']:.2f}",f"{opt['rss_after_mib']:.2f}"],
        ["Carga de sesión (ms)",f"{base['startup_ms']:.2f}",f"{opt['startup_ms']:.2f}"],
    ], [.42,.29,.29])
    p(f"La optimización reduce el p95 un {percent:.1f} % y conserva las 71 clases predichas. El RSS final varía de {base['rss_after_mib']:.2f} a {opt['rss_after_mib']:.2f} MiB. RSS incluye Python, bibliotecas y tensores; es una lectura residente final, no el máximo de memoria. No se utiliza GPU ni se mide VRAM.")
    sub("Cuantización estática INT8: ensayo y decisión")
    table(["Validación: 72 imágenes", "FP32", "INT8 QDQ S8S8"], [
        ["Tamaño del modelo (MB decimales)",f"{quant['fp32']['model_bytes']/1e6:.2f}",f"{quant['int8']['model_bytes']/1e6:.2f}"],
        ["p95 (ms)",f"{quant['fp32']['p95_ms']:.2f}",f"{quant['int8']['p95_ms']:.2f}"],
        ["F1 macro",f"{quant['fp32']['macro_f1']:.4f}",f"{quant['int8']['macro_f1']:.4f}"],
        ["RSS final (MiB)",f"{quant['fp32']['rss_after_mib']:.2f}",f"{quant['int8']['rss_after_mib']:.2f}"],
    ], [.50,.25,.25])
    p(f"Se calibró por canal con {quant['calibration_images']} originales de entrenamiento. El ensayo INT8 reduce almacenamiento y RSS, pero pierde calidad y empeora la latencia en {quant['cpu']}. Se adopta FP32 optimizado. Estos resultados provienen de un ensayo distinto al runner de la figura 1; las comparaciones válidas se realizan dentro de cada ensayo.", True)
    page()

    title("4. Procesamiento por lotes y alternativas")
    figure("historial-lotes.png", "Figura 2. Historial de validación y rendimiento por lotes. Los lotes se midieron en CPU AMD EPYC 9V74; el historial corresponde al ajuste previo del modelo.")
    table(["Lote", "p95 del lote (ms)", "Imágenes/s"], [
        [r["batch_size"], f"{r['p95_full_batch_ms']:.2f}", f"{r['images_per_second']:.1f}"] for r in batch_metrics
    ], [.20,.40,.40])
    p("Se midieron 25 recorridos intercalados de las 71 imágenes, con tensores precargados y lotes fijos de 1, 2, 4, 8 y 16. Los cinco tamaños conservaron las predicciones del lote unitario. El p95 corresponde a lotes completos; el rendimiento incorpora el último lote parcial. El lote 8 alcanzó el mayor rendimiento del ensayo, sin ser una garantía para otro equipo.")
    p(f"El contenedor liberado también procesó los 714 JPEG, incluyendo lectura del ZIP, SHA-256, preprocesamiento, inferencia y CSV: {docker['panel']['full']['processing_seconds']:.2f} s y {docker['panel']['full']['processing_images_per_second']:.1f} imágenes/s. Esta medición completa tiene un alcance distinto al benchmark del runtime.")
    sub("Batching dinámico y TensorRT")
    p("La API implementada recibe una imagen por solicitud. Para aumentar capacidad con solicitudes concurrentes se evaluaría Triton con cola, tamaño máximo de lote y demora máxima de agrupamiento. La ganancia de rendimiento debe contrastarse con la espera añadida al p95 HTTP. Las mediciones de lotes fijos no se presentan como batching dinámico implementado.")
    p("TensorRT FP16/INT8 requiere GPU NVIDIA compatible, perfiles de lote y validación de operadores. Se medirían VRAM, transferencias CPU/GPU, latencia extremo a extremo y paridad con FP32. El pruning exigiría evaluar soporte de esparsidad y posiblemente reajustar la red; reducir parámetros no asegura aceleración en un runtime denso. No se atribuyen resultados de estas alternativas al servicio CPU.")
    page()

    title("5. Plan de acción por fases")
    table(["Fase", "Actividad y entregable", "Condición de salida"], [
        ["1. Datos y ajuste", "Manifiesto, partición por grupos, historial y modelo seleccionado por validación.", "714 originales verificados, grupos sin cruce, selección documentada."],
        ["2. Exportación", "ONNX opset 17, preprocesamiento único y comparación numérica.", "Artefacto válido y paridad PyTorch/ONNX registrada."],
        ["3. Optimización", "Comparar grafo, INT8 y lotes; medir calidad, RAM y latencia.", "Elegir FP32 optimizado y documentar compromisos."],
        ["4. Integración REST", "FastAPI, OpenAPI, panel, límites y manejo de errores.", "Contrato probado, 67/71 y CSV de 714 filas reproducidos."],
        ["5. Empaquetado", "Dockerfile, dependencias fijadas, usuario 10001 y healthcheck.", "Contenedor healthy y pruebas HTTP aprobadas con hashes correctos."],
        ["6. CI/MLOps", "GitHub Actions, quality gate, registro MLflow y artefactos.", "Validación aprobada y registro persistido FINISHED."],
        ["7. Liberación", "Tag, imagen empaquetada, SHA-256 y despliegue en runner independiente.", "Carga y prueba del mismo artefacto liberado aprobadas."],
        ["8. Operación", "Seguimiento de calidad, latencia, errores, RAM y distribución P/NP.", "Investigar degradación y revertir a una imagen previamente validada."],
    ], [.19,.42,.39])
    sub("Gestión de fallos y reversión")
    p("Un fallo de integridad o calidad bloquea la liberación. Un fallo de despliegue exige revisar logs y healthcheck antes de servir solicitudes. Se conserva la imagen anterior con su huella; para revertir se carga, se comprueba /health y una predicción conocida, y se cambia el enrutamiento o puerto hacia esa instancia. El modelo no se modifica dentro de un contenedor en ejecución.")
    sub("Continuidad del monitoreo")
    p("En una operación prolongada se deben instrumentar p95 HTTP, tasa 4xx/5xx, RAM y distribución de predicciones. Las nuevas etiquetas permiten medir calidad por clase. Un cambio de distribución inicia una investigación; no demuestra por sí solo pérdida de exactitud. Este monitoreo continuo es un plan operativo, mientras que las pruebas automatizadas de liberación sí están ejecutadas y archivadas.")
    page()

    title("6. Integración continua y MLOps")
    p("El workflow se ubica en .github/workflows/ci.yml y responde a cambios en main, ramas de validación, pull requests, ejecución manual y tags v*. El job validate instala dependencias, verifica datos y API, mide calidad y lotes, construye y prueba Docker, y registra la validación en MLflow. Cualquier paso fallido detiene la liberación.")
    table(["Control", "Implementación"], [
        ["Versionado", "Git y tag enlazan código, ONNX y datos; SHA-256 identifica modelo y manifiesto; el manifiesto identifica cada JPEG."],
        ["Calidad y velocidad", "F1 macro >= 0,94; p95 ONNX <= 25 ms; 355 mediciones y paridad de clases."],
        ["Contrato", "Estado, inferencia, errores 413/415/422, panel, galería y CSV completo."],
        ["MLflow", "Parámetros, métricas, commit, hashes, modelo ONNX, manifiesto y JSON de validación. SQLite y artefactos archivados."],
        ["Despliegue continuo", "Un tag válido empaqueta la imagen con SHA-256; deploy la descarga, verifica, carga y prueba en otro runner."],
    ], [.27,.73])
    p("El modelo y el conjunto de datos son pequeños y quedan preservados en Git. MLflow agrega trazabilidad por ejecución. No se requiere DVC para esta estrategia; en conjuntos de mayor tamaño se evaluaría almacenamiento externo versionado. El registro actual importa el historial anterior con una etiqueta explícita, sin representar esa importación como un nuevo entrenamiento.")
    sub("Evidencia ejecutada")
    web_link("Pipeline de liberación aprobado", run_url)
    p(f"Commit liberado: {release['git_commit']}", True)
    p(f"Registro MLflow: {mlops['run_id']} | estado: {mlops['status']}", True)
    p("Se conservan evidencia-eva4, evidencia-mlops, eva4-image y evidencia-despliegue-eva4. El despliegue de aceptación es efímero y no equivale a un servicio público permanente. El backend SQLite permite inspeccionar parámetros y métricas; los artefactos conservan también los archivos originales en la descarga.")
    page()

    title("Evidencia visual de la liberación")
    screenshot = figures / "pipeline-v1.2.0-aprobado.jpg"
    image_width, image_height = PILImage.open(screenshot).size
    display_width = 210
    screenshot_image = Image(str(screenshot), width=display_width, height=display_width*image_height/image_width)
    right = [
        Paragraph("Pipeline aprobado", styles["SubInforme"]),
        Paragraph("La captura corresponde a la ejecución real del tag v1.2.0. GitHub muestra Status Success, con los jobs validate y deploy completados correctamente.", styles["TextoInforme"]),
        Paragraph("validate comprueba datos, modelo, calidad, latencia, API, Docker y registro MLflow. deploy carga la imagen liberada y vuelve a probarla por HTTP en un runner independiente.", styles["TextoInforme"]),
        Paragraph("Acceso del docente", styles["SubInforme"]),
    ]
    for label, url in [
        ("1. Repositorio del proyecto", "https://github.com/Rhaguer/HAMBEE"),
        ("2. Versión v1.2.0 y archivos de entrega", "https://github.com/Rhaguer/HAMBEE/releases/tag/v1.2.0"),
        ("3. Pipeline de liberación aprobado", run_url),
    ]:
        right.append(Paragraph(f'<link href="{escape(url)}" color="#164d8f"><u>{escape(label)}</u></link><br/>{escape(url)}', styles["PequenoInforme"]))
        right.append(Spacer(1, 9))
        markdown.append(f"[{label}]({url})\n")
    right.append(Paragraph("Los tres enlaces son clicables en este PDF. El repositorio identifica el código; la versión contiene los archivos de entrega; el pipeline conserva jobs, logs y artefactos verificables.", styles["TextoInforme"]))
    visual = Table([[screenshot_image, right]], colWidths=[226, width-226], hAlign="LEFT")
    visual.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
                               ("LEFTPADDING",(0,0),(0,0),0), ("RIGHTPADDING",(0,0),(0,0),16),
                               ("LEFTPADDING",(1,0),(1,0),0), ("RIGHTPADDING",(1,0),(1,0),0),
                               ("TOPPADDING",(0,0),(-1,-1),0), ("BOTTOMPADDING",(0,0),(-1,-1),0)]))
    story.append(visual)
    p("Figura 3. Captura de GitHub de la ejecución 44284173203. Fuente: página pública de GitHub Actions; captura realizada el 18 de noviembre de 2026.", True)
    markdown.append("![Pipeline aprobado](figuras/pipeline-v1.2.0-aprobado.jpg)\n")
    page()

    title("7. API y contenedor probados")
    table(["Ruta", "Contrato y resultado"], [
        ["GET /health", "200: status ready, versión 1.2.0 y CPUExecutionProvider."],
        ["POST /predict", "Multipart file JPEG/PNG; clase NP/P, confidence, probabilities, model_version e inference_ms."],
        ["GET /docs", "Documentación OpenAPI con entrada, respuesta y ejemplos de error."],
        ["GET /", "Panel de clasificación, evaluación, galería y descarga del CSV."],
        ["Errores", "422 sin file; 415 con contenido inválido; 413 sobre 5 MiB o 12 megapíxeles."],
    ], [.28,.72])
    p("Los puntajes softmax suman aproximadamente uno y describen la salida del clasificador; no se presentan como probabilidades calibradas. inference_ms mide exclusivamente session.run. La latencia HTTP incluye envío local, carga y preprocesamiento, por lo que no debe confundirse con la del runtime.")
    table(["Prueba de liberación", "Resultado"], [
        ["Docker Engine / plataforma", f"{docker['docker_engine']} / {docker['docker_platform']}"],
        ["Usuario / estado", f"UID {docker['docker_container_uid']} / {docker['docker_healthcheck']}"],
        ["HTTP secuencial", f"142 solicitudes; p50 {docker['http_sequential']['p50_ms']:.2f} ms; p95 {docker['http_sequential']['p95_ms']:.2f} ms"],
        ["Panel / CSV", f"{docker['panel']['gallery_images']} imágenes y {docker['panel']['csv_rows']} filas verificadas"],
        ["Prueba del clasificador", "67/71; matriz [[34,0],[4,33]]"],
    ], [.40,.60])
    p("El job deploy ejecuta la imagen recuperada del artefacto de liberación, no una reconstrucción independiente. Valida su huella de archivo, healthcheck, usuario sin privilegios, modelo, datos y respuestas HTTP. api-release-ci.json registra el identificador real de imagen y el enlace al pipeline.")
    p("Las solicitudes HTTP medidas son secuenciales sobre loopback en el runner. La carga concurrente, los fallos de red y el dimensionamiento para clientes externos requieren otra prueba de capacidad. La implementación local y el contenedor probado satisfacen el alcance local permitido por la pauta.")
    page()

    title("8. Reproducción y demostración")
    sub("Ejecución local")
    p("Descomprimir la entrega y abrir EVA4. Con Python 3.12 instalado, ejecutar el archivo Ejecutar HAMBEE.cmd o el comando: py -3.12 ejecutar.py. El lanzador crea .venv, instala dependencias y abre el panel; requiere internet durante la primera instalación.")
    sub("Construcción y prueba Docker")
    p("Desde EVA4, con Docker Engine disponible: docker build -t hambee:1.2.0 . Luego: docker run --rm -p 127.0.0.1:8000:8000 hambee:1.2.0. Abrir http://127.0.0.1:8000. Para la verificación completa instalar requirements-test.txt y ejecutar python validar_docker.py.")
    sub("Pruebas reproducibles")
    table(["Comando desde EVA4", "Verificación"], [
        ["python verificar_dataset.py data/Dataset.zip", "714 originales, hashes y particiones."],
        ["python smoke.py --panel", "API, evaluación, galería y CSV."],
        ["python benchmark.py data/Dataset.zip --output benchmark-ci.json", "Latencia, RAM y calidad del runtime."],
        ["python quality_gate.py benchmark-ci.json", "Umbrales y paridad de clases."],
        ["python benchmark_lotes.py data/Dataset.zip --output lotes-ci.json", "Capacidad por tamaño de lote."],
        ["python registrar_mlflow.py --salida resultados-mlops", "Requiere requirements-mlops.txt y evidencia Docker vigente."],
    ], [.65,.35])
    sub("Secuencia de defensa")
    p("Mostrar /health, clasificar un archivo y abrir /docs. Ejecutar la evaluación reservada y descargar las 714 predicciones, diferenciando los dos alcances. Explicar las cuatro equivocaciones P, la elección de FP32 y los compromisos de RAM/latencia. Mostrar el pipeline aprobado, MLflow y la prueba del contenedor liberado. La guía DEFENSA_EVA4.md contiene preguntas y respuestas de apoyo.")
    p("Colab ofrece una demostración complementaria en CPU utilizando el mismo ONNX. No se necesita volver a entrenar para defender la entrega. El reentrenamiento completo es opcional y puede variar numéricamente entre entornos.")
    page()

    title("9. Trazabilidad y referencias")
    sub("Artefactos de la entrega")
    web_link("Repositorio del proyecto HAMBEE", "https://github.com/Rhaguer/HAMBEE")
    web_link("Versión v1.2.0 y archivos de entrega", f"https://github.com/Rhaguer/HAMBEE/releases/tag/{release['tag']}")
    p("La carpeta evidencia/ci-liberacion conserva los resultados descargados del pipeline. liberacion.json enlaza commit, tag, jobs y estado final. El paquete académico agrega este informe y la guía de defensa al código liberado; no modifica los pesos ni los datos de la imagen validada.")
    p("Modelo SHA-256: " + metadata["model_sha256"], True)
    p("Manifiesto SHA-256: " + metadata["dataset_manifest_sha256"], True)
    p("Imagen Docker ID: " + docker["docker_image_id"], True)
    sub("Fuentes")
    p("[1] INACAP. Evaluación Sumativa N.º 4 de Machine Learning. Unidad 4: Implementación modelo Deep learning. Documento de pauta proporcionado; criterios 4.1.1 a 4.1.4, máximo 24 puntos.")
    p("[2] ONNX Runtime. Graph optimizations. https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html", True)
    p("[3] ONNX Runtime. Quantize ONNX models. https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html", True)
    p("[4] MLflow. ML Experiment Tracking. https://mlflow.org/docs/latest/ml/tracking/", True)
    p("[5] NVIDIA Triton. Dynamic Batcher. https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html", True)
    p("[6] GitHub. Documentación de artefactos de GitHub Actions. https://docs.github.com/en/actions/using-workflows/storing-workflow-data-as-artifacts", True)
    sub("Alcance y conclusión técnica")
    p("La entrega implementa y verifica una solución local de inferencia con API, optimización, empaquetado Docker, integración continua y trazabilidad MLOps. Las evidencias permiten reproducir el funcionamiento y fundamentar la decisión técnica. La evaluación del corpus es limitada y la calificación académica depende de la pauta y de la defensa de los autores.")

    def header(canvas, doc):
        canvas.saveState()
        canvas.drawImage(str(TEMPLATE / "encabezado.png"), 0, 792-123.6, width=618.95, height=116.65, mask="auto")
        canvas.drawImage(str(TEMPLATE / "pie.jpg"), 0, 8.35, width=614.7, height=6)
        canvas.setFillColor(colors.black);canvas.setFont(normal, 9)
        canvas.drawRightString(569.6, 23.3, str(doc.page+1))
        canvas.restoreState()
    buffer = BytesIO()
    doc = BaseDocTemplate(buffer, pagesize=(612, 792), title="HAMBEE - Informe técnico EVA4",
                          author="Ricardo Haguer y Yerko Maldonado")
    frame = Frame(42.6, 50, width, 612, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id="INACAP", frames=[frame], onPage=header))
    doc.build(story)
    cover_buffer = BytesIO()
    cover = pdfcanvas.Canvas(cover_buffer, pagesize=(612, 792))
    cover.drawImage(str(TEMPLATE / "encabezado.png"),0,792-123.6,width=618.95,height=116.65,mask="auto")
    cover.drawImage(str(TEMPLATE / "pie.jpg"),0,8.35,width=614.7,height=6)
    cover.setFillColor(colors.HexColor("#c8202d"));cover.setFont(bold,25)
    cover.drawCentredString(306,612,"HAMBEE")
    cover.setFillColor(colors.black);cover.setFont(normal,15)
    cover.drawCentredString(306,577,"Implementación y optimización de un clasificador de abejas con y sin polen")
    cover.setFillColor(colors.HexColor("#52606d"));cover.setFont(normal,8.8)
    cover.drawString(42.6,546,"Evaluación Sumativa N.º 4 · Machine Learning")
    cover.drawString(42.6,534,"Unidad 4 · Implementación de modelos Deep Learning")
    cover.drawImage(str(TEMPLATE / "logo-hambee.png"),181,792-496.2,width=250,height=234.35,mask="auto")
    cover.setFillColor(colors.black);cover.setFont(normal,9.9)
    for index, line in enumerate([
        "Carrera: Ingeniería en Informática", "Sección: TIEL26 · 101-8V", "Asignatura: Machine Learning",
        "Autores: Ricardo Haguer y Yerko Maldonado", "Docente: Francisco Valdivieso Tapia",
        "Sede: INACAP Puente Alto", "Fecha de esta versión: 18 de noviembre de 2026",
    ]):
        cover.drawString(42.6,792-(632+18*index),line)
    cover.setFont(normal,9);cover.drawRightString(569.6,23.3,"1")
    cover.showPage();cover.save()
    final = pymupdf.open(stream=cover_buffer.getvalue(),filetype="pdf")
    body = pymupdf.open(stream=buffer.getvalue(), filetype="pdf")
    final.insert_pdf(body)
    final.set_metadata({"title": "HAMBEE - Informe técnico EVA4", "author": "Ricardo Haguer y Yerko Maldonado"})
    final.save(OUT, garbage=4, deflate=True)
    portada = "# HAMBEE\n\nImplementación y optimización de un clasificador de abejas con y sin polen\n\nEvaluación Sumativa N.º 4 · Machine Learning\n\nUnidad 4 · Implementación de modelos Deep Learning\n\nCarrera: Ingeniería en Informática\n\nSección: TIEL26 · 101-8V\n\nAsignatura: Machine Learning\n\nAutores: Ricardo Haguer y Yerko Maldonado\n\nDocente: Francisco Valdivieso Tapia\n\nSede: INACAP Puente Alto\n\nFecha de esta versión: 18 de noviembre de 2026\n\n"
    (DOCS / "EVA4_Informe_Haguer_Maldonado.md").write_text(portada+"\n".join(markdown), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
