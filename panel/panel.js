"use strict";
const $ = (id) => document.getElementById(id);
const number = (value, digits = 0) => new Intl.NumberFormat("es-CL", {minimumFractionDigits: digits, maximumFractionDigits: digits}).format(value);
let evidence;
let galleryPage = 1, listRequest = 0;
function text(id, value) { $(id).textContent = value; }
function element(tag, value, className) {
  const node = document.createElement(tag);
  if (value) node.textContent = value;
  if (className) node.className = className;
  return node;
}
async function request(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "No se pudo completar la solicitud.");
  return data;
}
function status(message, error = false) {
  text("status", message);
  $("status").toggleAttribute("data-error", error);
}
function show(test, full) {
  text("accuracy", number(100 * test.accuracy, 2) + " %");
  text("test-count", test.correct + " de " + test.images + " imágenes · F1 macro " + number(test.macro_f1, 4));
  $("matrix").replaceChildren(...test.confusion_np_p.map((row, i) => {
    const tr = element("tr");
    const label = element("th", ["NP", "P"][i]); label.scope = "row"; tr.append(label);
    row.forEach((count, j) => tr.append(element("td", String(count), i === j ? "good" : count ? "bad" : "")));
    return tr;
  }));
  text("errors", test.errors ? (test.errors.length ? test.errors.map((r) => r.image + " · " + r.actual + " → " + r.predicted).join(", ") : "Sin errores en esta prueba.") : "");
  if (!full) return;
  text("image-count", number(full.processed));
  text("full-time", "Lotes de " + full.batch_size + " · " + number(full.processing_seconds, 2) + " s");
  text("throughput", number(full.processing_images_per_second, 1));
  $("partitions").replaceChildren(...[["train", "Entrenamiento"], ["val", "Validación"], ["test", "Prueba"]].map(([key, label]) => {
    const line = element("div");
    line.append(element("dt", label), element("dd", number(full.partitions[key])));
    return line;
  }));
}
function drawBars(id, data, digits, unit) {
  const maximum = Math.max(...data.map((item) => item[1]));
  $(id).replaceChildren(...data.map(([label, value, reference]) => {
    const row = element("div", "", "bar");
    row.toggleAttribute("data-reference", Boolean(reference));
    const track = element("div", "", "track");
    const fill = element("i"); fill.style.width = value / maximum * 100 + "%"; track.append(fill);
    row.append(element("span", label), track, element("strong", number(value, digits) + unit));
    return row;
  }));
}
function drawHistory() {
  if (!evidence) return;
  const canvas = $("history"), ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  canvas.width = width * ratio; canvas.height = height * ratio;
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio);
  const history = evidence.model.history, selected = evidence.model.best_epoch;
  const xmin = 49, xmax = width - 17, ymin = 18, ymax = height - 31;
  const x = (epoch) => xmin + (epoch - 1) / (history.length - 1) * (xmax - xmin);
  const logs = history.map((item) => Math.log10(item.val_loss));
  const low = Math.floor(Math.min(...logs)), high = Math.ceil(Math.max(...logs));
  const y = (loss) => ymin + (high - Math.log10(loss)) / (high - low || 1) * (ymax - ymin);
  ctx.font = "12px system-ui"; ctx.fillStyle = "#b0c0b8"; ctx.strokeStyle = "#33433d";
  for (let exponent = low; exponent <= high; exponent++) {
    const value = 10 ** exponent, at = y(value);
    ctx.beginPath(); ctx.moveTo(xmin, at); ctx.lineTo(xmax, at); ctx.stroke();
    ctx.fillText(number(value, Math.max(0, -exponent)), 2, at + 4);
  }
  for (const epoch of [1, 10, 20, 30, 40]) ctx.fillText(String(epoch), x(epoch) - 5, height - 11);
  ctx.beginPath(); ctx.strokeStyle = "#e7bc68"; ctx.lineWidth = 2;
  history.forEach((item, i) => ctx[i ? "lineTo" : "moveTo"](x(item.epoch), y(item.val_loss)));
  ctx.stroke();
  const best = history.find((item) => item.epoch === selected);
  ctx.beginPath(); ctx.fillStyle = "#9fd4b8"; ctx.arc(x(best.epoch), y(best.val_loss), 5.5, 0, 2 * Math.PI); ctx.fill();
}
async function run() {
  const button = $("run");
  button.disabled = true; $("csv").hidden = true;
  status("Evaluando las 71 imágenes de prueba y procesando los lotes disponibles…");
  try {
    const result = await request("/demo/evaluate", {method: "POST"});
    show(result.test, result.full);
    $("partition").value = "test";
    $("search").value = "";
    $("gallery").hidden = false;
    await listImages(1);
    $("csv").hidden = false;
    status("Evaluación completada en esta sesión. El CSV ya está disponible.");
  } catch (error) { status(error.message, true); }
  finally { button.disabled = false; }
}
async function listImages(page) {
  const ticket = ++listRequest;
  const params = new URLSearchParams({partition: $("partition").value, search: $("search").value, page});
  try {
    const result = await request("/demo/images?" + params);
    if (ticket !== listRequest) return;
    galleryPage = result.page;
    $("images").replaceChildren(...result.images.map((row) => {
      const path = row.ruta.split("/").pop();
      const card = element("article", "", "card specimen");
      const preview = element("img"); preview.src = "/demo/images/" + row.id; preview.alt = path; preview.loading = "lazy";
      const predicted = row.clase_predicha === "P" ? "Con polen (P)" : "Sin polen (NP)";
      card.append(preview, element("strong", predicted), element("small", path),
        element("small", "Etiqueta real: " + row.clase_real));
      if (row.puntaje_p) {
        const score = 100 * Number(row.clase_predicha === "P" ? row.puntaje_p : row.puntaje_np);
        card.append(element("small", "Puntaje: " + number(score, 2) + " %"));
      }
      return card;
    }));
    text("pages", number(result.total) + (result.total === 1 ? " imagen" : " imágenes") +
      " · página " + result.page + " de " + result.pages);
    $("prev").disabled = result.page <= 1;
    $("next").disabled = result.page >= result.pages;
  } catch (error) {
    if (ticket === listRequest) text("pages", error.message);
  }
}
$("partition").addEventListener("change", () => listImages(1));
$("search").addEventListener("input", () => listImages(1));
$("prev").addEventListener("click", () => listImages(galleryPage - 1));
$("next").addEventListener("click", () => listImages(galleryPage + 1));
$("run").addEventListener("click", run);
$("upload").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button");
  const body = new FormData(); body.set("file", $("image").files[0]);
  button.disabled = true;
  text("prediction", "Clasificando imagen…");
  try {
    const result = await request("/predict", {method: "POST", body});
    $("prediction").replaceChildren(element("strong", result.label === "P" ? "Con polen (P)" : "Sin polen (NP)"),
      element("span", "P " + number(result.probabilities.P * 100, 2) + " % · NP " + number(result.probabilities.NP * 100, 2) + " % · " + number(result.inference_ms, 2) + " ms"));
  } catch (error) { text("prediction", error.message); }
  finally { button.disabled = false; }
});
window.addEventListener("resize", drawHistory);
(async () => {
  try {
    const [health, stored] = await Promise.all([request("/health"), request("/demo/evidence")]);
    evidence = stored;
    text("health", "Servicio disponible · Modelo " + health.model_version + " · ONNX Runtime CPU");
    text("epoch", stored.model.best_epoch + " / " + stored.model.history.length);
    text("validation", number(stored.model.history.find((row) => row.epoch === stored.model.best_epoch).val_accuracy * 100, 2) + " % en validación");
    const ref = stored.benchmark.sin_optimizacion, opt = stored.benchmark.optimizado;
    show({...opt, correct: stored.full.test.correct, images: stored.full.test.images}, stored.full);
    drawBars("latency", [["p50 · base", ref.p50_ms, true], ["p50 · optimizado", opt.p50_ms],
      ["p95 · base", ref.p95_ms, true], ["p95 · optimizado", opt.p95_ms]], 2, " ms");
    drawBars("batches", stored.batches.batch_metrics.map((row) => [row.batch_size + " por lote", row.images_per_second]), 0, "");
    text("history-caption", "40 épocas · Escala logarítmica · Punto verde: época " + stored.model.best_epoch);
    drawHistory();
    $("run").disabled = !stored.dataset_available;
    status(stored.dataset_available ? "714 originales disponibles para analizar." : "Configura el dataset para ejecutar la evaluación.");
    if (stored.dataset_available && new URLSearchParams(location.search).has("ejecutar")) await run();
  } catch (error) { status("No se pudo cargar el panel. " + error.message, true); }
})();
