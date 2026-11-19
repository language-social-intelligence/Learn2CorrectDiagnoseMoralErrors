"use strict";
const $ = id => document.getElementById(id);
let info = null, active = false, latest = null;
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

async function api(path, options) {
  const response = await fetch(path, {...options, signal: AbortSignal.timeout(15000)});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || "The server could not complete the request.");
  return value;
}

function values() {
  return Object.fromEntries(info.fields.map(field => [field.name, $("field-" + field.name).value]));
}

function count() {
  const length = Object.values(values()).reduce((sum, text) => sum + text.length, 0);
  $("count").textContent = length.toLocaleString() + " / 4,000";
}

function showMessage(text, error = false) {
  $("message").textContent = text;
  $("message").className = "message" + (error ? " error" : "");
}

function renderInfo(data) {
  $("title").textContent = data.title;
  document.title = data.title + " · Light model";
  $("description").textContent = data.description;
  $("model").textContent = data.model_label;
  for (const field of data.fields) {
    const label = document.createElement("label");
    label.htmlFor = "field-" + field.name;
    label.textContent = field.label;
    const textarea = document.createElement("textarea");
    textarea.id = label.htmlFor;
    textarea.name = field.name;
    textarea.placeholder = field.placeholder;
    textarea.maxLength = 4000;
    textarea.required = true;
    textarea.addEventListener("input", count);
    $("fields").append(label, textarea);
  }
  for (const example of data.examples) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = example.label;
    button.addEventListener("click", () => {
      if (active) return;
      for (const [name, value] of Object.entries(example.values)) $("field-" + name).value = value;
      count();
      $("field-" + data.fields[0].name).focus();
    });
    $("examples").append(button);
  }
}

async function refresh() {
  try {
    const data = await api("/api/info");
    if (!info) { info = data; renderInfo(data); }
    info = data;
    const label = data.status === "ready" ? "Model online" : data.status === "loading" ? "Loading model" : "Model unavailable";
    $("connection").replaceChildren(Object.assign(document.createElement("i"), {ariaHidden: "true"}), document.createTextNode(label));
    $("connection").className = data.status;
    $("run").disabled = active || data.status !== "ready";
  } catch {
    $("connection").textContent = "Connection unavailable";
    $("connection").className = "error";
    $("run").disabled = true;
  }
}

$("form").addEventListener("submit", async event => {
  event.preventDefault();
  if (active || !info || info.status !== "ready") return;
  const input = values();
  if (Object.values(input).reduce((sum, text) => sum + text.length, 0) > 4000) {
    showMessage("Please keep the input under 4,000 characters.", true); return;
  }
  active = true;
  latest = null;
  $("run").disabled = true;
  $("run").textContent = "Running model…";
  $("output-panel").setAttribute("aria-busy", "true");
  $("result").hidden = true;
  $("empty").hidden = false;
  $("verdict").hidden = true;
  showMessage("Submitting your text…");
  try {
    const job = await api("/api/generate", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(input)});
    const deadline = Date.now() + 8 * 60 * 1000;
    let result;
    while (Date.now() < deadline) {
      const state = await api("/api/results/" + encodeURIComponent(job.id));
      if (state.status === "error") throw new Error(state.error);
      if (state.status === "complete") {result = state.result; break;}
      showMessage(state.status === "queued" ? "Your request is queued. Please keep this page open." : "The model is generating its response…");
      await pause(1500);
    }
    if (!result) throw new Error("This request took too long. Please try again.");
    latest = {model: info.model_label, input, ...result, created_at: new Date().toISOString()};
    $("output").textContent = result.output || "The model returned an empty response.";
    $("truncated").hidden = !result.truncated;
    $("timing").textContent = result.seconds + " seconds · " + result.tokens + " tokens";
    $("verdict").textContent = result.verdict || "";
    $("verdict").hidden = !result.verdict;
    $("empty").hidden = true;
    $("result").hidden = false;
    showMessage("Response complete.");
  } catch (error) {
    showMessage(error.name === "TimeoutError" ? "The connection timed out. Please try again." : error.message, true);
  } finally {
    active = false;
    $("output-panel").setAttribute("aria-busy", "false");
    $("run").textContent = "Run model ↗";
    await refresh();
  }
});

$("download").addEventListener("click", () => {
  if (!latest) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(latest, null, 2)], {type: "application/json"}));
  const link = document.createElement("a");
  link.href = url; link.download = "light-model-result.json"; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
refresh();
setInterval(refresh, 10000);
