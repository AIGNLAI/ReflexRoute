const state = {
  config: null,
  mode: "zero-shot",
  busy: false,
};

const elements = {
  body: document.body,
  liveStatus: document.querySelector("#liveStatus"),
  query: document.querySelector("#queryInput"),
  charCount: document.querySelector("#charCount"),
  resetQuery: document.querySelector("#resetQuery"),
  routeButton: document.querySelector("#routeButton"),
  compareButton: document.querySelector("#compareButton"),
  replayButton: document.querySelector("#replayButton"),
  modeButtons: [...document.querySelectorAll(".mode-button")],
  resultGrid: document.querySelector("#resultGrid"),
  emptyState: document.querySelector("#emptyState"),
  resultTemplate: document.querySelector("#resultTemplate"),
  runKind: document.querySelector("#runKind"),
  notice: document.querySelector("#notice"),
  candidateGrid: document.querySelector("#candidateGrid"),
  candidateCount: document.querySelector("#candidateCount"),
  evidenceList: document.querySelector("#evidenceList"),
  evidenceLabel: document.querySelector("#evidenceLabel"),
  benchmarkBars: document.querySelector("#benchmarkBars"),
  benchmarkCount: document.querySelector("#benchmarkCount"),
  benchmarkCost: document.querySelector("#benchmarkCost"),
};

async function requestJSON(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({ error: "Invalid server response" }));
  if (!response.ok) {
    throw new Error(payload.error || `Request failed with status ${response.status}`);
  }
  return payload;
}

function shortLabel(model) {
  const configured = state.config?.models.find((candidate) => candidate.id === model);
  return configured?.label || model.split("/").pop();
}

function formatCost(value) {
  return typeof value === "number" ? `$${value.toFixed(6)}` : "n/a";
}

function updateCharacterCount() {
  elements.charCount.textContent = `${elements.query.value.length.toLocaleString()} characters`;
}

function setNotice(message = "", kind = "info") {
  elements.notice.textContent = message;
  elements.notice.classList.toggle("hidden", !message);
  elements.notice.classList.toggle("error", kind === "error");
}

function setBusy(busy, label = "Routing…") {
  state.busy = busy;
  elements.body.classList.toggle("busy", busy);
  elements.routeButton.querySelector("span").textContent = busy ? label : "Route live";
}

function setMode(mode) {
  state.mode = mode;
  for (const button of elements.modeButtons) {
    button.classList.toggle("active", button.dataset.mode === mode);
  }
}

function renderCandidates(models, selectedModels = []) {
  elements.candidateGrid.replaceChildren();
  elements.candidateCount.textContent = `${models.length} MODELS`;
  for (const model of models) {
    const card = document.createElement("article");
    card.className = "candidate-card";
    card.dataset.model = model.id;
    card.classList.toggle("selected", selectedModels.includes(model.id));

    const provider = document.createElement("span");
    provider.className = "candidate-provider";
    provider.textContent = model.id.split("/")[0].slice(0, 2).toUpperCase();

    const title = document.createElement("h3");
    title.textContent = model.label;

    const description = document.createElement("p");
    description.textContent = model.description;

    const tags = document.createElement("div");
    tags.className = "candidate-tags";
    for (const tag of model.strengths.slice(0, 2)) {
      const chip = document.createElement("span");
      chip.textContent = tag;
      tags.append(chip);
    }

    card.append(provider, title, description, tags);
    elements.candidateGrid.append(card);
  }
}

function renderResult(result) {
  const node = elements.resultTemplate.content.firstElementChild.cloneNode(true);
  node.dataset.mode = result.mode;
  node.querySelector(".result-mode").textContent =
    result.mode === "few-shot" ? "Few-shot · priors + history" : "Zero-shot · built-in priors";
  node.querySelector(".selected-model").textContent = result.selected_label;

  const confidence = Math.max(0, Math.min(1, Number(result.confidence) || 0));
  node.querySelector(".confidence-value").textContent = `${Math.round(confidence * 100)}%`;
  node.querySelector(".confidence-ring").style.setProperty(
    "--confidence-angle",
    `${confidence * 360}deg`,
  );

  const probabilityList = node.querySelector(".probability-list");
  const probabilities = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
  for (const [model, probability] of probabilities) {
    const row = document.createElement("div");
    row.className = "probability-row";
    row.classList.toggle("selected", model === result.selected_model);

    const label = document.createElement("span");
    label.className = "probability-label";
    label.textContent = shortLabel(model);

    const track = document.createElement("div");
    track.className = "probability-track";
    const fill = document.createElement("div");
    fill.className = "probability-fill";
    fill.style.setProperty("--probability-width", `${Math.max(0, probability) * 100}%`);
    track.append(fill);

    const number = document.createElement("span");
    number.className = "probability-number";
    number.textContent = `${Math.round(probability * 100)}%`;
    row.append(label, track, number);
    probabilityList.append(row);
  }

  node.querySelector(".latency-value").textContent = `${Math.round(result.latency_ms)} ms`;
  node.querySelector(".cost-value").textContent = formatCost(result.route_cost_usd);
  node.querySelector(".evidence-value").textContent = `${result.evidence.length} rows`;
  return node;
}

function renderResults(results, recorded = false) {
  elements.emptyState.classList.add("hidden");
  elements.resultGrid.replaceChildren(...results.map(renderResult));
  elements.resultGrid.classList.toggle("comparison", results.length > 1);
  elements.runKind.textContent = recorded ? "RECORDED" : "LIVE";
  const selected = [...new Set(results.map((result) => result.selected_model))];
  renderCandidates(state.config.models, selected);

  const fewShot = results.find((result) => result.mode === "few-shot");
  renderEvidence(fewShot?.evidence || []);
}

function renderEvidence(evidence) {
  elements.evidenceList.replaceChildren();
  elements.evidenceLabel.textContent = evidence.length ? `${evidence.length} RETRIEVED` : "FEW-SHOT ONLY";
  if (!evidence.length) {
    const message = document.createElement("p");
    message.className = "muted-copy";
    message.textContent = "Zero-shot decisions use model priors only.";
    elements.evidenceList.append(message);
    return;
  }

  for (const item of evidence.slice(0, 6)) {
    const row = document.createElement("div");
    row.className = "evidence-item";

    const model = document.createElement("span");
    model.className = "evidence-model";
    model.textContent = item.model_label;

    const query = document.createElement("span");
    query.className = "evidence-query";
    query.textContent = item.query;
    query.title = item.query;

    const score = document.createElement("div");
    score.className = "evidence-score";
    const performance = document.createElement("strong");
    performance.textContent = Number(item.performance).toFixed(2);
    const similarity = document.createElement("span");
    similarity.textContent = `${Math.round(item.similarity * 100)}% similar`;
    score.append(performance, similarity);

    row.append(model, query, score);
    elements.evidenceList.append(row);
  }
}

function renderBenchmark(payload) {
  elements.benchmarkBars.replaceChildren();
  elements.benchmarkCount.textContent = `${payload.target_count} QUERIES`;
  const modes = [
    ["Zero-shot", payload.zero_shot, "zero-shot"],
    ["Few-shot K=4", payload.few_shot, "few-shot"],
  ];
  for (const [label, metrics, className] of modes) {
    const row = document.createElement("div");
    row.className = `benchmark-row ${className}`;
    const top = document.createElement("div");
    top.className = "benchmark-row-top";
    const name = document.createElement("span");
    name.textContent = label;
    const value = document.createElement("strong");
    value.textContent = Number(metrics.quality_mean).toFixed(4);
    top.append(name, value);
    const track = document.createElement("div");
    track.className = "benchmark-track";
    const fill = document.createElement("div");
    fill.className = "benchmark-fill";
    fill.style.setProperty("--benchmark-width", `${metrics.quality_mean * 100}%`);
    track.append(fill);
    row.append(top, track);
    elements.benchmarkBars.append(row);
  }
  elements.benchmarkCost.textContent =
    `$${payload.zero_shot.router_cost_per_1000_usd.toFixed(3)} → ` +
    `$${payload.few_shot.router_cost_per_1000_usd.toFixed(3)}`;
}

async function routeLive(mode) {
  return requestJSON("/api/route", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: elements.query.value, mode }),
  });
}

async function runSingle() {
  if (state.busy || !state.config.live_available) return;
  setNotice();
  setBusy(true);
  try {
    const result = await routeLive(state.mode);
    renderResults([result]);
  } catch (error) {
    setNotice(error.message, "error");
  } finally {
    setBusy(false);
  }
}

async function compareModes() {
  if (state.busy || !state.config.live_available) return;
  setNotice("Running zero-shot, then few-shot. This makes two Jev decisions.");
  setBusy(true, "Comparing…");
  try {
    const zeroShot = await routeLive("zero-shot");
    const fewShot = await routeLive("few-shot");
    renderResults([zeroShot, fewShot]);
    setNotice("Same query, same candidates—only the historical evidence changed.");
  } catch (error) {
    setNotice(error.message, "error");
  } finally {
    setBusy(false);
  }
}

async function replayDemo() {
  if (state.busy) return;
  setNotice();
  setBusy(true, "Loading…");
  try {
    const replay = await requestJSON("/api/replay");
    elements.query.value = replay.query;
    updateCharacterCount();
    renderResults(replay.results, true);
    setNotice("Recorded replay: user evidence changes the selected route.");
  } catch (error) {
    setNotice(error.message, "error");
  } finally {
    setBusy(false);
  }
}

async function initialize() {
  try {
    const [config, benchmark] = await Promise.all([
      requestJSON("/api/config"),
      requestJSON("/api/benchmark"),
    ]);
    state.config = config;
    elements.query.value = config.default_query;
    updateCharacterCount();
    renderCandidates(config.models);
    renderBenchmark(benchmark);

    elements.liveStatus.classList.add(config.live_available ? "live" : "replay");
    elements.liveStatus.querySelector("span:last-child").textContent = config.live_available
      ? "Live routing ready"
      : "Replay mode";
    elements.routeButton.disabled = !config.live_available;
    elements.compareButton.disabled = !config.live_available;
    if (!config.live_available) {
      elements.routeButton.title = "Set OPENROUTER_API_KEY on the server to enable live routing";
      elements.compareButton.title = elements.routeButton.title;
    }
    await replayDemo();
  } catch (error) {
    setNotice(`Playground failed to initialize: ${error.message}`, "error");
  }
}

for (const button of elements.modeButtons) {
  button.addEventListener("click", () => setMode(button.dataset.mode));
}
elements.query.addEventListener("input", updateCharacterCount);
elements.resetQuery.addEventListener("click", () => {
  elements.query.value = state.config.default_query;
  updateCharacterCount();
});
elements.routeButton.addEventListener("click", runSingle);
elements.compareButton.addEventListener("click", compareModes);
elements.replayButton.addEventListener("click", replayDemo);

initialize();
