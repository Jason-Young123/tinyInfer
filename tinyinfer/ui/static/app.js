const USERS = ["user1", "user2", "user3", "user4", "user5"];
const PERFORMANCE_METRICS = [
  ["queue_time", "Queue", "latency"],
  ["prefill_time", "Prefill", "latency"],
  ["decode_time", "Decode", "latency"],
  ["ttft", "TTFT", "latency"],
  ["service_time", "Service", "latency"],
  ["e2e_latency", "E2E", "latency"],
  ["tpot", "TPOT", "latency"],
  ["decode_throughput", "Decode tok/s", "throughput"],
  ["request_throughput", "Request tok/s", "throughput"],
];

const states = new Map();
const usersRoot = document.querySelector("#users");
const startAll = document.querySelector("#start-all");
const blockGrid = document.querySelector("#block-grid");
const performanceGrid = document.querySelector("#performance-grid");
const sdStatistics = document.querySelector("#sd-statistics");
const sdControls = document.querySelector("#sd-controls");
const sdMaxDraft = document.querySelector("#sd-max-draft");
const sdMaxDraftValue = document.querySelector("#sd-max-draft-value");
const sdControlStatus = document.querySelector("#sd-control-status");
const sdAcceptanceRate = document.querySelector("#sd-acceptance-rate");
const sdAvgAccepted = document.querySelector("#sd-avg-accepted");
const sdTrendChart = document.querySelector("#sd-trend-chart");
const sdTrendWrap = document.querySelector("#sd-chart-viewport");
const sdTrendEmpty = document.querySelector("#sd-trend-empty");
const sdTrendCount = document.querySelector("#sd-trend-count");
const globalController = document.querySelector("#global-controller");
const ws = new WebSocket(`ws://${location.host}/ws`);

const SVG_NS = "http://www.w3.org/2000/svg";
const SD_TREND_VISIBLE_POINTS = 100;
let previousSdCounters = null;
let sdTrendSamples = [];

let blockNodes = [];
let activeBlocks = new Set();
let cachedBlocks = new Set();
let sdEnabled = false;
let confirmedMaxDraftTokens = null;

function makePanel(userId, index) {
  const panel = document.createElement("section");
  panel.className = "panel user-panel";
  panel.innerHTML = `
    <div class="panel-head">
      <div class="panel-title">User ${index + 1}</div>
      <div class="status idle">Idle</div>
    </div>
    <div class="output"></div>
    <div class="user-lower">
      <div class="composer">
        <textarea placeholder="Type a prompt..."></textarea>
        <button class="primary start-button" disabled>Start</button>
      </div>
      <div class="user-tools">
        <section class="mini-panel request-controls">
          <div class="mini-title">Controller</div>
          <div class="control-row">
            <span>max tokens</span>
            <input class="max-tokens" type="range" min="32" max="8192" step="32" value="8192">
            <span class="control-value max-value">8192</span>
          </div>
          <div class="control-row">
            <span>temperature</span>
            <input class="temperature" type="range" min="0.1" max="2.0" step="0.1" value="0.8">
            <span class="control-value temp-value">0.8</span>
          </div>
          <div class="toggle-row">
            <span>greedy</span>
            <label class="switch">
              <input class="greedy" type="checkbox">
              <span class="slider-toggle"></span>
            </label>
          </div>
        </section>
        <section class="mini-panel user-statistics">
          <div class="mini-title">Last request</div>
          <div class="user-stat-grid">
            <div class="user-stat-row single">
              <div class="user-stat"><span>TTFT</span><strong class="stat-ttft">--</strong></div>
            </div>
            <div class="user-stat-row single">
              <div class="user-stat"><span>Decode tok/s</span><strong class="stat-decode-throughput">--</strong></div>
            </div>
            <div class="user-stat-row pair">
              <div class="user-stat"><span>TPOT mean</span><strong class="stat-tpot">--</strong></div>
              <div class="user-stat"><span>TPOT p50</span><strong class="stat-tpot-p50">--</strong></div>
            </div>
            <div class="user-stat-row pair">
              <div class="user-stat"><span>TPOT p95</span><strong class="stat-tpot-p95">--</strong></div>
              <div class="user-stat"><span>TPOT p99</span><strong class="stat-tpot-p99">--</strong></div>
            </div>
            <div class="user-stat-row single">
              <div class="user-stat"><span>E2E</span><strong class="stat-e2e">--</strong></div>
            </div>
            <div class="user-stat-row single"> <!-- # [MOD] -->
              <div class="user-stat"><span>Cache Hit Rate</span><strong class="stat-cache-hit-rate">--</strong></div> <!-- # [MOD] -->
            </div> <!-- # [MOD] -->
          </div>
        </section>
      </div>
    </div>`;

  const state = {
    userId,
    panel,
    input: panel.querySelector("textarea"),
    button: panel.querySelector(".start-button"),
    output: panel.querySelector(".output"),
    status: panel.querySelector(".status"),
    maxTokens: panel.querySelector(".max-tokens"),
    maxValue: panel.querySelector(".max-value"),
    temperature: panel.querySelector(".temperature"),
    tempValue: panel.querySelector(".temp-value"),
    greedy: panel.querySelector(".greedy"),
    statTTFT: panel.querySelector(".stat-ttft"),
    statDecodeThroughput: panel.querySelector(".stat-decode-throughput"),
    statTPOT: panel.querySelector(".stat-tpot"),
    statTPOTP50: panel.querySelector(".stat-tpot-p50"), // [MOD]
    statTPOTP95: panel.querySelector(".stat-tpot-p95"), // [MOD]
    statTPOTP99: panel.querySelector(".stat-tpot-p99"), // [MOD]
    statE2E: panel.querySelector(".stat-e2e"),
    statCacheHitRate: panel.querySelector(".stat-cache-hit-rate"), // # [MOD]
    busy: false,
    outputBase: "",
    followOutput: true,
  };
  states.set(userId, state);

  // Only auto-follow while the user is already viewing the bottom of this panel.
  state.output.addEventListener("scroll", () => {
    state.followOutput = isNearBottom(state.output);
  }, {passive: true});

  state.input.addEventListener("input", refreshControls);
  state.button.addEventListener("click", () => submitOne(state));
  state.maxTokens.addEventListener("input", () => {
    state.maxValue.textContent = state.maxTokens.value;
  });
  state.temperature.addEventListener("input", () => {
    state.tempValue.textContent = Number(state.temperature.value).toFixed(1);
  });

  usersRoot.appendChild(panel);
}

function ready(state) {
  return !state.busy && state.input.value.trim().length > 0;
}

function setBusy(state, busy) {
  state.busy = busy;
  state.input.readOnly = busy;
  state.maxTokens.disabled = busy;
  state.temperature.disabled = busy;
  state.greedy.disabled = busy;
  state.status.textContent = busy ? "Running" : "Idle";
  state.status.classList.toggle("busy", busy);
  state.status.classList.toggle("idle", !busy);
  refreshControls();
}

function refreshControls() {
  for (const state of states.values()) state.button.disabled = !ready(state) || ws.readyState !== WebSocket.OPEN;
  startAll.disabled = ws.readyState !== WebSocket.OPEN || ![...states.values()].some(ready);

  const allUsersIdle = [...states.values()].every(state => !state.busy);
  sdMaxDraft.disabled = !sdEnabled || ws.readyState !== WebSocket.OPEN || !allUsersIdle;
  if (sdEnabled) sdControlStatus.textContent = allUsersIdle
    ? "Global K · editable while all users are idle"
    : "Global K · locked while requests are running";
}

function requestPayload(state) {
  return {
    user_id: state.userId,
    text: state.input.value,
    max_tokens: Number(state.maxTokens.value),
    temperature: Number(state.temperature.value),
    greedy: state.greedy.checked,
  };
}

function isNearBottom(element, threshold = 24) {
  return element.scrollHeight - element.clientHeight - element.scrollTop <= threshold;
}

function updateOutput(state, text) {
  state.output.textContent = text;

  // Preserve the user's reading position when they have scrolled into history.
  if (state.followOutput) {
    state.output.scrollTop = state.output.scrollHeight;
  }
}

function beginLocal(state) {
  state.outputBase = state.output.textContent;
  setBusy(state, true);
}

function submitOne(state) {
  if (!ready(state) || ws.readyState !== WebSocket.OPEN) return;
  const payload = requestPayload(state);
  beginLocal(state);
  ws.send(JSON.stringify({type: "submit", ...payload}));
}

startAll.addEventListener("click", () => {
  if (ws.readyState !== WebSocket.OPEN) return;
  const requests = [];
  for (const state of states.values()) {
    if (!ready(state)) continue;
    requests.push(requestPayload(state));
    beginLocal(state);
  }
  if (requests.length) {
    ws.send(JSON.stringify({type: "submit_many", requests}));
  }
});

sdMaxDraft.addEventListener("input", () => {
  sdMaxDraftValue.textContent = sdMaxDraft.value;
});

sdMaxDraft.addEventListener("change", () => {
  if (sdMaxDraft.disabled || ws.readyState !== WebSocket.OPEN) return;
  const value = Number(sdMaxDraft.value);
  if (value === confirmedMaxDraftTokens) return;
  sdControlStatus.textContent = "Updating global K...";
  ws.send(JSON.stringify({type: "configure_sd", max_draft_tokens: value}));
});

function formatLatency(value) {
  if (value == null) return "--";
  return `${(value * 1000).toFixed(value < 0.1 ? 1 : 0)} ms`;
}

function formatSeconds(value) {
  if (value == null) return "--";
  return `${value.toFixed(2)} s`;
}

function formatThroughput(value) {
  if (value == null) return "--";
  return `${value.toFixed(2)} tok/s`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "--";
  const mib = bytes / (1024 ** 2);
  if (mib < 1024) return `${mib.toFixed(2)} MB`;
  return `${(bytes / (1024 ** 3)).toFixed(2)} GB`;
}

function formatMemoryPart(bytes, total) {
  if (!total) return "--";
  const percent = 100 * bytes / total;
  return `${formatBytes(bytes)} · ${percent.toFixed(1)}%`;
}

function formatRate(value) { // # [MOD]
  return value == null ? "--" : `${(value * 100).toFixed(1)}%`; // # [MOD]
} // # [MOD]

function updateUserStatistics(state, statistics) {
  if (!statistics) return;
  state.statTTFT.textContent = formatLatency(statistics.ttft);

  // TPOT is measured in seconds/token, so its reciprocal is tokens/second.
  const tpot = Number(statistics.tpot);
  const decodeThroughput = Number.isFinite(tpot) && tpot > 0
    ? 1 / tpot
    : null;
  state.statDecodeThroughput.textContent = formatThroughput(decodeThroughput);

  state.statTPOT.textContent = formatLatency(statistics.tpot);
  state.statTPOTP50.textContent = formatLatency(statistics.tpot_p50); // [MOD]
  state.statTPOTP95.textContent = formatLatency(statistics.tpot_p95); // [MOD]
  state.statTPOTP99.textContent = formatLatency(statistics.tpot_p99); // [MOD]
  state.statE2E.textContent = formatSeconds(statistics.e2e_latency);
  state.statCacheHitRate.textContent = formatRate(statistics.cache_hit_rate); // # [MOD]
}

function setSegment(id, bytes, total) {
  const node = document.querySelector(id);
  const percent = total > 0 ? Math.max(0, 100 * bytes / total) : 0;
  node.style.width = `${percent}%`;
}

function ensureBlockGrid(numBlocks) {
  if (blockNodes.length === numBlocks) return;
  blockGrid.replaceChildren();
  blockNodes = new Array(numBlocks);
  activeBlocks = new Set();
  cachedBlocks = new Set(); // [MOD]

  const fragment = document.createDocumentFragment();
  for (let blockId = 0; blockId < numBlocks; blockId += 1) {
    const node = document.createElement("div");
    node.className = "kv-block empty"; // [MOD] 默认状态明确为 empty。
    node.title = `block ${blockId} · empty`;
    blockNodes[blockId] = node;
    fragment.appendChild(node);
  }
  blockGrid.appendChild(fragment);
}

// [MOD] 一个 block 只属于 empty / cached / active 三种状态之一。
function paintBlock(blockId, activeSet, cachedSet) {
  const node = blockNodes[blockId];
  if (!node) return;

  node.classList.remove("empty", "cached", "active");
  if (activeSet.has(blockId)) {
    node.classList.add("active");
    node.title = `block ${blockId} · active`;
  } else if (cachedSet.has(blockId)) {
    node.classList.add("cached");
    node.title = `block ${blockId} · cached`;
  } else {
    node.classList.add("empty");
    node.title = `block ${blockId} · empty`;
  }
}

function updateBlockGrid(numBlocks, activeBlockIds, cachedBlockIds) {
  ensureBlockGrid(numBlocks);
  const nextActive = new Set(activeBlockIds ?? []);
  const nextCached = new Set(cachedBlockIds ?? []);

  // [MOD] 只重绘状态发生过变化的 block，避免每次刷新扫描全部 DOM。
  const touched = new Set([
    ...activeBlocks,
    ...cachedBlocks,
    ...nextActive,
    ...nextCached,
  ]);
  for (const blockId of touched) {
    paintBlock(blockId, nextActive, nextCached);
  }

  activeBlocks = nextActive;
  cachedBlocks = nextCached;
}

function updateResource(resource) {
  if (!resource) return;
  const total = resource.total_bytes ?? 0;
  const reserved = resource.reserved_bytes ?? 0;
  const weights = resource.weights_bytes ?? 0; // # [MOD]
  const temporary = resource.temporary_peak_bytes ?? 0; // # [MOD]
  const allocated = resource.allocated_kv_bytes ?? 0;
  const available = resource.available_kv_bytes ?? 0;

  document.querySelector("#mem-reserved").textContent = formatMemoryPart(reserved, total);
  document.querySelector("#mem-weights").textContent = formatMemoryPart(weights, total); // # [MOD]
  document.querySelector("#mem-temporary").textContent = formatMemoryPart(temporary, total); // # [MOD]
  document.querySelector("#mem-allocated").textContent = formatMemoryPart(allocated, total);
  document.querySelector("#mem-available").textContent = formatMemoryPart(available, total);

  setSegment("#segment-reserved", reserved, total);
  setSegment("#segment-weights", weights, total); // # [MOD]
  setSegment("#segment-temporary", temporary, total); // # [MOD]
  setSegment("#segment-allocated", allocated, total);
  setSegment("#segment-available", available, total);

  const numBlocks = resource.num_blocks ?? 0;
  const activeIds = resource.active_block_ids ?? [];
  const cachedIds = resource.cached_block_ids ?? []; // [MOD]
  // [MOD] 数量统计和单block容量拆成两行，避免挤压右侧三态图例。
  document.querySelector("#kv-summary").textContent =
    `${activeIds.length} active · ${cachedIds.length} cached · ${numBlocks} total`;
  document.querySelector("#kv-block-size").textContent =
    `${formatBytes(resource.block_bytes ?? 0)} / block`;
  updateBlockGrid(numBlocks, activeIds, cachedIds);
}

function initPerformanceGrid() {
  const fragment = document.createDocumentFragment();
  for (const [key, label] of PERFORMANCE_METRICS) {
    const card = document.createElement("div");
    card.className = "metric-card";
    card.dataset.metric = key;
    card.innerHTML = `<span>${label}</span><strong>--</strong>`;
    fragment.appendChild(card);
  }
  performanceGrid.appendChild(fragment);
}

function updatePerformance(performance) {
  if (!performance) return;
  const count = performance.completed_requests ?? 0;
  document.querySelector("#completed-requests").textContent = `${count} completed`;
  const averages = performance.averages ?? {};
  for (const [key, , kind] of PERFORMANCE_METRICS) {
    const card = performanceGrid.querySelector(`[data-metric="${key}"] strong`);
    card.textContent = kind === "throughput" ? formatThroughput(averages[key]) : formatLatency(averages[key]);
  }
  document.querySelector("#performance-cache-hit-rate").textContent = formatRate(performance.cache_hit_rate);
  const percentiles = performance.tpot_percentiles ?? {};
  for (const key of ["p50", "p95", "p99"]) {
    document.querySelector(`[data-tpot-percentile="${key}"]`).textContent = formatLatency(percentiles[key]);
  }
}

function svgElement(name, attributes = {}, text = null) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) {
    node.setAttribute(key, String(value));
  }
  if (text != null) node.textContent = text;
  return node;
}

function renderSdTrend() {
  const previousScrollLeft = sdTrendWrap.scrollLeft;
  const wasAtRight = previousScrollLeft + sdTrendWrap.clientWidth >= sdTrendWrap.scrollWidth - 4;
  sdTrendChart.replaceChildren();

  if (!sdTrendSamples.length) {
    sdTrendChart.hidden = true;
    sdTrendEmpty.hidden = false;
    sdTrendCount.textContent = "Waiting for rounds";
    sdTrendChart.style.width = "100%";
    return;
  }

  sdTrendChart.hidden = false;
  sdTrendEmpty.hidden = true;

  const height = 190;
  const margin = {top: 12, right: 16, bottom: 30, left: 42};
  const viewportWidth = Math.max(320, sdTrendWrap.clientWidth || 600);
  const viewportPlotWidth = Math.max(1, viewportWidth - margin.left - margin.right);
  const count = sdTrendSamples.length;

  // Keep exactly 100 samples visible per viewport. Older samples remain in
  // memory and extend the SVG horizontally instead of being discarded.
  const pointSpacing = viewportPlotWidth / (SD_TREND_VISIBLE_POINTS - 1);
  const plotWidth = count > SD_TREND_VISIBLE_POINTS
    ? pointSpacing * (count - 1)
    : viewportPlotWidth;
  const width = margin.left + plotWidth + margin.right;
  const x = index => margin.left + (
    count > SD_TREND_VISIBLE_POINTS
      ? index * pointSpacing
      : count > 1
        ? index * plotWidth / (count - 1)
        : plotWidth / 2
  );

  const sliderK = Number(sdMaxDraft.value);
  const currentK = confirmedMaxDraftTokens ?? (Number.isFinite(sliderK) ? sliderK : 5);
  const observedMax = sdTrendSamples.reduce((max, point) => Math.max(max, point.value), 0);
  const adaptiveMax = Math.max(1, Math.ceil(observedMax * 1.25 * 2) / 2);
  const yMax = Number.isFinite(currentK) && currentK > 0
    ? Math.max(1, Math.min(currentK, adaptiveMax))
    : adaptiveMax;
  const plotHeight = height - margin.top - margin.bottom;
  const y = value => margin.top + plotHeight - (value / yMax) * plotHeight;

  sdTrendChart.setAttribute("viewBox", `0 0 ${width} ${height}`);
  sdTrendChart.setAttribute("width", String(width));
  sdTrendChart.setAttribute("height", String(height));
  sdTrendChart.style.width = `${width}px`;

  const grid = svgElement("g", {"class": "sd-chart-grid"});
  for (const tick of [0, yMax / 2, yMax]) {
    const yy = y(tick);
    grid.appendChild(svgElement("line", {
      x1: margin.left, y1: yy, x2: width - margin.right, y2: yy,
    }));
    grid.appendChild(svgElement("text", {
      x: margin.left - 9, y: yy + 4, "text-anchor": "end",
    }, tick.toFixed(1)));
  }

  // Label every 20 samples so each 100-point viewport has readable round labels.
  const xTickIndices = new Set([0, count - 1]);
  for (let i = 20; i < count - 1; i += 20) xTickIndices.add(i);
  for (const index of [...xTickIndices].sort((a, b) => a - b)) {
    const xx = x(index);
    grid.appendChild(svgElement("line", {
      x1: xx, y1: margin.top, x2: xx, y2: height - margin.bottom,
      "class": "sd-chart-x-grid",
    }));
    grid.appendChild(svgElement("text", {
      x: xx, y: height - 7, "text-anchor": "middle",
    }, String(sdTrendSamples[index].round)));
  }
  sdTrendChart.appendChild(grid);

  // One SVG path represents the complete series; avoid one DOM node per point.
  const pathData = sdTrendSamples.map((point, index) =>
    `${index === 0 ? "M" : "L"} ${x(index).toFixed(2)} ${y(point.value).toFixed(2)}`
  ).join(" ");
  sdTrendChart.appendChild(svgElement("path", {
    d: pathData,
    "class": "sd-chart-line",
  }));

  const latest = sdTrendSamples[count - 1];
  const latestPoint = svgElement("circle", {
    cx: x(count - 1),
    cy: y(latest.value),
    r: 3.5,
    "class": "sd-chart-point",
  });
  latestPoint.appendChild(svgElement(
    "title",
    {},
    `Round ${latest.round} · backend cumulative avg ${latest.value.toFixed(2)}`
  ));
  sdTrendChart.appendChild(latestPoint);

  const firstRound = sdTrendSamples[0].round;
  const lastRound = latest.round;
  sdTrendCount.textContent = `${count} points · rounds ${firstRound}–${lastRound}`;

  // Follow new data only while the viewer is already at the right edge.
  // If the user has scrolled into history, preserve their current position.
  requestAnimationFrame(() => {
    const maxScroll = Math.max(0, sdTrendWrap.scrollWidth - sdTrendWrap.clientWidth);
    sdTrendWrap.scrollLeft = wasAtRight
      ? maxScroll
      : Math.min(previousScrollLeft, maxScroll);
  });
}

function updateSdTrend(speculative) {
  const accepted = Number(speculative.accepted_tokens ?? 0);
  const rounds = Number(speculative.verification_rounds ?? 0);
  const avgRaw = speculative.avg_acceptance_per_round;
  const avgAccepted = avgRaw == null ? null : Number(avgRaw);

  if (!Number.isFinite(accepted) || !Number.isFinite(rounds)) return;
  if (avgAccepted != null && !Number.isFinite(avgAccepted)) return;

  // Backend counters are cumulative. If K changes and the backend resets them,
  // start a new chart segment rather than connecting points from different runs.
  if (
    previousSdCounters &&
    (accepted < previousSdCounters.accepted || rounds < previousSdCounters.rounds)
  ) {
    sdTrendSamples = [];
  }
  previousSdCounters = {accepted, rounds};

  // Plot the backend's cumulative avg_acceptance_per_round directly.
  // Repeated runtime snapshots at the same round update that point; no
  // interval-delta calculation or frontend smoothing is applied.
  if (rounds > 0 && avgAccepted != null) {
    const last = sdTrendSamples[sdTrendSamples.length - 1];
    if (last && last.round === rounds) {
      last.value = avgAccepted;
    } else if (!last || rounds > last.round) {
      sdTrendSamples.push({round: rounds, value: avgAccepted});
    }
  }

  renderSdTrend();
}

window.addEventListener("resize", renderSdTrend);

function updateSpeculative(speculative) {
  if (!speculative) return;
  sdEnabled = Boolean(speculative.enabled);
  sdStatistics.hidden = !sdEnabled;
  sdControls.hidden = !sdEnabled;
  globalController.classList.toggle("has-sd", sdEnabled);
  if (!sdEnabled) {
    refreshControls();
    return;
  }

  const maxDraftTokens = Number(speculative.max_draft_tokens);
  if (Number.isFinite(maxDraftTokens) && maxDraftTokens >= 1) {
    confirmedMaxDraftTokens = maxDraftTokens;
    if (document.activeElement !== sdMaxDraft) {
      sdMaxDraft.value = String(maxDraftTokens);
      sdMaxDraftValue.textContent = String(maxDraftTokens);
    }
  }

  sdAcceptanceRate.textContent = formatRate(speculative.acceptance_rate);
  const avgAccepted = speculative.avg_acceptance_per_round;
  sdAvgAccepted.textContent = avgAccepted == null ? "--" : Number(avgAccepted).toFixed(2);
  updateSdTrend(speculative);
  refreshControls();
}

ws.addEventListener("message", (message) => {
  const event = JSON.parse(message.data);

  if (event.type === "runtime") {
    updateResource(event.resource);
    updatePerformance(event.performance);
    updateSpeculative(event.speculative);
    return;
  }

  if (event.type === "accepted") {
    const accepted = new Set(event.accepted_users);
    for (const userId of event.requested_users) {
      if (!accepted.has(userId)) {
        const state = states.get(userId);
        if (state) setBusy(state, false);
      }
    }
    return;
  }

  if (event.type === "sd_config") {
    if (event.speculative) updateSpeculative(event.speculative);
    sdControlStatus.textContent = "Global K updated";
    return;
  }

  if (event.type === "sd_error") {
    if (confirmedMaxDraftTokens != null) {
      sdMaxDraft.value = String(confirmedMaxDraftTokens);
      sdMaxDraftValue.textContent = String(confirmedMaxDraftTokens);
    }
    sdControlStatus.textContent = event.message ?? "Failed to update global K";
    return;
  }

  const state = states.get(event.user_id);
  if (!state) return;

  if (event.type === "token") {
    updateOutput(state, state.outputBase + (event.text ?? ""));
    return;
  }

  if (event.type === "finished") {
    updateOutput(state, state.outputBase + (event.text ?? "") + "\n\n");
    updateUserStatistics(state, event.statistics);
    state.input.value = "";
    setBusy(state, false);
    return;
  }

  if (event.type === "error") {
    updateOutput(state, state.outputBase + `[error] ${event.message}\n\n`);
    setBusy(state, false);
  }
});

ws.addEventListener("open", refreshControls);
ws.addEventListener("close", refreshControls);

USERS.forEach(makePanel);
initPerformanceGrid();
refreshControls();