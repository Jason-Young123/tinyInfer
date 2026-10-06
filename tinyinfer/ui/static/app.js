// [MOD] User panel 从 3 个扩展到 5 个。
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
const ws = new WebSocket(`ws://${location.host}/ws`);

let blockNodes = [];
let activeBlocks = new Set();
let cachedBlocks = new Set(); // [MOD] 独立保存 cached-free block。

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
    statTPOT: panel.querySelector(".stat-tpot"),
    statTPOTP50: panel.querySelector(".stat-tpot-p50"), // [MOD]
    statTPOTP95: panel.querySelector(".stat-tpot-p95"), // [MOD]
    statTPOTP99: panel.querySelector(".stat-tpot-p99"), // [MOD]
    statE2E: panel.querySelector(".stat-e2e"),
    busy: false,
    outputBase: "",
  };
  states.set(userId, state);

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
  for (const state of states.values()) {
    state.button.disabled = !ready(state) || ws.readyState !== WebSocket.OPEN;
  }
  startAll.disabled =
    ws.readyState !== WebSocket.OPEN ||
    ![...states.values()].some(ready);
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

function updateUserStatistics(state, statistics) {
  if (!statistics) return;
  state.statTTFT.textContent = formatLatency(statistics.ttft);
  state.statTPOT.textContent = formatLatency(statistics.tpot);
  state.statTPOTP50.textContent = formatLatency(statistics.tpot_p50); // [MOD]
  state.statTPOTP95.textContent = formatLatency(statistics.tpot_p95); // [MOD]
  state.statTPOTP99.textContent = formatLatency(statistics.tpot_p99); // [MOD]
  state.statE2E.textContent = formatSeconds(statistics.e2e_latency);
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
  const unused = resource.unused_bytes ?? 0;
  const reserved = resource.reserved_bytes ?? 0;
  const allocated = resource.allocated_kv_bytes ?? 0;
  const available = resource.available_kv_bytes ?? 0;

  document.querySelector("#mem-unused").textContent = formatMemoryPart(unused, total);
  document.querySelector("#mem-reserved").textContent = formatMemoryPart(reserved, total);
  document.querySelector("#mem-allocated").textContent = formatMemoryPart(allocated, total);
  document.querySelector("#mem-available").textContent = formatMemoryPart(available, total);

  setSegment("#segment-unused", unused, total);
  setSegment("#segment-reserved", reserved, total);
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
    const value = averages[key];
    card.textContent = kind === "throughput"
      ? formatThroughput(value)
      : formatLatency(value);
  }

  // [MOD] 全局 TPOT 分位数是所有已完成请求的 request-level TPOT 分布。
  const percentiles = performance.tpot_percentiles ?? {};
  for (const key of ["p50", "p95", "p99"]) {
    const node = document.querySelector(`[data-tpot-percentile="${key}"]`);
    node.textContent = formatLatency(percentiles[key]);
  }
}

ws.addEventListener("message", (message) => {
  const event = JSON.parse(message.data);

  if (event.type === "runtime") {
    updateResource(event.resource);
    updatePerformance(event.performance);
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

  const state = states.get(event.user_id);
  if (!state) return;

  if (event.type === "token") {
    state.output.textContent = state.outputBase + (event.text ?? "");
    state.output.scrollTop = state.output.scrollHeight;
    return;
  }

  if (event.type === "finished") {
    state.output.textContent = state.outputBase + (event.text ?? "") + "\n\n";
    state.output.scrollTop = state.output.scrollHeight;
    updateUserStatistics(state, event.statistics);
    state.input.value = "";
    setBusy(state, false);
    return;
  }

  if (event.type === "error") {
    state.output.textContent = state.outputBase + `[error] ${event.message}\n\n`;
    state.output.scrollTop = state.output.scrollHeight;
    setBusy(state, false);
  }
});

ws.addEventListener("open", refreshControls);
ws.addEventListener("close", () => {
  for (const state of states.values()) state.button.disabled = true;
  startAll.disabled = true;
});

USERS.forEach(makePanel);
initPerformanceGrid();
refreshControls();