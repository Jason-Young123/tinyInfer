const USERS = ["user1", "user2", "user3"];
const states = new Map();
const usersRoot = document.querySelector("#users");
const startAll = document.querySelector("#start-all");
const ws = new WebSocket(`ws://${location.host}/ws`);

function makePanel(userId, index) {
  const panel = document.createElement("section");
  panel.className = "panel user-panel";
  panel.innerHTML = `
    <div class="panel-head">
      <div class="panel-title">User ${index + 1}</div>
      <div class="status">Idle</div>
    </div>
    <div class="output"></div>
    <div class="composer">
      <textarea placeholder="Type a prompt..."></textarea>
      <button class="primary" disabled>Start</button>
    </div>`;

  const state = {
    userId,
    panel,
    input: panel.querySelector("textarea"),
    button: panel.querySelector("button"),
    output: panel.querySelector(".output"),
    status: panel.querySelector(".status"),
    busy: false,
    outputBase: "",
  };
  states.set(userId, state);

  state.input.addEventListener("input", refreshControls);
  state.button.addEventListener("click", () => submitOne(state));
  usersRoot.appendChild(panel);
}

function ready(state) {
  return !state.busy && state.input.value.trim().length > 0;
}

function setBusy(state, busy) {
  state.busy = busy;
  state.input.readOnly = busy;
  state.status.textContent = busy ? "Running" : "Idle";
  state.status.classList.toggle("busy", busy);
  refreshControls();
}

function refreshControls() {
  for (const state of states.values()) {
    state.button.disabled = !ready(state);
  }
  startAll.disabled = ![...states.values()].some(ready);
}

function beginLocal(state) {
  state.outputBase = state.output.textContent;
  setBusy(state, true);
}

function submitOne(state) {
  if (!ready(state) || ws.readyState !== WebSocket.OPEN) return;
  const text = state.input.value;
  beginLocal(state);
  ws.send(JSON.stringify({type: "submit", user_id: state.userId, text}));
}

startAll.addEventListener("click", () => {
  if (ws.readyState !== WebSocket.OPEN) return;
  const requests = [];
  for (const state of states.values()) {
    if (!ready(state)) continue;
    requests.push({user_id: state.userId, text: state.input.value});
    beginLocal(state);
  }
  if (requests.length) {
    ws.send(JSON.stringify({type: "submit_many", requests}));
  }
});

ws.addEventListener("message", (message) => {
  const event = JSON.parse(message.data);

  if (event.type === "accepted") {
    const accepted = new Set(event.accepted_users);
    for (const userId of event.requested_users) {
      if (!accepted.has(userId)) setBusy(states.get(userId), false);
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
refreshControls();



