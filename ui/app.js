const snapshotUrl = "../dataset/network_snapshots.json";
const decisionsUrl = "../dataset/routing_decisions.csv";

const COLORS = {
  rpl: "#536878",
  rl: "#147d75",
  packet: "#f59f00",
  green: "#3f9b57",
  amber: "#d99022",
  red: "#c0392b",
  graphite: "#293241",
};

const state = {
  config: null,
  protocols: {},
  comparison: null,
  decisions: [],
  frame: 0,
  playing: false,
  timer: null,
};

const rplCanvas = document.getElementById("rplCanvas");
const rplCtx = rplCanvas.getContext("2d");
const rlCanvas = document.getElementById("deepQCanvas");
const rlCtx = rlCanvas.getContext("2d");
const metricCanvas = document.getElementById("metricCanvas");
const metricCtx = metricCanvas.getContext("2d");
const slider = document.getElementById("roundSlider");
const playButton = document.getElementById("playButton");
const prevButton = document.getElementById("prevButton");
const nextButton = document.getElementById("nextButton");
const searchInput = document.getElementById("searchInput");
const timeFilter = document.getElementById("roundFilter");

async function boot() {
  try {
    const cacheBust = `fresh=${Date.now()}`;
    const [snapshotResponse, decisionResponse] = await Promise.all([
      fetch(`${snapshotUrl}?${cacheBust}`),
      fetch(`${decisionsUrl}?${cacheBust}`),
    ]);

    const snapshotData = await snapshotResponse.json();
    state.config = snapshotData.config;
    state.protocols = normalizeProtocols(snapshotData.protocols || {});
    state.comparison = snapshotData.comparison || null;
    state.decisions = parseCsv(await decisionResponse.text());

    slider.max = String(frameCount() - 1);
    document.getElementById("lastRoundLabel").textContent = `Time ${lastSnapshotTime()}s`;
    fillTimeFilter();
    setStatus("Data loaded", true);
    render();
  } catch (error) {
    setStatus("Run python visual_simulation.py first", false);
    console.error(error);
  }
}

function normalizeProtocols(protocols) {
  return {
    traditional: protocols.traditional || { name: "Traditional RPL", snapshots: [], metrics: [] },
    rl_rpl: protocols.rl_rpl || protocols.deep_q || { name: "Q-learning RL-RPL", snapshots: [], metrics: [] },
  };
}

function frameCount() {
  return Math.min(
    state.protocols.traditional?.snapshots?.length || 0,
    state.protocols.rl_rpl?.snapshots?.length || 0,
  );
}

function lastSnapshotTime() {
  const last = state.protocols.rl_rpl?.snapshots?.[frameCount() - 1];
  return last ? Number(last.time).toFixed(0) : "--";
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];
    if (char === '"' && quoted && next === '"') {
      cell += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(cell);
      if (row.some((value) => value.length > 0)) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }
  if (cell || row.length) {
    row.push(cell);
    rows.push(row);
  }

  const headers = rows.shift() || [];
  return rows.map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] || ""])));
}

function setStatus(text, ok) {
  document.getElementById("statusText").textContent = text;
  document.getElementById("statusDot").style.background = ok ? COLORS.green : COLORS.amber;
}

function fillTimeFilter() {
  timeFilter.innerHTML = '<option value="all">All times</option>';
  state.protocols.rl_rpl.snapshots.forEach((snapshot) => {
    const option = document.createElement("option");
    option.value = String(snapshot.time);
    option.textContent = `Time ${Number(snapshot.time).toFixed(0)}s`;
    timeFilter.appendChild(option);
  });
}

function render() {
  const rplSnapshot = snapshotFor("traditional");
  const rlSnapshot = snapshotFor("rl_rpl");
  if (!rplSnapshot || !rlSnapshot) return;

  slider.value = String(state.frame);
  updateKpis(rplSnapshot, rlSnapshot);
  drawTree(rplCanvas, rplCtx, rplSnapshot, COLORS.rpl);
  drawTree(rlCanvas, rlCtx, rlSnapshot, COLORS.rl);
  drawMetricComparison();
  renderWinnerBanner();
  renderComparisonTable();
  renderDecisionTable();
  updateFocus();
}

function snapshotFor(protocolKey) {
  return state.protocols[protocolKey]?.snapshots?.[state.frame] || null;
}

function metricFor(protocolKey) {
  return state.protocols[protocolKey]?.metrics?.[state.frame] || snapshotFor(protocolKey);
}

function updateKpis(rplSnapshot, rlSnapshot) {
  const rplMetric = metricFor("traditional");
  const rlMetric = metricFor("rl_rpl");
  const packet = rlSnapshot.current_packet;
  const winner = state.comparison?.overall_winner || "--";

  document.getElementById("roundValue").textContent = `${Number(rlSnapshot.time).toFixed(0)}s`;
  document.getElementById("energySavedValue").textContent =
    `${(Number(rplMetric.total_energy_consumed) - Number(rlMetric.total_energy_consumed)).toFixed(3)} J`;
  document.getElementById("energySavedHint").textContent = "RPL minus RL-RPL";
  document.getElementById("winnerValue").textContent = winner.replace("Q-learning ", "");
  document.getElementById("winnerHint").textContent =
    `RL wins ${state.comparison?.rl_rpl_wins ?? "--"} rows | RPL wins ${state.comparison?.traditional_wins ?? "--"}`;
  document.getElementById("aliveValue").textContent = `${rlSnapshot.alive_nodes} vs ${rplSnapshot.alive_nodes}`;
  document.getElementById("pdrValue").textContent = `${percent(rlMetric.packet_delivery_ratio)} vs ${percent(rplMetric.packet_delivery_ratio)}`;
  document.getElementById("delayValue").textContent = `${Number(rlMetric.average_delay).toFixed(2)}s vs ${Number(rplMetric.average_delay).toFixed(2)}s`;

  const packetText = packet
    ? `${packet.packet_id} | sensor ${packet.source_sensor_id} | ${Number(packet.temperature).toFixed(1)} C | ${packet.status}`
    : "No packet at this time";
  document.getElementById("focusTitle").textContent = "Current temperature packet";
  document.getElementById("focusBody").textContent = packetText;
}

function percent(value) {
  return `${Math.round(Number(value) * 100)}%`;
}

function drawTree(canvasElement, drawingContext, snapshot, linkColor) {
  const width = canvasElement.width;
  const height = canvasElement.height;
  const margin = 28;
  const sx = (x) => margin + (x / state.config.area_width) * (width - margin * 2);
  const sy = (y) => height - margin - (y / state.config.area_height) * (height - margin * 2);
  const nodes = snapshot.nodes;

  drawingContext.clearRect(0, 0, width, height);
  drawingContext.fillStyle = "#fdfefe";
  drawingContext.fillRect(0, 0, width, height);
  drawGrid(drawingContext, width, height, margin);

  Object.entries(nodes).forEach(([id, node]) => {
    if (id === "0" || node.parent === null || !node.alive) return;
    const parent = nodes[String(node.parent)];
    if (!parent) return;
    drawLine(drawingContext, sx(node.x), sy(node.y), sx(parent.x), sy(parent.y), linkColor, 1.5, 0.48);
  });

  drawPacketRoute(drawingContext, snapshot, sx, sy);

  Object.entries(nodes).forEach(([id, node]) => {
    const x = sx(node.x);
    const y = sy(node.y);
    if (node.is_root || node.is_sink) {
      drawRoot(drawingContext, x, y);
      drawLabel(drawingContext, "ROOT", x + 12, y - 12, true);
      return;
    }
    if (!node.alive) {
      drawDeadNode(drawingContext, x, y);
    } else {
      drawSensorNode(drawingContext, x, y, Number(node.energy) / state.config.initial_energy);
    }
    drawLabel(drawingContext, id, x + 9, y - 7, false);
  });

  drawSnapshotBadge(drawingContext, snapshot, width);
}

function drawGrid(drawingContext, width, height, margin) {
  drawingContext.strokeStyle = "#edf1f2";
  drawingContext.lineWidth = 1;
  for (let i = 0; i <= 5; i += 1) {
    const x = margin + ((width - margin * 2) / 5) * i;
    const y = margin + ((height - margin * 2) / 5) * i;
    drawingContext.beginPath();
    drawingContext.moveTo(x, margin);
    drawingContext.lineTo(x, height - margin);
    drawingContext.stroke();
    drawingContext.beginPath();
    drawingContext.moveTo(margin, y);
    drawingContext.lineTo(width - margin, y);
    drawingContext.stroke();
  }
}

function drawPacketRoute(drawingContext, snapshot, sx, sy) {
  const packet = snapshot.current_packet;
  if (!packet || !Array.isArray(packet.route_path) || packet.route_path.length < 2) return;
  const route = packet.route_path.map(String);
  for (let i = 0; i < route.length - 1; i += 1) {
    const node = snapshot.nodes[route[i]];
    const parent = snapshot.nodes[route[i + 1]];
    if (!node || !parent) continue;
    drawLine(drawingContext, sx(node.x), sy(node.y), sx(parent.x), sy(parent.y), COLORS.packet, 3, 0.9);
  }
  const last = snapshot.nodes[route[route.length - 1]];
  if (last) {
    drawingContext.fillStyle = COLORS.packet;
    drawingContext.strokeStyle = "#17202a";
    drawingContext.lineWidth = 1;
    drawingContext.beginPath();
    drawingContext.arc(sx(last.x), sy(last.y), 6, 0, Math.PI * 2);
    drawingContext.fill();
    drawingContext.stroke();
  }
}

function drawLine(drawingContext, x1, y1, x2, y2, color, width, alpha) {
  drawingContext.strokeStyle = hexToRgba(color, alpha);
  drawingContext.lineWidth = width;
  drawingContext.beginPath();
  drawingContext.moveTo(x1, y1);
  drawingContext.lineTo(x2, y2);
  drawingContext.stroke();
}

function drawSensorNode(drawingContext, x, y, energyRatio) {
  const color = energyRatio > 0.65 ? COLORS.green : energyRatio > 0.32 ? COLORS.amber : COLORS.red;
  drawingContext.fillStyle = color;
  drawingContext.strokeStyle = "#17202a";
  drawingContext.lineWidth = 1.4;
  drawingContext.beginPath();
  drawingContext.arc(x, y, 8, 0, Math.PI * 2);
  drawingContext.fill();
  drawingContext.stroke();
}

function drawRoot(drawingContext, x, y) {
  drawingContext.fillStyle = COLORS.red;
  drawingContext.strokeStyle = "#17202a";
  drawingContext.lineWidth = 1.4;
  drawingContext.beginPath();
  for (let i = 0; i < 10; i += 1) {
    const radius = i % 2 === 0 ? 15 : 7;
    const angle = -Math.PI / 2 + (i * Math.PI) / 5;
    const px = x + Math.cos(angle) * radius;
    const py = y + Math.sin(angle) * radius;
    if (i === 0) drawingContext.moveTo(px, py);
    else drawingContext.lineTo(px, py);
  }
  drawingContext.closePath();
  drawingContext.fill();
  drawingContext.stroke();
}

function drawDeadNode(drawingContext, x, y) {
  drawingContext.strokeStyle = COLORS.graphite;
  drawingContext.lineWidth = 3;
  drawingContext.beginPath();
  drawingContext.moveTo(x - 7, y - 7);
  drawingContext.lineTo(x + 7, y + 7);
  drawingContext.moveTo(x + 7, y - 7);
  drawingContext.lineTo(x - 7, y + 7);
  drawingContext.stroke();
}

function drawLabel(drawingContext, text, x, y, strong) {
  drawingContext.fillStyle = "#17202a";
  drawingContext.font = `${strong ? "700 " : ""}11px Segoe UI, Arial`;
  drawingContext.fillText(text, x, y);
}

function drawSnapshotBadge(drawingContext, snapshot, width) {
  const packet = snapshot.current_packet;
  const text = packet
    ? `${packet.packet_id}: ${Number(packet.temperature).toFixed(1)} C, ${packet.status}`
    : "No packet generated at this second";
  drawingContext.fillStyle = "rgba(255, 255, 255, 0.86)";
  drawingContext.fillRect(12, 12, Math.min(width - 24, 360), 30);
  drawingContext.fillStyle = "#17202a";
  drawingContext.font = "12px Segoe UI, Arial";
  drawingContext.fillText(text, 22, 32);
}

function drawMetricComparison() {
  const rplMetrics = state.protocols.traditional.metrics;
  const rlMetrics = state.protocols.rl_rpl.metrics;
  metricCtx.clearRect(0, 0, metricCanvas.width, metricCanvas.height);
  metricCtx.fillStyle = "#ffffff";
  metricCtx.fillRect(0, 0, metricCanvas.width, metricCanvas.height);

  const visibleRpl = rplMetrics.slice(0, state.frame + 1);
  const visibleRl = rlMetrics.slice(0, state.frame + 1);
  const maxEnergy = Math.max(
    1,
    ...rplMetrics.map((metric) => Number(metric.total_energy_consumed)),
    ...rlMetrics.map((metric) => Number(metric.total_energy_consumed)),
  );
  const maxDelay = Math.max(
    1,
    ...rplMetrics.map((metric) => Number(metric.average_delay)),
    ...rlMetrics.map((metric) => Number(metric.average_delay)),
  );

  drawSeries(visibleRpl, "total_energy_consumed", COLORS.rpl, maxEnergy, "RPL energy", 0);
  drawSeries(visibleRl, "total_energy_consumed", COLORS.rl, maxEnergy, "RL-RPL energy", 1);
  drawSeries(visibleRpl, "average_delay", "#7b2cbf", maxDelay, "RPL delay", 2);
  drawSeries(visibleRl, "average_delay", COLORS.packet, maxDelay, "RL-RPL delay", 3);
}

function drawSeries(data, key, color, maxValue, label, labelIndex) {
  if (data.length < 2) return;
  const width = metricCanvas.width;
  const height = metricCanvas.height;
  const margin = 30;

  metricCtx.strokeStyle = color;
  metricCtx.lineWidth = 2.6;
  metricCtx.beginPath();
  data.forEach((row, index) => {
    const x = margin + (index / Math.max(1, frameCount() - 1)) * (width - margin * 2);
    const normalized = Math.min(1, Number(row[key]) / maxValue);
    const y = height - margin - normalized * (height - margin * 2);
    if (index === 0) metricCtx.moveTo(x, y);
    else metricCtx.lineTo(x, y);
  });
  metricCtx.stroke();
  metricCtx.fillStyle = color;
  metricCtx.font = "12px Segoe UI, Arial";
  metricCtx.fillText(label, margin + 8 + (labelIndex % 2) * 160, margin + 16 + Math.floor(labelIndex / 2) * 18);
}

function renderWinnerBanner() {
  const banner = document.getElementById("winnerBanner");
  const label = document.getElementById("winnerBannerLabel");
  const title = document.getElementById("overallWinnerName");
  const detail = document.getElementById("overallWinnerDetail");
  if (!banner || !label || !title || !detail) return;

  const winner = state.comparison?.overall_winner || "Tie";
  banner.className = `winner-banner ${winnerClass(winner)}`;
  label.textContent = "Final Winner";
  title.textContent = winner === "Tie" ? "No Overall Winner" : `${winner} Performs Better`;
  detail.textContent =
    `Comparison rows: RL-RPL ${state.comparison?.rl_rpl_wins ?? 0}, Traditional RPL ${state.comparison?.traditional_wins ?? 0}, ties ${state.comparison?.ties ?? 0}.`;
}

function renderComparisonTable() {
  const tbody = document.getElementById("comparisonTable");
  const rows = state.comparison?.rows || [];
  tbody.innerHTML = rows
    .map((row) => `
      <tr>
        <td>${escapeHtml(row.Metric)}</td>
        <td>${escapeHtml(row["Traditional RPL"])}</td>
        <td>${escapeHtml(row["Q-learning RL-RPL"])}</td>
        <td><span class="winner-badge ${winnerClass(row["Better Approach"])}">${escapeHtml(row["Better Approach"])}</span></td>
      </tr>
    `)
    .join("");
}

function renderDecisionTable() {
  const tbody = document.getElementById("decisionTable");
  const selectedTime = timeFilter.value === "all" ? String(snapshotFor("rl_rpl")?.time ?? "") : timeFilter.value;
  const term = searchInput.value.trim().toLowerCase();
  const rows = state.decisions
    .filter((row) => timeFilter.value === "all" || nearlyEqual(Number(row.time), Number(selectedTime)))
    .filter((row) => !term || `${row.protocol} ${row.node_id} ${row.selected_parent} ${row.decision_mode}`.toLowerCase().includes(term))
    .slice(0, 100);

  tbody.innerHTML = rows
    .map((row) => `
      <tr>
        <td>${Number(row.time).toFixed(0)}s</td>
        <td>${escapeHtml(row.protocol)}</td>
        <td>${escapeHtml(row.node_id)}</td>
        <td>${escapeHtml(row.candidate_parent)}</td>
        <td>${escapeHtml(row.selected_parent)}</td>
        <td>${escapeHtml(row.decision_mode)}</td>
        <td>${formatNumber(row.selection_value, 3)}</td>
        <td>${formatNumber(row.reward, 3)}</td>
      </tr>
    `)
    .join("");
}

function updateFocus() {
  const currentTime = Number(snapshotFor("rl_rpl")?.time ?? 0);
  const row = state.decisions.find((item) => item.protocol === "Q-learning RL-RPL" && nearlyEqual(Number(item.time), currentTime));
  if (!row) return;
  const packet = snapshotFor("rl_rpl")?.current_packet;
  document.getElementById("focusTitle").textContent =
    `RL-RPL node ${row.node_id} selected parent ${row.selected_parent}`;
  document.getElementById("focusBody").textContent = packet
    ? `${packet.packet_id}: ${Number(packet.temperature).toFixed(1)} C, path ${packet.selected_route_path}, status ${packet.status}.`
    : `Decision mode ${row.decision_mode}; selection ${formatNumber(row.selection_value, 3)}.`;
}

function winnerClass(winner) {
  if (winner === "Q-learning RL-RPL") return "deep-q";
  if (winner === "Traditional RPL") return "rpl";
  return "tie";
}

function nearlyEqual(left, right) {
  return Math.abs(Number(left) - Number(right)) < 0.000001;
}

function formatNumber(value, digits) {
  const numeric = Number(value);
  return Number.isNaN(numeric) ? "--" : numeric.toFixed(digits);
}

function hexToRgba(hex, alpha) {
  const value = hex.replace("#", "");
  const r = parseInt(value.slice(0, 2), 16);
  const g = parseInt(value.slice(2, 4), 16);
  const b = parseInt(value.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[char]);
}

function step(delta) {
  state.frame = Math.max(0, Math.min(frameCount() - 1, state.frame + delta));
  render();
}

playButton.addEventListener("click", () => {
  state.playing = !state.playing;
  playButton.textContent = state.playing ? "Pause" : "Play";
  if (state.playing) {
    state.timer = setInterval(() => {
      state.frame = (state.frame + 1) % frameCount();
      render();
    }, 450);
  } else {
    clearInterval(state.timer);
  }
});

prevButton.addEventListener("click", () => step(-1));
nextButton.addEventListener("click", () => step(1));
slider.addEventListener("input", (event) => {
  state.frame = Number(event.target.value);
  render();
});
searchInput.addEventListener("input", renderDecisionTable);
timeFilter.addEventListener("change", renderDecisionTable);

boot();
