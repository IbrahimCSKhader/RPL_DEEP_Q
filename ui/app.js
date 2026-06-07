const snapshotUrl = "../dataset/network_snapshots.json";
const decisionsUrl = "../dataset/rl_rpl_decisions.csv";

const CHART_COLORS = {
  green: "#2e7d32",
  red: "#c62828",
  purple: "#7b2cbf",
  yellow: "#f2c94c",
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
const deepQCanvas = document.getElementById("deepQCanvas");
const deepQCtx = deepQCanvas.getContext("2d");
const metricCanvas = document.getElementById("metricCanvas");
const metricCtx = metricCanvas.getContext("2d");
const slider = document.getElementById("roundSlider");
const playButton = document.getElementById("playButton");
const prevButton = document.getElementById("prevButton");
const nextButton = document.getElementById("nextButton");
const searchInput = document.getElementById("searchInput");
const roundFilter = document.getElementById("roundFilter");

async function boot() {
  try {
    const cacheBust = `fresh=${Date.now()}`;
    const [snapshotResponse, decisionResponse] = await Promise.all([
      fetch(`${snapshotUrl}?${cacheBust}`),
      fetch(`${decisionsUrl}?${cacheBust}`),
    ]);

    const snapshotData = await snapshotResponse.json();
    state.config = snapshotData.config;
    state.protocols = normalizeProtocols(snapshotData);
    state.comparison = snapshotData.comparison || null;
    state.decisions = parseCsv(await decisionResponse.text());

    slider.max = String(roundCount() - 1);
    document.getElementById("lastRoundLabel").textContent = `Round ${roundCount()}`;
    fillRoundFilter();
    setStatus("Data loaded", true);
    render();
  } catch (error) {
    setStatus("Run python visual_simulation.py first", false);
    console.error(error);
  }
}

function normalizeProtocols(snapshotData) {
  if (snapshotData.protocols?.traditional && snapshotData.protocols?.deep_q) {
    return snapshotData.protocols;
  }
  return {
    traditional: {
      name: "Traditional RPL",
      snapshots: snapshotData.snapshots || [],
      metrics: [],
    },
    deep_q: {
      name: "Deep Q RL-RPL",
      snapshots: snapshotData.snapshots || [],
      metrics: [],
    },
  };
}

function roundCount() {
  return Math.min(
    state.protocols.traditional?.snapshots?.length || 0,
    state.protocols.deep_q?.snapshots?.length || 0,
  );
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
  document.getElementById("statusDot").style.background = ok ? "#3f9b57" : "#d99022";
}

function fillRoundFilter() {
  roundFilter.innerHTML = '<option value="all">All rounds</option>';
  for (let index = 0; index < roundCount(); index += 1) {
    const round = state.protocols.deep_q.snapshots[index].round;
    const option = document.createElement("option");
    option.value = String(round);
    option.textContent = `Round ${round}`;
    roundFilter.appendChild(option);
  }
}

function render() {
  const rplSnapshot = snapshotFor("traditional");
  const deepQSnapshot = snapshotFor("deep_q");
  if (!rplSnapshot || !deepQSnapshot) return;

  slider.value = String(state.frame);
  updateKpis(rplSnapshot, deepQSnapshot);
  drawTree(rplCanvas, rplCtx, rplSnapshot, "#687076");
  drawTree(deepQCanvas, deepQCtx, deepQSnapshot, "#147d75");
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

function updateKpis(rplSnapshot, deepQSnapshot) {
  const rplMetric = metricFor("traditional");
  const deepQMetric = metricFor("deep_q");
  const rplEnergyPerPacket = energyPerDelivered("traditional", rplMetric);
  const deepQEnergyPerPacket = energyPerDelivered("deep_q", deepQMetric);
  const energySaved = rplEnergyPerPacket - deepQEnergyPerPacket;
  const energySavedPercent = rplEnergyPerPacket
    ? (energySaved / rplEnergyPerPacket) * 100
    : 0;

  document.getElementById("roundValue").textContent = rplSnapshot.round;
  document.getElementById("energySavedValue").textContent = `${energySaved.toFixed(4)} J`;
  document.getElementById("energySavedHint").textContent = `${energySavedPercent.toFixed(1)}% vs RPL`;
  document.getElementById("aliveValue").textContent = `${deepQSnapshot.alive_nodes} vs ${rplSnapshot.alive_nodes}`;
  document.getElementById("pdrValue").textContent = `${percent(deepQMetric.packet_delivery_ratio)} vs ${percent(rplMetric.packet_delivery_ratio)}`;
  document.getElementById("delayValue").textContent = `${Number(deepQMetric.average_delay).toFixed(2)} vs ${Number(rplMetric.average_delay).toFixed(2)}`;

  const roundWinner = currentRoundWinner();
  document.getElementById("winnerValue").textContent = roundWinner.name;
  document.getElementById("winnerHint").textContent =
    `Deep Q ${roundWinner.deepQScore} pts | RPL ${roundWinner.rplScore} pts`;
}

function energyPerDelivered(protocolKey, currentMetric) {
  const delivered = state.protocols[protocolKey].metrics
    .slice(0, state.frame + 1)
    .reduce((total, metric) => total + Number(metric.delivered_packets), 0);
  return delivered ? Number(currentMetric.total_energy_consumed) / delivered : 0;
}

function overallWinner() {
  if (state.comparison?.weighted_winner) {
    const weighted = state.comparison.weighted_winner;
    const isDeepQ = weighted.overall === "Deep Q RL-RPL";
    return {
      name: isDeepQ ? "Deep Q" : "RPL",
      fullName: weighted.overall,
      className: isDeepQ ? "deep-q" : "rpl",
      score: isDeepQ ? weighted.deep_q_score : weighted.traditional_score,
      deepQWins: weighted.deep_q_wins,
      rplWins: weighted.traditional_wins,
      tieWins: weighted.tie_wins,
      deepQScore: weighted.deep_q_score,
      rplScore: weighted.traditional_score,
      tieScore: weighted.tie_score,
      method: weighted.method,
    };
  }
  const winners = Object.values(state.comparison?.winner_by_metric || {});
  const deepQWins = winners.filter((winner) => winner === "Deep Q RL-RPL").length;
  const rplWins = winners.filter((winner) => winner === "Traditional RPL").length;
  const tieWins = winners.filter((winner) => winner === "Tie").length;
  if (deepQWins === rplWins) {
    return {
      name: "Tie",
      fullName: "Tie",
      className: "tie",
      score: deepQWins,
      deepQWins,
      rplWins,
      tieWins,
      deepQScore: deepQWins,
      rplScore: rplWins,
      tieScore: tieWins,
      method: "Simple metric count",
    };
  }
  if (deepQWins > rplWins) {
    return {
      name: "Deep Q",
      fullName: "Deep Q RL-RPL",
      className: "deep-q",
      score: deepQWins,
      deepQWins,
      rplWins,
      tieWins,
      deepQScore: deepQWins,
      rplScore: rplWins,
      tieScore: tieWins,
      method: "Simple metric count",
    };
  }
  return {
    name: "RPL",
    fullName: "Traditional RPL",
    className: "rpl",
    score: rplWins,
    deepQWins,
    rplWins,
    tieWins,
    deepQScore: deepQWins,
    rplScore: rplWins,
    tieScore: tieWins,
    method: "Simple metric count",
  };
}

function currentRoundWinner() {
  const rplMetric = metricFor("traditional");
  const deepQMetric = metricFor("deep_q");
  const weights = state.comparison?.metric_weights || {};
  const comparisons = {
    energy_consumption: lowerWinner(
      Number(rplMetric.total_energy_consumed),
      Number(deepQMetric.total_energy_consumed),
    ),
    energy_per_delivered_packet: lowerWinner(
      energyPerDelivered("traditional", rplMetric),
      energyPerDelivered("deep_q", deepQMetric),
    ),
    average_remaining_energy: higherWinner(
      Number(rplMetric.average_remaining_energy),
      Number(deepQMetric.average_remaining_energy),
    ),
    alive_nodes: higherWinner(
      Number(rplMetric.alive_nodes),
      Number(deepQMetric.alive_nodes),
    ),
    delivered_packets: higherWinner(
      Number(rplMetric.delivered_packets),
      Number(deepQMetric.delivered_packets),
    ),
    packet_delivery_ratio: higherWinner(
      Number(rplMetric.packet_delivery_ratio),
      Number(deepQMetric.packet_delivery_ratio),
    ),
    average_delay: delayWinner(rplMetric, deepQMetric),
  };

  let deepQScore = 0;
  let rplScore = 0;
  let tieScore = 0;
  let deepQWins = 0;
  let rplWins = 0;
  let tieWins = 0;

  Object.entries(comparisons).forEach(([metric, winner]) => {
    const weight = weights[metric] || 1;
    if (winner === "Deep Q RL-RPL") {
      deepQScore += weight;
      deepQWins += 1;
    } else if (winner === "Traditional RPL") {
      rplScore += weight;
      rplWins += 1;
    } else {
      tieScore += weight;
      tieWins += 1;
    }
  });

  let fullName = "Tie";
  if (deepQScore > rplScore) fullName = "Deep Q RL-RPL";
  if (rplScore > deepQScore) fullName = "Traditional RPL";
  if (deepQScore === rplScore) {
    fullName = comparisons.packet_delivery_ratio === "Deep Q RL-RPL"
      ? "Deep Q RL-RPL"
      : comparisons.packet_delivery_ratio === "Traditional RPL"
        ? "Traditional RPL"
        : "Tie";
  }

  return {
    name: fullName === "Deep Q RL-RPL" ? "Deep Q" : fullName === "Traditional RPL" ? "RPL" : "Tie",
    fullName,
    className: winnerClass(fullName),
    deepQScore,
    rplScore,
    tieScore,
    deepQWins,
    rplWins,
    tieWins,
    comparisons,
  };
}

function lowerWinner(rplValue, deepQValue) {
  if (nearlyEqual(rplValue, deepQValue)) return "Tie";
  return rplValue < deepQValue ? "Traditional RPL" : "Deep Q RL-RPL";
}

function higherWinner(rplValue, deepQValue, deepQFirst = false) {
  if (nearlyEqual(rplValue, deepQValue)) return "Tie";
  if (deepQFirst) return deepQValue > rplValue ? "Deep Q RL-RPL" : "Traditional RPL";
  return rplValue > deepQValue ? "Traditional RPL" : "Deep Q RL-RPL";
}

function delayWinner(rplMetric, deepQMetric) {
  const rplDelivered = Number(rplMetric.delivered_packets);
  const deepQDelivered = Number(deepQMetric.delivered_packets);
  if (rplDelivered === 0 && deepQDelivered > 0) return "Deep Q RL-RPL";
  if (deepQDelivered === 0 && rplDelivered > 0) return "Traditional RPL";
  if (rplDelivered === 0 && deepQDelivered === 0) return "Tie";
  return lowerWinner(Number(rplMetric.average_delay), Number(deepQMetric.average_delay));
}

function nearlyEqual(left, right) {
  return Math.abs(Number(left) - Number(right)) < 0.000001;
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
    drawingContext.strokeStyle = hexToRgba(linkColor, 0.48);
    drawingContext.lineWidth = 1.5;
    drawingContext.beginPath();
    drawingContext.moveTo(sx(node.x), sy(node.y));
    drawingContext.lineTo(sx(parent.x), sy(parent.y));
    drawingContext.stroke();
  });

  Object.entries(nodes).forEach(([id, node]) => {
    const x = sx(node.x);
    const y = sy(node.y);
    if (node.is_sink) {
      drawSink(drawingContext, x, y);
      drawLabel(drawingContext, "ROOT", x + 12, y - 12, true);
      return;
    }
    if (!node.alive) {
      drawDeadNode(drawingContext, x, y);
    } else {
      drawRouterNode(drawingContext, x, y, Number(node.energy) / state.config.initial_energy);
    }
    drawLabel(drawingContext, id, x + 9, y - 7, false);
  });
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

function drawRouterNode(drawingContext, x, y, energyRatio) {
  const color = energyRatio > 0.65 ? "#3f9b57" : energyRatio > 0.32 ? "#d99022" : "#c0392b";
  drawingContext.fillStyle = color;
  drawingContext.strokeStyle = "#17202a";
  drawingContext.lineWidth = 1.4;
  drawingContext.beginPath();
  drawingContext.arc(x, y, 8, 0, Math.PI * 2);
  drawingContext.fill();
  drawingContext.stroke();
}

function drawSink(drawingContext, x, y) {
  drawingContext.fillStyle = "#c0392b";
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
  drawingContext.strokeStyle = "#293241";
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

function drawMetricComparison() {
  const rplMetrics = state.protocols.traditional.metrics;
  const deepQMetrics = state.protocols.deep_q.metrics;
  metricCtx.clearRect(0, 0, metricCanvas.width, metricCanvas.height);
  metricCtx.fillStyle = "#ffffff";
  metricCtx.fillRect(0, 0, metricCanvas.width, metricCanvas.height);

  const visibleRpl = rplMetrics.slice(0, state.frame + 1);
  const visibleDeepQ = deepQMetrics.slice(0, state.frame + 1);
  const maxEnergy = Math.max(
    1,
    ...rplMetrics.map((metric) => Number(metric.total_energy_consumed)),
    ...deepQMetrics.map((metric) => Number(metric.total_energy_consumed)),
  );
  const maxDelay = Math.max(
    1,
    ...rplMetrics.map((metric) => Number(metric.average_delay)),
    ...deepQMetrics.map((metric) => Number(metric.average_delay)),
  );

  drawSeries(visibleRpl, "total_energy_consumed", CHART_COLORS.green, maxEnergy, "RPL energy", 0);
  drawSeries(visibleDeepQ, "total_energy_consumed", CHART_COLORS.red, maxEnergy, "Deep Q energy", 1);
  drawSeries(visibleRpl, "average_delay", CHART_COLORS.purple, maxDelay, "RPL delay", 2);
  drawSeries(visibleDeepQ, "average_delay", CHART_COLORS.yellow, maxDelay, "Deep Q delay", 3);
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
    const x = margin + (index / Math.max(1, roundCount() - 1)) * (width - margin * 2);
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

  const roundWinner = currentRoundWinner();
  const round = snapshotFor("deep_q")?.round || state.frame + 1;
  banner.className = `winner-banner ${roundWinner.className}`;
  label.textContent = `Round ${round} Winner`;
  title.textContent = roundWinner.fullName === "Tie"
    ? "No Winner This Round"
    : `${roundWinner.fullName} Wins This Round`;
  detail.textContent =
    `Round weighted score: Deep Q ${roundWinner.deepQScore} pts, Traditional RPL ${roundWinner.rplScore} pts. Metric wins this round: Deep Q ${roundWinner.deepQWins}, RPL ${roundWinner.rplWins}, ties ${roundWinner.tieWins}.`;
}

function renderComparisonTable() {
  const tbody = document.getElementById("comparisonTable");
  const comparison = state.comparison;
  if (!comparison) {
    tbody.innerHTML = "";
    return;
  }
  const rows = [
    ["Energy consumed", comparison.traditional.total_energy_consumed, comparison.deep_q.total_energy_consumed, "energy_consumption", "J"],
    ["Energy per delivered packet", comparison.traditional.energy_per_delivered_packet, comparison.deep_q.energy_per_delivered_packet, "energy_per_delivered_packet", "J"],
    ["Avg remaining energy", comparison.traditional.average_remaining_energy, comparison.deep_q.average_remaining_energy, "average_remaining_energy", "J"],
    ["Alive nodes", comparison.traditional.alive_nodes, comparison.deep_q.alive_nodes, "alive_nodes", ""],
    ["Delivered packets", comparison.traditional.delivered_packets, comparison.deep_q.delivered_packets, "delivered_packets", ""],
    ["Packet delivery ratio", comparison.traditional.packet_delivery_ratio, comparison.deep_q.packet_delivery_ratio, "packet_delivery_ratio", "%"],
    ["Average delay", comparison.traditional.average_delay, comparison.deep_q.average_delay, "average_delay", ""],
    ["Network lifetime", comparison.traditional.network_lifetime, comparison.deep_q.network_lifetime, "network_lifetime", "rounds"],
    ["First node death", comparison.traditional.first_node_death_round, comparison.deep_q.first_node_death_round, "first_node_death_round", "round"],
  ];

  tbody.innerHTML = rows
    .map(([label, rpl, deepQ, winnerKey, unit]) => {
      const winner = comparison.winner_by_metric[winnerKey] || "Tie";
      const weight = comparison.metric_weights?.[winnerKey] || 1;
      return `
        <tr>
          <td>${escapeHtml(label)}</td>
          <td><span class="weight-pill">${weight}</span></td>
          <td>${formatComparisonValue(label, rpl, unit)}</td>
          <td>${formatComparisonValue(label, deepQ, unit)}</td>
          <td>
            <span class="winner-badge ${winnerClass(winner)}">${escapeHtml(winner)}</span>
          </td>
        </tr>
      `;
    })
    .join("");
}

function winnerClass(winner) {
  if (winner === "Deep Q RL-RPL") return "deep-q";
  if (winner === "Traditional RPL") return "rpl";
  return "tie";
}

function formatComparisonValue(label, value, unit) {
  if (value === null || value === undefined) return "--";
  if (label === "Packet delivery ratio") return percent(value);
  const numeric = Number(value);
  if (Number.isNaN(numeric)) return escapeHtml(value);
  const formatted = Number.isInteger(numeric) ? String(numeric) : numeric.toFixed(3);
  return `${formatted}${unit ? ` ${unit}` : ""}`;
}

function renderDecisionTable() {
  const tbody = document.getElementById("decisionTable");
  const selectedRound = roundFilter.value === "all" ? String(snapshotFor("deep_q")?.round || "") : roundFilter.value;
  const term = searchInput.value.trim().toLowerCase();
  const rows = state.decisions
    .filter((row) => row.round === selectedRound)
    .filter((row) => !term || `${row.current_node} ${row.selected_parent} ${row.decision_rule}`.toLowerCase().includes(term))
    .slice(0, 80);

  tbody.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${escapeHtml(row.round)}</td>
          <td>${escapeHtml(row.current_node)}</td>
          <td>${escapeHtml(row.selected_parent)}</td>
          <td>${escapeHtml(row.selection_mode || "")}</td>
          <td title="${escapeHtml(row.equations || row.candidate_details)}">${escapeHtml(row.decision_rule)}</td>
          <td>${formatNumber(row.selection_value, 3)}</td>
          <td>${formatNumber(row.reward, 3)}</td>
          <td class="${row.delivered === "1" ? "yes" : "no"}">${row.delivered === "1" ? "Yes" : "No"}</td>
        </tr>
      `,
    )
    .join("");
}

function updateFocus() {
  const round = String(snapshotFor("deep_q")?.round || "");
  const row = state.decisions.find((item) => item.round === round);
  if (!row) return;
  document.getElementById("focusTitle").textContent = `Deep Q node ${row.current_node} selected parent ${row.selected_parent}`;
  document.getElementById("focusBody").textContent =
    `${row.decision_rule}. Selection=${formatNumber(row.selection_value, 3)}, reward=${formatNumber(row.reward, 3)}.`;
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
  state.frame = Math.max(0, Math.min(roundCount() - 1, state.frame + delta));
  render();
}

playButton.addEventListener("click", () => {
  state.playing = !state.playing;
  playButton.textContent = state.playing ? "Pause" : "Play";
  if (state.playing) {
    state.timer = setInterval(() => {
      state.frame = (state.frame + 1) % roundCount();
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
roundFilter.addEventListener("change", renderDecisionTable);

boot();
