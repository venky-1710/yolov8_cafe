/* ═══════════════════════════════════════════════════════════
   Cafe Occupancy Monitor — Real-Time Dashboard Logic
   Video: JS snapshot polling onto <canvas> (works in all browsers)
   ═══════════════════════════════════════════════════════════ */

// ── State ────────────────────────────────────────────────
let currentSource   = "Webcam";
let trendChart      = null;
let streamActive    = false;
let streamFailCount = 0;
const MAX_CHART_POINTS = 20;
const chartLabels   = [];
const chartData     = [];

// ── DOM refs ─────────────────────────────────────────────
const canvas         = document.getElementById("videoCanvas");
const ctx            = canvas.getContext("2d");
const elStreamLoad   = document.getElementById("streamLoading");
const elCount        = document.getElementById("kpiCount");
const elMax          = document.getElementById("kpiMax");
const elPct          = document.getElementById("kpiPct");
const elStatus       = document.getElementById("kpiStatus");
const elOverlay      = document.getElementById("overlayCount");
const elStatusChip   = document.getElementById("statusChip");
const elLiveBadge    = document.getElementById("liveBadge");
const elPulseDot     = document.getElementById("pulseDot");
const elMeterArc     = document.getElementById("meterArc");
const elMeterBig     = document.getElementById("meterBig");
const elMeterLabel   = document.getElementById("meterLabel");
const elGradStop1    = document.getElementById("gradStop1");
const elGradStop2    = document.getElementById("gradStop2");
const elFooterSource = document.getElementById("footerSource");
const elFooterTime   = document.getElementById("footerTime");
const elToast        = document.getElementById("toast");
const elSettingsMsg  = document.getElementById("settingsMsg");
const elProgressFill = document.getElementById("progressFill");
const elProgressLbl  = document.getElementById("progressLabel");
const elUploadProg   = document.getElementById("uploadProgress");

// ── Canvas sizing ─────────────────────────────────────────
function resizeCanvas() {
  const wrapper = canvas.parentElement;
  canvas.width  = wrapper.clientWidth;
  canvas.height = wrapper.clientHeight;
}
resizeCanvas();
window.addEventListener("resize", resizeCanvas);

// ── Video snapshot polling ────────────────────────────────
// Uses an off-screen Image object to fetch each JPEG frame,
// then draws it onto the canvas. This avoids all MJPEG browser issues.
const frameImg = new Image();
let framePending = false;

function fetchFrame() {
  if (framePending) return;   // don't pile up requests
  framePending = true;

  frameImg.onload = () => {
    framePending = false;
    streamFailCount = 0;

    // Hide loading overlay on first successful frame
    if (!streamActive) {
      streamActive = true;
      elStreamLoad.classList.add("hidden");
      setTimeout(() => { elStreamLoad.style.display = "none"; }, 450);
    }

    // Draw frame scaled to canvas
    ctx.drawImage(frameImg, 0, 0, canvas.width, canvas.height);

    // Release object URL to avoid memory leak
    URL.revokeObjectURL(frameImg.src);
  };

  frameImg.onerror = () => {
    framePending = false;
    streamFailCount++;
    if (streamFailCount > 10) {
      // Show reconnecting message on canvas
      ctx.fillStyle = "rgba(10,8,25,0.85)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#8888aa";
      ctx.font = "16px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Reconnecting to camera…", canvas.width / 2, canvas.height / 2);
    }
    URL.revokeObjectURL(frameImg.src);
  };

  // Fetch the snapshot as a blob, create an object URL
  fetch("/video_snapshot?t=" + Date.now())
    .then(r => {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.blob();
    })
    .then(blob => {
      frameImg.src = URL.createObjectURL(blob);
    })
    .catch(() => {
      framePending = false;
      streamFailCount++;
    });
}

// Poll at ~25 fps (40ms interval) — smooth display, low CPU
setInterval(fetchFrame, 40);
fetchFrame();   // kick off immediately

// ── Animated counter ─────────────────────────────────────
function animateValue(el, to, duration = 400) {
  const raw  = el.textContent.replace(/[^0-9]/g, "");
  const from = parseInt(raw) || 0;
  if (from === to) return;
  const start = performance.now();
  function step(now) {
    const p    = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - p, 3);
    el.textContent = Math.round(from + (to - from) * ease);
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ── Circular meter ───────────────────────────────────────
const CIRCUMFERENCE = 2 * Math.PI * 80;   // r=80 → 502.65

function updateMeter(pct) {
  const clamped = Math.max(0, Math.min(100, pct));
  const offset  = CIRCUMFERENCE * (1 - clamped / 100);
  elMeterArc.style.strokeDashoffset = offset;
  elMeterBig.textContent   = clamped + "%";
  elMeterLabel.textContent = clamped + "%";

  if (clamped < 60) {
    elGradStop1.setAttribute("stop-color", "#00e676");
    elGradStop2.setAttribute("stop-color", "#00bcd4");
  } else if (clamped < 85) {
    elGradStop1.setAttribute("stop-color", "#ff9800");
    elGradStop2.setAttribute("stop-color", "#ffb74d");
  } else {
    elGradStop1.setAttribute("stop-color", "#f44336");
    elGradStop2.setAttribute("stop-color", "#e91e63");
  }
}

// ── Status helpers ───────────────────────────────────────
function applyStatus(status) {
  const s   = status.toUpperCase();
  const cls = s === "SAFE" ? "safe" : s === "WARNING" ? "warning" : "danger";

  elStatusChip.textContent = s;
  elStatusChip.className   = "status-chip " + cls;

  elLiveBadge.className = "live-badge " + cls;
  elPulseDot.style.background =
    cls === "safe" ? "var(--safe)" : cls === "warning" ? "var(--warning)" : "var(--danger)";

  elStatus.textContent = s;
  elStatus.className   = "kpi-value status-value " + cls;
}

// ── Clock ────────────────────────────────────────────────
function updateClock() {
  elFooterTime.textContent = new Date().toLocaleTimeString();
}
setInterval(updateClock, 1000);
updateClock();

// ── Fetch occupancy (every 2 s) ──────────────────────────
async function fetchOccupancy() {
  try {
    const res  = await fetch("/api/occupancy");
    if (!res.ok) return;
    const data = await res.json();

    animateValue(elCount,   data.current_count);
    animateValue(elOverlay, data.current_count);
    elMax.textContent = data.max_capacity;

    // Update % KPI (keep the % unit span)
    elPct.innerHTML = data.occupancy_percent + '<span class="kpi-unit">%</span>';

    updateMeter(data.occupancy_percent);
    applyStatus(data.status);

  } catch (e) {
    console.warn("[Occupancy] fetch error:", e);
  }
}

// ── Chart.js trend (every 5 s) ───────────────────────────
function initChart() {
  const ctx2 = document.getElementById("trendChart").getContext("2d");
  trendChart = new Chart(ctx2, {
    type: "line",
    data: {
      labels:   chartLabels,
      datasets: [{
        label:           "Occupancy %",
        data:            chartData,
        borderColor:     "rgba(124,58,237,0.9)",
        backgroundColor: "rgba(124,58,237,0.12)",
        borderWidth:     2.5,
        pointRadius:     3,
        pointBackgroundColor: "rgba(124,58,237,1)",
        fill:            true,
        tension:         0.4,
      }],
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      animation:           { duration: 500 },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(10,10,30,0.9)",
          borderColor:     "rgba(124,58,237,0.4)",
          borderWidth:     1,
          callbacks: { label: c => ` ${c.parsed.y}% occupancy` },
        },
      },
      scales: {
        x: {
          ticks: { color: "#8888aa", font: { size: 10 } },
          grid:  { color: "rgba(255,255,255,0.04)" },
        },
        y: {
          min:   0,
          max:   100,
          ticks: { color: "#8888aa", font: { size: 10 }, callback: v => v + "%" },
          grid:  { color: "rgba(255,255,255,0.06)" },
        },
      },
    },
  });
}

async function fetchHistory() {
  try {
    const res  = await fetch("/api/history?limit=20");
    if (!res.ok) return;
    const data = await res.json();
    if (!data.length) return;

    chartLabels.length = 0;
    chartData.length   = 0;
    data.forEach(r => {
      const t = new Date(r.timestamp);
      chartLabels.push(t.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }));
      chartData.push(r.occupancy_percent);
    });
    trendChart.update();
  } catch (e) {
    console.warn("[History] fetch error:", e);
  }
}

// ── Settings — update capacity ────────────────────────────
async function updateCapacity() {
  const val = parseInt(document.getElementById("capacityInput").value);
  if (!val || val < 1) { showSettingsMsg("Enter a valid capacity.", "err"); return; }
  try {
    const res  = await fetch("/api/settings", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ max_capacity: val }),
    });
    const data = await res.json();
    if (data.success) {
      showSettingsMsg(`✓ Capacity updated to ${val}`, "ok");
      showToast(`Max capacity set to ${val}`, "success");
    } else {
      showSettingsMsg(data.error || "Error", "err");
    }
  } catch (e) { showSettingsMsg("Network error.", "err"); }
}

// ── Settings — switch source ──────────────────────────────
async function switchSource() {
  const src = document.getElementById("rtspInput").value.trim();
  if (!src) { showSettingsMsg("Enter a source URL or index.", "err"); return; }
  try {
    const res  = await fetch("/api/source", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ source: src }),
    });
    const data = await res.json();
    if (data.success) {
      currentSource = src === "0" ? "Webcam" : src;
      elFooterSource.textContent = "Source: " + currentSource;
      resetStream();
      showSettingsMsg("✓ Source switched.", "ok");
      showToast("Video source switched.", "success");
    } else {
      showSettingsMsg(data.error || "Cannot open source.", "err");
    }
  } catch (e) { showSettingsMsg("Network error.", "err"); }
}

// ── Switch back to webcam ─────────────────────────────────
async function switchToWebcam() {
  try {
    const res  = await fetch("/api/source", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ source: "0" }),
    });
    const data = await res.json();
    if (data.success) {
      currentSource = "Webcam";
      elFooterSource.textContent = "Source: Webcam";
      resetStream();
      showToast("Switched to webcam.", "success");
    } else {
      showToast(data.error || "Cannot open webcam.", "error");
    }
  } catch (e) { showToast("Network error.", "error"); }
}

// ── Reset stream state (after source switch) ──────────────
function resetStream() {
  streamActive    = false;
  streamFailCount = 0;
  elStreamLoad.style.display = "flex";
  elStreamLoad.classList.remove("hidden");
}

// ── Upload video file ─────────────────────────────────────
async function uploadVideo(input) {
  const file = input.files[0];
  if (!file) return;

  elUploadProg.style.display = "flex";
  elProgressFill.style.width = "0%";
  elProgressLbl.textContent  = `Uploading ${file.name}…`;

  const formData = new FormData();
  formData.append("video", file);

  try {
    await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/upload");

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          const pct = Math.round((e.loaded / e.total) * 100);
          elProgressFill.style.width = pct + "%";
          elProgressLbl.textContent  = `Uploading… ${pct}%`;
        }
      };

      xhr.onload = () => {
        if (xhr.status === 200) {
          const data = JSON.parse(xhr.responseText);
          if (data.success) {
            elProgressFill.style.width = "100%";
            elProgressLbl.textContent  = "✓ Processing video…";
            currentSource = file.name;
            elFooterSource.textContent = "Source: " + file.name;
            resetStream();
            setTimeout(() => { elUploadProg.style.display = "none"; }, 1200);
            showToast(`✓ Now detecting in: ${file.name}`, "success");
            resolve();
          } else {
            reject(new Error(data.error || "Upload failed"));
          }
        } else {
          reject(new Error("Server error: " + xhr.status));
        }
      };
      xhr.onerror = () => reject(new Error("Network error"));
      xhr.send(formData);
    });
  } catch (err) {
    elUploadProg.style.display = "none";
    showToast("Upload failed: " + err.message, "error");
  }
  input.value = "";
}

// ── Toast notification ────────────────────────────────────
let toastTimer = null;
function showToast(msg, type = "success") {
  elToast.textContent = msg;
  elToast.className   = "toast show " + type;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { elToast.className = "toast"; }, 3500);
}

// ── Settings message ──────────────────────────────────────
let msgTimer = null;
function showSettingsMsg(msg, cls) {
  elSettingsMsg.textContent = msg;
  elSettingsMsg.className   = "settings-msg " + cls;
  clearTimeout(msgTimer);
  msgTimer = setTimeout(() => { elSettingsMsg.textContent = ""; }, 4000);
}

// ── Boot ──────────────────────────────────────────────────
initChart();
fetchOccupancy();
fetchHistory();
setInterval(fetchOccupancy, 2000);
setInterval(fetchHistory,   5000);
