const canvas = document.getElementById("pad");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clear");
const results = document.getElementById("results");
const statusEl = document.getElementById("status");

ctx.lineWidth = 14;
ctx.lineCap = "round";
ctx.lineJoin = "round";
ctx.strokeStyle = "#111";

let drawing = false;
let lastX = 0, lastY = 0;
let dirty = false;
let pollTimer = null;
let inflight = false;

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
  const y = (e.touches ? e.touches[0].clientY : e.clientY) - rect.top;
  return [x * (canvas.width / rect.width), y * (canvas.height / rect.height)];
}

function startDraw(e) {
  e.preventDefault();
  drawing = true;
  [lastX, lastY] = getPos(e);
}

function moveDraw(e) {
  if (!drawing) return;
  e.preventDefault();
  const [x, y] = getPos(e);
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();
  [lastX, lastY] = [x, y];
  dirty = true;
}

function endDraw() {
  drawing = false;
}

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", moveDraw);
window.addEventListener("mouseup", endDraw);
canvas.addEventListener("touchstart", startDraw);
canvas.addEventListener("touchmove", moveDraw);
canvas.addEventListener("touchend", endDraw);

clearBtn.addEventListener("click", () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  results.innerHTML = "";
  dirty = false;
  statusEl.textContent = "ready";
});

function isBlank() {
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  // Canvas starts transparent; any drawn pixel will have alpha > 0.
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] !== 0) return false;
  }
  return true;
}

async function predict() {
  if (inflight) return;
  if (isBlank()) {
    results.innerHTML = "";
    statusEl.textContent = "ready";
    return;
  }
  inflight = true;
  statusEl.textContent = "thinking...";
  try {
    const dataUrl = canvas.toDataURL("image/png");
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl }),
    });
    const json = await res.json();
    render(json.predictions);
    statusEl.textContent = "updated";
  } catch (err) {
    statusEl.textContent = "error: " + err.message;
  } finally {
    inflight = false;
  }
}

function render(preds) {
  results.innerHTML = "";
  for (const p of preds) {
    const row = document.createElement("div");
    row.className = "row";
    const pct = Math.round(p.prob * 100);
    row.innerHTML = `
      <span class="name">${p.label}</span>
      <span class="bar"><span style="width:${pct}%"></span></span>
      <span class="pct">${pct}%</span>
    `;
    results.appendChild(row);
  }
}

setInterval(() => {
  if (dirty) {
    dirty = false;
    predict();
  }
}, 250);
