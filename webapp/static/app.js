// Visionary Atelier: front end for the try-on server (server.py).
// Photo mode: upload a portrait, instant classical render + optional AI studio render.
// Live mode: webcam frames over a WebSocket, classical render at ~20 FPS.

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const MAX_UPLOAD_MB = 15;
const DEFAULTS = { width: 1.15, length: 1.0, lighting: 0.3, shading: 0.45, feather: 2, sleeves: true, fit: true, tps: true, occlusion: true };

const state = {
  mode: "photo",
  view: "compare",
  garmentId: null,
  photo: null,                  // {id, url, width, height, confidence}
  results: { instant: null, studio: null },
  shown: "instant",
  stages: null,
  busy: false,
  ws: null,
  stream: null,
};

// ================================================================ feedback
function setStatus(text, kind = "") {
  $("#status-text").textContent = text;
  $("#status").className = kind ? `status is-${kind}` : "status";
}

let toastTimer;
function toast(message) {
  $("#toast-text").textContent = message;
  $("#toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ($("#toast").hidden = true), 9000);
  setStatus("Needs attention", "error");
}
$("#toast-close").addEventListener("click", () => ($("#toast").hidden = true));

function busy(text) {
  state.busy = Boolean(text);
  $("#busy").hidden = !text;
  if (text) { $("#busy-text").textContent = text; setStatus(text, "busy"); }
  updateControls();
}

async function api(path, options = {}) {
  let res;
  try {
    res = await fetch(path, options);
  } catch {
    throw new Error("The server is not reachable. Is it still running? (uvicorn server:app)");
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status}).`);
  return body;
}
const postJSON = (path, data) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

function checkImage(file) {
  if (!file) return false;
  if (!file.type.startsWith("image/")) { toast("That file is not an image. Use a JPG or PNG."); return false; }
  if (file.size > MAX_UPLOAD_MB * 1e6) { toast(`That image is over ${MAX_UPLOAD_MB} MB. Use a smaller one.`); return false; }
  return true;
}

function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

// ================================================================ options
function readOptions() {
  const o = {};
  $$("[data-opt]").forEach((el) => {
    const k = el.dataset.opt;
    o[k] = el.type === "checkbox" ? el.checked : parseFloat(el.value);
    const out = $(`#out-${k}`);
    if (out) out.textContent = k === "feather" ? `${el.value} px` : `${Math.round(el.value * 100)}%`;
  });
  return o;
}

const optionsChanged = debounce(() => {
  readOptions();
  if (state.mode === "photo") renderInstant();
  else sendLiveSettings();
}, 120);
$$("[data-opt]").forEach((el) => el.addEventListener("input", optionsChanged));

$("#btn-reset").addEventListener("click", () => {
  $$("[data-opt]").forEach((el) => {
    const v = DEFAULTS[el.dataset.opt];
    if (el.type === "checkbox") el.checked = v; else el.value = v;
  });
  optionsChanged();
});

// ================================================================ garments
async function loadGarments(selectId) {
  const { garments } = await api("/api/garments");
  state.provider = "Apple GPU";
  const list = $("#pieces");
  list.querySelectorAll("li.g").forEach((li) => li.remove());
  for (const g of garments) {
    const li = document.createElement("li");
    li.className = "g";
    const b = document.createElement("button");
    b.className = "piece";
    b.dataset.id = g.id;
    b.setAttribute("aria-pressed", "false");
    b.innerHTML = `<span class="frame"><img alt="" loading="lazy"></span><span class="piece-name"></span>`;
    b.querySelector("img").src = g.image;
    b.querySelector(".piece-name").textContent = g.name;
    b.title = g.name;
    b.addEventListener("click", () => selectGarment(g.id));
    li.append(b);
    list.append(li);
  }
  const pick = selectId || state.garmentId || garments[0]?.id;
  if (pick) selectGarment(pick);
}

function selectGarment(id) {
  if (state.garmentId === id) return;
  state.garmentId = id;
  $$(".piece[data-id]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.id === id)));
  state.results.studio = null;
  if (state.shown === "studio") showRender("instant");
  if (state.mode === "photo") renderInstant();
  else sendLiveSettings();
  updateControls();
}

async function addGarment(file) {
  if (!checkImage(file) || state.busy) return;
  busy("Cutting out the garment");
  try {
    const form = new FormData();
    form.append("file", file);
    const g = await api("/api/garments", { method: "POST", body: form });
    state.garmentId = null;
    await loadGarments(g.id);
    setStatus(`Added ${g.name}`);
  } catch (e) { toast(e.message); }
  finally { busy(null); $("#garment-file").value = ""; }
}
$("#garment-file").addEventListener("change", (e) => addGarment(e.target.files[0]));
dropZone($("#garment-drop"), addGarment);

// ================================================================ photo mode
async function uploadPhoto(file) {
  if (!checkImage(file) || state.busy) return;
  busy("Reading your pose");
  try {
    const form = new FormData();
    form.append("file", file);
    const p = await api("/api/photos", { method: "POST", body: form });
    if (state.photo?.url) URL.revokeObjectURL(state.photo.url);
    state.photo = { ...p, url: URL.createObjectURL(file) };
    state.results = { instant: null, studio: null };
    state.shown = "instant";
    $("#img-before").src = state.photo.url;
    $("#you-thumb").src = state.photo.url;
    $("#you-text").textContent = p.personFound ? "Portrait ready" : "No person found";
    $("#m-conf").textContent = p.personFound ? `${Math.round(p.confidence * 100)}%` : "—";
    $("#m-fps").textContent = "—";
    $("#m-pose").textContent = "—";
    if (!p.personFound) toast("We couldn't find a person in that photo. Use one facing the camera with shoulders and hips visible.");
    showPhotoUI();
    await renderInstant();
  } catch (e) { toast(e.message); }
  finally { busy(null); $("#photo-file").value = ""; }
}

let instantSeq = 0;
async function renderInstant() {
  if (!state.photo || !state.garmentId) return;
  const seq = ++instantSeq;
  try {
    const r = await postJSON("/api/tryon", { photoId: state.photo.id, garmentId: state.garmentId, options: readOptions() });
    if (seq !== instantSeq) return;                 // a newer render is coming
    state.results.instant = r.image;
    state.stages = r.stages;
    if (state.shown === "instant") $("#img-after").src = r.image;
    $$("[data-stage]").forEach((img) => {
      const src = r.stages?.[img.dataset.stage];
      img.parentElement.hidden = !src;
      if (src) img.src = src;
    });
    $("#m-render").textContent = r.timing?.total ? `${Math.round(r.timing.total)} ms` : "—";
    if (r.ok) setStatus("Instant render ready");
    else toast(r.reason);
  } catch (e) {
    if (e.message.includes("expired")) { state.photo = null; showPhotoUI(); }
    toast(e.message);
  }
  updateControls();
}

// Studio render: runs locally (our geometry + CatVTON on the Apple GPU) as a
// server job; we poll it for real progress.
async function renderStudio() {
  if (state.busy) return;
  if (state.mode === "live") {                     // take the current frame first
    if (!(await snapshot())) return;
  }
  if (!state.photo || !state.garmentId) return;
  state.busy = true;
  updateControls();
  $("#tailoring").hidden = false;
  $("#tailor-step").textContent = "Preparing";
  setStatus("Studio render in progress", "busy");
  try {
    const { jobId } = await postJSON("/api/studio", {
      photoId: state.photo.id, garmentId: state.garmentId,
    });
    let job;
    for (;;) {
      await new Promise((r) => setTimeout(r, 700));
      job = await api(`/api/studio/${jobId}`);
      $("#tailor-step").textContent = job.stage;
      $("#tailor-time").textContent = Math.round(job.elapsed);
      $("#tailor-bar").style.width = `${Math.round(job.progress * 100)}%`;
      if (job.status !== "running") break;
    }
    if (job.status === "error") throw new Error(job.error);
    state.results.studio = job.image;
    showRender("studio");
    setView("compare");
    setStatus(`Studio render ready in ${job.seconds} s`);
  } catch (e) { toast(e.message); }
  finally {
    $("#tailoring").hidden = true;
    $("#tailor-bar").style.width = "0";
    state.busy = false;
    updateControls();
  }
}
$("#btn-hd").addEventListener("click", renderStudio);

function showRender(which) {
  state.shown = which;
  $("#cta-title").textContent = which === "studio" ? "Studio render" : "Instant render";
  $("#cta-text").textContent = which === "studio"
    ? `Our pose, homography warp and mask place the garment; CatVTON renders it locally on the ${state.provider}. Outside the garment area the photo is untouched.`
    : "Classical pipeline, updates as you adjust. For a photorealistic fitting, render it in the studio.";
  $("#after-label").textContent = which === "studio" ? "Studio · AI" : "Instant";
  const img = state.results[which];
  if (img) $("#img-after").src = img;
}

function showPhotoUI() {
  const has = Boolean(state.photo);
  $("#photo-drop").hidden = has || state.mode !== "photo";
  $("#views").hidden = !has || state.mode !== "photo";
  $("#you-photo").hidden = !has;
  $("#you-empty").hidden = has || state.mode === "live";
  setView(state.view);
  updateControls();
}

function setView(view) {
  state.view = view;
  $$("[data-view]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.view === view)));
  const photoVisible = state.mode === "photo" && Boolean(state.photo);
  $("#compare").hidden = !photoVisible || view === "pipeline";
  $("#pipeline").hidden = !photoVisible || view !== "pipeline";
  $("#compare").classList.toggle("result-only", view === "result");
  if (view === "pipeline" && state.shown === "studio") showRender("instant");
}
$$("[data-view]").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));

$("#compare-range").addEventListener("input", (e) => $("#compare").style.setProperty("--pos", `${e.target.value}%`));
$("#photo-file").addEventListener("change", (e) => uploadPhoto(e.target.files[0]));
$("#btn-replace").addEventListener("click", () => $("#photo-file").click());
dropZone($("#photo-drop"), uploadPhoto);
dropZone($("#stage"), (f) => state.mode === "photo" && uploadPhoto(f));

$("#btn-save").addEventListener("click", () => {
  const url = state.mode === "live" ? $("#live-canvas").toDataURL("image/jpeg", 0.92) : state.results[state.shown];
  if (!url) return;
  const a = document.createElement("a");
  a.href = url;
  a.download = `visionary-${state.mode === "live" ? "live" : state.shown}-${Date.now()}.jpg`;
  a.click();
});

// ================================================================ live mode
const video = Object.assign(document.createElement("video"), { playsInline: true, muted: true });
const grab = document.createElement("canvas");
let frames = 0, fpsT = performance.now();

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    toast("This browser cannot use the camera here. Open the app at http://localhost:8000 in Chrome or Safari.");
    return;
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, audio: false,
    });
  } catch (e) {
    toast(e.name === "NotAllowedError" ? "Camera access was blocked. Allow the camera for this page in the browser settings."
      : e.name === "NotFoundError" ? "No camera was found. Use Photo mode and upload a picture instead."
      : e.name === "NotReadableError" ? "The camera is in use by another app. Close it and try again."
      : `Could not start the camera: ${e.message}`);
    return;
  }
  video.srcObject = state.stream;
  await video.play();
  for (const c of [grab, $("#live-canvas"), $("#live-overlay")]) {
    c.width = video.videoWidth;
    c.height = video.videoHeight;
  }
  $("#live-welcome").hidden = true;
  $("#live-controls").hidden = false;

  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/live`);
  ws.binaryType = "blob";
  state.ws = ws;
  let info = null;
  ws.onopen = () => { sendLiveSettings(); sendFrame(); setStatus("Live"); };
  ws.onmessage = async (ev) => {
    if (typeof ev.data === "string") { info = JSON.parse(ev.data); return; }
    const bmp = await createImageBitmap(ev.data);
    $("#live-canvas").getContext("2d").drawImage(bmp, 0, 0);
    liveStats(info);
    sendFrame();                                   // one frame in flight at a time
  };
  ws.onerror = () => toast("Lost the connection to the server.");
  ws.onclose = () => { if (state.stream) { stopCamera(); toast("The live connection closed. Start the camera again."); } };
  updateControls();
}

function sendFrame() {
  const ws = state.ws;
  if (!ws || ws.readyState !== WebSocket.OPEN || !state.stream) return;
  grab.getContext("2d").drawImage(video, 0, 0);
  grab.toBlob((b) => b && ws.readyState === WebSocket.OPEN && ws.send(b), "image/jpeg", 0.75);
}

function sendLiveSettings() {
  if (state.ws?.readyState === WebSocket.OPEN)
    state.ws.send(JSON.stringify({ garmentId: state.garmentId, options: readOptions() }));
}

let lastReason = "";
function liveStats(info) {
  frames++;
  const now = performance.now();
  if (now - fpsT > 1000) {
    $("#m-fps").textContent = (frames * 1000 / (now - fpsT)).toFixed(1);
    frames = 0;
    fpsT = now;
  }
  if (!info) return;
  $("#m-conf").textContent = `${Math.round((info.confidence || 0) * 100)}%`;
  $("#m-pose").textContent = `${Math.round(info.poseMs)} ms`;
  $("#m-render").textContent = `${Math.round(info.serverMs)} ms`;
  const reason = info.ok ? "" : info.reason;
  if (reason !== lastReason) setStatus(reason || "Live", reason ? "error" : "");
  lastReason = reason;
  const ctx = $("#live-overlay").getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  if ($("#opt-points").checked) drawPoints(ctx, info.keypoints);
}

function drawPoints(ctx, kp) {
  if (!kp?.leftShoulder) return;
  const ok = (k) => kp[k] && kp[k][2] > 0.5;
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#a8834f";
  const quad = ["leftShoulder", "rightShoulder", "rightHip", "leftHip"];
  if (quad.every(ok)) {
    ctx.beginPath();
    quad.forEach((k, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, kp[k][0], kp[k][1]));
    ctx.closePath();
    ctx.stroke();
  }
  ctx.strokeStyle = "#fbfaf7";
  for (const [a, b] of [["leftShoulder", "leftElbow"], ["rightShoulder", "rightElbow"]]) {
    if (!ok(a) || !ok(b)) continue;
    ctx.beginPath(); ctx.moveTo(kp[a][0], kp[a][1]); ctx.lineTo(kp[b][0], kp[b][1]); ctx.stroke();
  }
  for (const v of Object.values(kp)) {
    if (v[2] < 0.5) continue;
    ctx.beginPath(); ctx.arc(v[0], v[1], 5, 0, Math.PI * 2);
    ctx.fillStyle = "#141414"; ctx.fill();
    ctx.strokeStyle = "#fbfaf7"; ctx.stroke();
  }
}

function stopCamera() {
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
  const ws = state.ws;
  state.ws = null;
  ws?.close();
  $("#live-welcome").hidden = false;
  $("#live-controls").hidden = true;
  $("#m-fps").textContent = "—";
  setStatus("Camera stopped");
  updateControls();
}

// Freeze the current (mirrored, as seen) frame and continue in Photo mode.
async function snapshot() {
  if (!state.stream) { toast("Start the camera first."); return false; }
  const c = document.createElement("canvas");
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  const ctx = c.getContext("2d");
  ctx.translate(c.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0);
  const blob = await new Promise((r) => c.toBlob(r, "image/jpeg", 0.95));
  stopCamera();
  setMode("photo");
  await uploadPhoto(new File([blob], "snapshot.jpg", { type: "image/jpeg" }));
  return Boolean(state.photo);
}

$("#btn-start").addEventListener("click", startCamera);
$("#btn-stop").addEventListener("click", stopCamera);
$("#btn-snapshot").addEventListener("click", snapshot);

// ================================================================ modes
function setMode(mode) {
  if (state.mode === mode) return;
  state.mode = mode;
  $$("[data-mode]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === mode)));
  $("#live").hidden = mode !== "live";
  if (mode === "photo" && state.stream) stopCamera();
  if (mode === "live") $("#you-empty").hidden = true;
  setStatus(mode === "live" ? "Camera off" : state.photo ? "Portrait ready" : "Ready");
  if (mode === "live") {
    $("#cta-title").textContent = "Live mirror";
    $("#cta-text").textContent = "The classical pipeline at camera speed. Studio render takes this frame and tailors it with AI.";
  } else showRender(state.shown);
  showPhotoUI();
}
$$("[data-mode]").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));

function updateControls() {
  const live = state.mode === "live";
  $("#btn-hd").disabled = state.busy || !state.garmentId || (live ? !state.stream : !state.photo);
  $("#btn-save").disabled = live ? !state.stream : !state.results[state.shown];
}

// ================================================================ drag & drop
function dropZone(el, onFile) {
  el.addEventListener("dragover", (e) => { e.preventDefault(); el.classList.add("drag"); });
  el.addEventListener("dragleave", () => el.classList.remove("drag"));
  el.addEventListener("drop", (e) => {
    e.preventDefault();
    e.stopPropagation();
    el.classList.remove("drag");
    onFile(e.dataTransfer.files[0]);
  });
}

// ================================================================ start
readOptions();
showPhotoUI();
loadGarments().catch((e) => toast(e.message));
