// Nesne Tespit — YOLO11n (320px) tarayıcı içi gerçek zamanlı nesne tespiti
// ONNX Runtime Web: WebGPU (varsa) → WASM (int8 kuantize model) yedek yolu.
import { LABELS_EN, LABELS_TR } from "./labels.js";

const INPUT = 320;              // model giriş boyutu
const IOU_NMS = 0.45;
const MAX_DET = 50;
const MODEL_GPU = "model/yolo11n_320.onnx";       // fp32 — WebGPU
const MODEL_CPU = "model/yolo11n_320_int8.onnx";  // int8 — WASM (CPU)

const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("overlay"), ctx = canvas.getContext("2d");
const ui = {
  fps: $("fps"), infer: $("infer"), backend: $("backend"), count: $("count"),
  status: $("status"), barFill: $("barFill"), splash: $("splash"), startBtn: $("startBtn"),
  flipBtn: $("flipBtn"), langBtn: $("langBtn"), confBtn: $("confBtn"),
};

const state = {
  session: null, inputName: null, outputName: null,
  facing: "environment", stream: null, running: false, busy: false,
  lang: localGet("lang", "TR"), conf: Number(localGet("conf", "0.35")),
  tracks: [], nextId: 1, lastInferTs: 0,
  fpsFrames: 0, fpsT0: performance.now(), inferMs: 0,
  wakeLock: null,
};

// Ön işleme için tekrar kullanılan tamponlar (GC baskısını önler)
const prep = document.createElement("canvas");
prep.width = prep.height = INPUT;
const prepCtx = prep.getContext("2d", { willReadFrequently: true });
const inputData = new Float32Array(3 * INPUT * INPUT);

function localGet(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } }
function localSet(k, v) { try { localStorage.setItem(k, v); } catch {} }

// ---------------------------------------------------------------- model
async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    if (total) onProgress(Math.min(1, got / total));
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

async function loadModel() {
  ort.env.wasm.wasmPaths = new URL("vendor/", location.href).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
  ort.env.wasm.simd = true;

  const attempts = [];
  if ("gpu" in navigator) attempts.push({ ep: "webgpu", url: MODEL_GPU, label: "WebGPU" });
  attempts.push({ ep: "wasm", url: MODEL_CPU, label: "WASM int8" });

  let lastErr;
  for (const a of attempts) {
    try {
      ui.status.textContent = `Model yükleniyor (${a.label})…`;
      const bytes = await fetchWithProgress(a.url, (p) => (ui.barFill.style.width = `${Math.round(p * 90)}%`));
      const session = await ort.InferenceSession.create(bytes, {
        executionProviders: [a.ep],
        graphOptimizationLevel: "all",
      });
      // Isınma çalıştırması (ilk karedeki gecikmeyi önler)
      const warm = new ort.Tensor("float32", new Float32Array(3 * INPUT * INPUT), [1, 3, INPUT, INPUT]);
      await session.run({ [session.inputNames[0]]: warm });
      state.session = session;
      state.inputName = session.inputNames[0];
      state.outputName = session.outputNames[0];
      ui.backend.textContent = a.label;
      ui.barFill.style.width = "100%";
      return;
    } catch (e) {
      console.warn(`${a.label} başarısız:`, e);
      lastErr = e;
    }
  }
  throw lastErr;
}

// ---------------------------------------------------------------- kamera
async function startCamera() {
  stopCamera();
  const constraints = {
    audio: false,
    video: {
      facingMode: { ideal: state.facing },
      width: { ideal: 1280 }, height: { ideal: 720 },
      frameRate: { ideal: 30, max: 60 },
    },
  };
  state.stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = state.stream;
  const track = state.stream.getVideoTracks()[0];
  const actual = track.getSettings().facingMode;
  const isFront = actual ? actual === "user" : state.facing === "user";
  video.classList.toggle("mirror", isFront);
  state.mirror = isFront;
  await video.play();
  state.tracks = [];
}
function stopCamera() {
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  state.stream = null;
}

async function requestWakeLock() {
  try { state.wakeLock = await navigator.wakeLock?.request("screen"); } catch {}
}

// ---------------------------------------------------------------- ön/son işleme
function preprocess() {
  const vw = video.videoWidth, vh = video.videoHeight;
  const scale = Math.min(INPUT / vw, INPUT / vh);
  const nw = Math.round(vw * scale), nh = Math.round(vh * scale);
  const dx = (INPUT - nw) >> 1, dy = (INPUT - nh) >> 1;
  prepCtx.fillStyle = "#727272"; // YOLO letterbox dolgu rengi (114)
  prepCtx.fillRect(0, 0, INPUT, INPUT);
  prepCtx.drawImage(video, 0, 0, vw, vh, dx, dy, nw, nh);
  const px = prepCtx.getImageData(0, 0, INPUT, INPUT).data;
  const area = INPUT * INPUT;
  for (let i = 0, j = 0; i < area; i++, j += 4) {
    inputData[i] = px[j] / 255;
    inputData[i + area] = px[j + 1] / 255;
    inputData[i + 2 * area] = px[j + 2] / 255;
  }
  return { scale, dx, dy, vw, vh };
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return u > 0 ? inter / u : 0;
}

// Çıktı: [1, 84, N] → 4 kutu + 80 sınıf skoru
function postprocess(output, meta) {
  const data = output.data, dims = output.dims;
  const C = dims[1], N = dims[2], nc = C - 4;
  const cand = [];
  for (let i = 0; i < N; i++) {
    let best = 0, cls = -1;
    for (let c = 0; c < nc; c++) {
      const s = data[(4 + c) * N + i];
      if (s > best) { best = s; cls = c; }
    }
    if (best < state.conf) continue;
    const cx = data[i], cy = data[N + i], w = data[2 * N + i], h = data[3 * N + i];
    // letterbox → video piksel koordinatı
    const x1 = (cx - w / 2 - meta.dx) / meta.scale, y1 = (cy - h / 2 - meta.dy) / meta.scale;
    const x2 = (cx + w / 2 - meta.dx) / meta.scale, y2 = (cy + h / 2 - meta.dy) / meta.scale;
    cand.push({
      box: [Math.max(0, x1), Math.max(0, y1), Math.min(meta.vw, x2), Math.min(meta.vh, y2)],
      score: best, cls,
    });
  }
  cand.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const d of cand) {
    if (keep.length >= MAX_DET) break;
    if (keep.every((k) => k.cls !== d.cls || iou(k.box, d.box) < IOU_NMS)) keep.push(d);
  }
  return keep;
}

// ---------------------------------------------------------------- takip (tracking)
// IoU tabanlı basit takipçi + üstel yumuşatma + hız tahmini:
// iki çıkarım arasında kutular tahmini hızla ilerletilir → kamera hareketinde gecikmesiz his.
function updateTracks(dets, now) {
  const tracks = state.tracks;
  const used = new Set();
  const pairs = [];
  for (let ti = 0; ti < tracks.length; ti++)
    for (let di = 0; di < dets.length; di++) {
      if (tracks[ti].cls !== dets[di].cls) continue;
      const v = iou(predictBox(tracks[ti], now), dets[di].box);
      if (v > 0.2) pairs.push([v, ti, di]);
    }
  pairs.sort((a, b) => b[0] - a[0]);
  const matchedT = new Set();
  for (const [, ti, di] of pairs) {
    if (matchedT.has(ti) || used.has(di)) continue;
    matchedT.add(ti); used.add(di);
    const t = tracks[ti], d = dets[di];
    const dt = Math.max(1, now - t.ts);
    const a = 0.65; // yumuşatma (yüksek = daha hızlı tepki)
    const nb = t.box.map((v, k) => v + a * (d.box[k] - v));
    t.vel = t.vel.map((v, k) => 0.5 * v + 0.5 * ((nb[k] - t.box[k]) / dt));
    t.box = nb; t.ts = now; t.score = 0.5 * t.score + 0.5 * d.score;
    t.hits++; t.miss = 0;
  }
  for (let ti = 0; ti < tracks.length; ti++) if (!matchedT.has(ti)) tracks[ti].miss++;
  for (let di = 0; di < dets.length; di++) {
    if (used.has(di)) continue;
    const d = dets[di];
    tracks.push({ id: state.nextId++, cls: d.cls, box: d.box.slice(), vel: [0, 0, 0, 0], ts: now, score: d.score, hits: 1, miss: 0 });
  }
  state.tracks = tracks.filter((t) => t.miss <= 3);
}

function predictBox(t, now) {
  const dt = Math.min(120, now - t.ts); // en fazla 120 ms ileri tahmin
  return t.box.map((v, k) => v + t.vel[k] * dt);
}

// ---------------------------------------------------------------- döngüler
async function inferLoop() {
  if (!state.running) return;
  if (!state.busy && video.readyState >= 2 && video.videoWidth) {
    state.busy = true;
    try {
      const t0 = performance.now();
      const meta = preprocess();
      const tensor = new ort.Tensor("float32", inputData, [1, 3, INPUT, INPUT]);
      const out = await state.session.run({ [state.inputName]: tensor });
      const dets = postprocess(out[state.outputName], meta);
      const now = performance.now();
      updateTracks(dets, now);
      state.inferMs = state.inferMs ? state.inferMs * 0.8 + (now - t0) * 0.2 : now - t0;
      state.fpsFrames++;
    } catch (e) {
      console.error(e);
    } finally {
      state.busy = false;
    }
  }
  // Bir sonraki kareyi bekle (destekleniyorsa video karesine kilitlen)
  if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(() => inferLoop());
  else requestAnimationFrame(() => inferLoop());
}

function labelFor(cls) {
  if (state.lang === "EN") return LABELS_EN[cls];
  if (state.lang === "TR") return LABELS_TR[cls];
  return `${LABELS_TR[cls]} / ${LABELS_EN[cls]}`;
}
const colorFor = (cls) => `hsl(${(cls * 137.508) % 360} 95% 55%)`;

function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  return dpr;
}

function renderLoop() {
  requestAnimationFrame(renderLoop);
  const dpr = resizeCanvas();
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!state.running || !video.videoWidth) return;

  // object-fit: cover eşlemesi (video → ekran)
  const vw = video.videoWidth, vh = video.videoHeight;
  const s = Math.max(W / vw, H / vh);
  const ox = (W - vw * s) / 2, oy = (H - vh * s) / 2;
  const now = performance.now();
  const fontPx = Math.round(15 * dpr);
  ctx.font = `600 ${fontPx}px -apple-system, Roboto, sans-serif`;
  ctx.textBaseline = "top";
  ctx.lineWidth = Math.max(2, 2.5 * dpr);

  let shown = 0;
  for (const t of state.tracks) {
    if (t.hits < 2 && t.miss === 0 && t.score < state.conf + 0.15) continue; // tek karelik titremeleri gizle
    const b = predictBox(t, now);
    let x1 = b[0] * s + ox, x2 = b[2] * s + ox;
    const y1 = b[1] * s + oy, y2 = b[3] * s + oy;
    if (state.mirror) { const m1 = W - x2, m2 = W - x1; x1 = m1; x2 = m2; }
    const alpha = t.miss ? Math.max(0.25, 1 - t.miss * 0.25) : 1;
    const col = colorFor(t.cls);
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = col;
    roundRect(x1, y1, x2 - x1, y2 - y1, 6 * dpr);
    ctx.stroke();

    const text = `${labelFor(t.cls)} ${Math.round(t.score * 100)}%`;
    const tw = ctx.measureText(text).width + 10 * dpr, th = fontPx + 8 * dpr;
    let ty = y1 - th; if (ty < 0) ty = y1;
    let tx = Math.min(Math.max(0, x1), W - tw);
    ctx.fillStyle = col;
    ctx.fillRect(tx, ty, tw, th);
    ctx.fillStyle = "#000";
    ctx.fillText(text, tx + 5 * dpr, ty + 4 * dpr);
    shown++;
  }
  ctx.globalAlpha = 1;

  ui.count.textContent = `${shown} nesne`;
  const el = now - state.fpsT0;
  if (el > 500) {
    ui.fps.textContent = `${Math.round((state.fpsFrames * 1000) / el)} FPS`;
    ui.infer.textContent = `${Math.round(state.inferMs)} ms`;
    state.fpsFrames = 0; state.fpsT0 = now;
  }
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
  else ctx.rect(x, y, w, h);
}

// ---------------------------------------------------------------- UI
function updateButtons() {
  ui.langBtn.textContent = state.lang === "BOTH" ? "TR/EN" : state.lang;
  ui.confBtn.textContent = `≥ ${Math.round(state.conf * 100)}%`;
}

ui.langBtn.onclick = () => {
  state.lang = { TR: "EN", EN: "BOTH", BOTH: "TR" }[state.lang] || "TR";
  localSet("lang", state.lang); updateButtons();
};
ui.confBtn.onclick = () => {
  const steps = [0.25, 0.35, 0.5, 0.65];
  state.conf = steps[(steps.indexOf(state.conf) + 1) % steps.length];
  localSet("conf", String(state.conf)); updateButtons();
};
ui.flipBtn.onclick = async () => {
  state.facing = state.facing === "environment" ? "user" : "environment";
  try { await startCamera(); } catch (e) { alert("Kamera değiştirilemedi: " + e.message); }
};

async function begin() {
  try {
    ui.startBtn.hidden = true;
    ui.status.textContent = "Kamera açılıyor…";
    await startCamera();
    ui.splash.classList.add("hide");
    state.running = true;
    requestWakeLock();
    inferLoop();
  } catch (e) {
    console.error(e);
    ui.status.textContent = "Kamera izni gerekli. Lütfen izin verip tekrar deneyin.";
    ui.startBtn.hidden = false;
  }
}
ui.startBtn.onclick = begin;

document.addEventListener("visibilitychange", async () => {
  if (document.hidden) { state.running = false; stopCamera(); }
  else if (state.session && ui.splash.classList.contains("hide")) {
    try { await startCamera(); state.running = true; requestWakeLock(); inferLoop(); } catch {}
  }
});

// ---------------------------------------------------------------- başlat
(async function main() {
  updateButtons();
  renderLoop();
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("service-worker.js").catch(() => {});
  if (!navigator.mediaDevices?.getUserMedia) {
    ui.status.textContent = "Bu tarayıcı kamerayı desteklemiyor (HTTPS gerekli).";
    return;
  }
  try {
    await loadModel();
  } catch (e) {
    ui.status.textContent = "Model yüklenemedi: " + e.message;
    return;
  }
  ui.status.textContent = "Hazır";
  await begin(); // Otomatik başlat; izin reddedilirse buton gösterilir
})();
