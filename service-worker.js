// Çevrimdışı çalışma: uygulama kabuğu + model + ONNX Runtime önbelleğe alınır.
const CACHE = "nesne-tespit-v1";
const ASSETS = [
  "./", "index.html", "style.css", "app.js", "labels.js", "manifest.json",
  "icons/icon-192.png", "icons/icon-512.png",
  "vendor/ort.webgpu.min.js", "vendor/ort-wasm-simd-threaded.jsep.mjs", "vendor/ort-wasm-simd-threaded.jsep.wasm",
  "model/yolo11n_320.onnx", "model/yolo11n_320_int8.onnx",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

// Önce önbellek, yoksa ağ (ve sonucu önbelleğe yaz)
self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  e.respondWith(
    caches.match(e.request, { ignoreSearch: true }).then((hit) =>
      hit || fetch(e.request).then((res) => {
        if (res.ok && new URL(e.request.url).origin === location.origin) {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(e.request, copy));
        }
        return res;
      })
    )
  );
});
