// Çevrimdışı çalışma: uygulama kabuğu + ONNX Runtime kurulumda önbelleğe alınır.
// Modeller büyük olduğu için yalnızca kullanıldıklarında indirilip önbelleğe yazılır
// (bir kez açılan mod sonraki seferlerde internetsiz çalışır).
const CACHE = "nesne-tespit-v2";
const SHELL = [
  "./", "index.html", "style.css", "app.js", "labels.js", "manifest.json",
  "icons/icon-192.png", "icons/icon-512.png",
  "vendor/ort.webgpu.min.js", "vendor/ort-wasm-simd-threaded.jsep.mjs", "vendor/ort-wasm-simd-threaded.jsep.wasm",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  const url = new URL(e.request.url);
  if (url.origin !== location.origin) return;
  const isShellCode = /\.(html|js|css|json)$/.test(url.pathname) || url.pathname.endsWith("/");
  if (isShellCode && !url.pathname.includes("/vendor/")) {
    // Uygulama kodu: önce ağ (güncellemeler hemen gelsin), çevrimdışıysa önbellek
    e.respondWith(
      fetch(e.request).then((res) => {
        if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(e.request, copy)); }
        return res;
      }).catch(() => caches.match(e.request, { ignoreSearch: true }))
    );
    return;
  }
  // Modeller ve vendor dosyaları: önce önbellek, yoksa ağ + önbelleğe yaz
  e.respondWith(
    caches.match(e.request, { ignoreSearch: true }).then((hit) =>
      hit || fetch(e.request).then((res) => {
        if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(e.request, copy)); }
        return res;
      })
    )
  );
});
