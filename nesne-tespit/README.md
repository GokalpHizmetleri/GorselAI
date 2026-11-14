# Nesne Tespit — Mobil Canlı YOLO11n

Telefon kamerasıyla **tamamen cihaz üzerinde** çalışan, gerçek zamanlı nesne tespiti.
80 COCO sınıfı (insan, araba, kedi, telefon, şişe, sandalye…) Türkçe/İngilizce etiket ve güven skoruyla gösterilir.

**Canlı demo:** https://gokalphizmetleri.github.io/GorselAI/

## Klasör yapısı
```
nesne-tespit/
├── web/                 # GitHub Pages'te yayınlanan mobil PWA
│   ├── index.html  app.js  style.css  labels.js
│   ├── manifest.json  service-worker.js  icons/
│   ├── model/           # yolo11n_320.onnx (WebGPU), yolo11n_320_int8.onnx (WASM)
│   └── vendor/          # onnxruntime-web 1.22 (çevrimdışı çalışma için yerel kopya)
├── models/              # Mobil uygulamalara gömülecek modeller (.tflite, .onnx, .pt)
├── scripts/             # prepare_dataset.py, train.py, export.py, evaluate.py
├── results/             # Eğitim grafikleri ve metrikleri
└── requirements.txt
```

## Veri seti
COCO 2017 val (Ultralytics YOLO etiketleri) içinden rastgele **2.500 görsel** alındı: 2.000 eğitim / 500 doğrulama, 80 sınıf.
Veri seti `.gitignore`'da; yeniden üretmek için:
```bash
curl -LO https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip
curl -LO http://images.cocodataset.org/zips/val2017.zip   # datasets/raw/ içine açın
python scripts/prepare_dataset.py
```

## Eğitim (CPU, 4 vCPU)
`yolo11n.pt` (COCO ön-eğitimli) → `imgsz=320`, SGD lr0=0.001, warmup yok, omurga donuk (`freeze=10`), en fazla 15 epoch
(erken durdurma 8. epoch sonunda devreye girdi, en iyi sonuç 2. epoch). 2.000 görselle agresif eğitim 118 bin görsellik ön-eğitimi bozduğundan
koruyucu bir ince ayar yapıldı.

## Sonuçlar (500 doğrulama görseli, 320 px)
| Model | Boyut | mAP50 | mAP50-95 | Kullanım |
|---|---|---|---|---|
| yolo11n_320_best.pt | 5.5 MB | 0.419 | 0.285 | PyTorch |
| yolo11n_320.onnx (fp32) | 10.6 MB | 0.427 | 0.288 | Tarayıcı – WebGPU |
| yolo11n_320_int8.onnx (QDQ statik int8) | 3.1 MB | 0.418 | 0.276 | Tarayıcı – WASM/CPU |
| yolo11n_320_int8.tflite (tam int8) | 3.2 MB | 0.419 | 0.282 | Android/iOS – NPU/DSP/CPU |
| yolo11n_320_w8a16.tflite (int8 ağırlık + 16-bit aktivasyon) | 3.1 MB | 0.421 | 0.280 | Android/iOS – GPU delegate |

Karşılaştırma: ön-eğitimli `yolo11n.pt` aynı sette 0.434 / 0.300 veriyor.

## Web uygulaması
- Arka kamera (`facingMode: "environment"`), ön/arka değiştirme butonu, tam ekran, ekranın açık kalması (Wake Lock).
- ONNX Runtime Web: önce **WebGPU** (fp32), desteklenmezse **WASM** (int8) kullanılır.
- Çıkarım döngüsü video karesine kilitli (`requestVideoFrameCallback`), çizim ayrı bir `requestAnimationFrame` döngüsünde.
  IoU tabanlı takipçi + yumuşatma + hız tahmini sayesinde kamera hareket ederken kutular iki çıkarım arasında da ilerletilir.
- Dil: TR → EN → TR/EN. Güven eşiği: 25 / 35 / 50 / 65 %.
- PWA: `manifest.json` + `service-worker.js`; ilk açılıştan sonra internet olmadan çalışır, “Ana ekrana ekle” ile uygulama gibi açılır.

## Mobil uygulamaya gömme (.tflite)
Giriş: `[1, 320, 320, 3]` float32 RGB, 0–1 aralığında (letterbox, dolgu 114). Çıkış: `[1, 84, 2100]` → 4 kutu (cx, cy, w, h; piksel) + 80 sınıf skoru; NMS uygulamada yapılır.
