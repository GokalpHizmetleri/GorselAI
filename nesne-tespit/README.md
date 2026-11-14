# Nesne Tespit — Mobil Canlı YOLO11n

Telefon kamerasıyla **tamamen cihaz üzerinde** çalışan, gerçek zamanlı nesne tespiti.
80 COCO sınıfı (insan, araba, kedi, telefon, şişe, sandalye…) Türkçe/İngilizce etiket ve güven skoruyla gösterilir.

**Canlı demo:** https://gokalphizmetleri.github.io/GorselAI/

## Modeller
Kendi 2.000 görsellik ince ayar modelimiz (mAP50 0.422) kaldırıldı. Yerine tam COCO veri setinde
(118 bin görsel) Ultralytics tarafından eğitilmiş resmi **YOLO26** modelleri kullanılıyor.
Arayüzdeki **mod butonu** ile seçilir:

| Mod | Model | Giriş | mAP50 (fp32 / int8) | Web dosyası |
|---|---|---|---|---|
| Hızlı | YOLO26n | 416 | 0.519 / 0.502 | 9.8 MB / 3.0 MB |
| **Dengeli** (WebGPU varsa varsayılan) | YOLO26s | 416 | 0.618 / 0.604 | 38 MB / 10 MB |
| Hassas | YOLO26m | 320 | 0.638 / 0.619 | 82 MB / 21 MB |

Ölçüm: COCO val2017'den 500 görsel. WebGPU olan telefonlarda fp32, olmayanlarda int8 (WASM) dosyası yüklenir.
WebGPU yoksa varsayılan mod Hızlı'dır. Modeller ilk kullanıldıklarında indirilir ve önbelleğe alınır.

### Mobil uygulamalar için (.tflite / LiteRT) — `models/`
| Dosya | mAP50 | Not |
|---|---|---|
| yolo26n_416_int8.tflite | 0.509 | Tam INT8 – NPU/DSP/CPU, en hızlı |
| yolo26n_416_w8a16.tflite | 0.516 | INT8 ağırlık + 16-bit aktivasyon – GPU delegate |
| yolo26s_416_int8.tflite | 0.612 | Tam INT8 |
| yolo26s_416_w8a16.tflite | 0.616 | GPU delegate |

Giriş `[1, 416, 416, 3]` float32 RGB 0–1 (letterbox, dolgu 114); çıkış `[1, 84, 3549]` → 4 kutu (cx, cy, w, h, piksel) + 80 sınıf skoru; NMS uygulamada yapılır.

## Klasör yapısı
```
nesne-tespit/
├── web/                 # GitHub Pages'te yayınlanan mobil PWA
│   ├── index.html  app.js  style.css  labels.js  manifest.json  service-worker.js  icons/
│   ├── model/           # hizli / dengeli / hassas (.onnx fp32 + _int8.onnx)
│   └── vendor/          # onnxruntime-web 1.22 (yerel kopya)
├── models/              # Tüm dışa aktarılmış modeller (.onnx, .tflite)
├── scripts/             # prepare_dataset.py (kalibrasyon/ölçüm verisi), export.py, evaluate.py
└── requirements.txt
```

## Yeniden üretme
```bash
pip install -r requirements.txt
# Kalibrasyon ve ölçüm için COCO alt kümesi (datasets/ .gitignore'da):
curl -LO https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip
curl -LO http://images.cocodataset.org/zips/val2017.zip   # ikisini datasets/raw/ içine açın
python scripts/prepare_dataset.py
python scripts/export.py     # ONNX + TFLite üretir, web/model'e kopyalar
python scripts/evaluate.py   # doğruluk tablosu
```

## Web uygulaması
- Arka kamera (`facingMode: "environment"`), ön/arka değiştirme, tam ekran, ekran açık kalır (Wake Lock).
- Mod butonu: Hızlı / Dengeli / Hassas. Yeni model yüklenirken eskisi çalışmaya devam eder.
- Çıkarım video karesine kilitli (`requestVideoFrameCallback`), çizim ayrı `requestAnimationFrame` döngüsünde;
  IoU takipçi + yumuşatma + hız tahmini ile kamera hareket ederken kutular akıcı ilerler.
- Dil: TR → EN → TR/EN. Güven eşiği: 25 / 35 / 50 / 65 %.
- PWA: uygulama ve kullanılan modeller önbelleğe alınır, sonra internetsiz çalışır.
