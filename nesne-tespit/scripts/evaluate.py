"""Disa aktarilan tum modelleri dogrulama setinde (500 gorsel) karsilastirir."""
from pathlib import Path
from ultralytics import YOLO
ROOT = Path(__file__).resolve().parent.parent
DATA = str(ROOT / "datasets/coco-mini/data.yaml")
rows = []
for f in ["yolo11n_320_best.pt", "yolo11n_320.onnx", "yolo11n_320_int8.onnx", "yolo11n_320_int8.tflite", "yolo11n_320_w8a16.tflite"]:
    r = YOLO(str(ROOT / "models" / f), task="detect").val(data=DATA, imgsz=320, batch=1, device="cpu", plots=False, verbose=False, project="/tmp/claude-0/val", name=f)
    rows.append((f, r.box.map50, r.box.map, r.speed["inference"]))
print("\n| Model | mAP50 | mAP50-95 | CPU çıkarım (ms) |\n|---|---|---|---|")
for f, a, b, s in rows: print(f"| {f} | {a:.3f} | {b:.3f} | {s:.1f} |")
