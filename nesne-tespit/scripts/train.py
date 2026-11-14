"""YOLO11n (Nano) mobil-optimizasyonlu ince ayar egitimi - CPU, imgsz=320.

COCO on-egitimli agirliklar 80 sinifi ('her sey') zaten biliyor. 2.000 gorselle
agresif egitim bu bilgiyi bozar; bu yuzden omurga dondurulur, dusuk LR ve
warmup'siz SGD ile koruyucu bir ince ayar (fine-tune) yapilir.
"""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
model = YOLO("yolo11n.pt")
model.train(
    data=str(ROOT / "datasets/coco-mini/data.yaml"),
    imgsz=320, epochs=15, batch=32, device="cpu", workers=4,
    optimizer="SGD", lr0=0.001, lrf=0.2, momentum=0.9, weight_decay=0.0005,
    warmup_epochs=0, warmup_bias_lr=0.0, cos_lr=True,
    freeze=10,                     # omurgayi (backbone) dondur
    mosaic=0.5, mixup=0.0, close_mosaic=5, patience=6, cache="ram",
    project=str(ROOT / "runs"), name="yolo11n_320", exist_ok=True, plots=True,
)
