"""Disa aktarilan modelleri dogrulama setinde (COCO val2017'den 500 gorsel) karsilastirir."""
import re
from pathlib import Path
from ultralytics import YOLO
ROOT = Path(__file__).resolve().parent.parent
DATA = str(ROOT / "datasets/coco-mini/data.yaml")
print("| Model | mAP50 | mAP50-95 | CPU çıkarım (ms) |\n|---|---|---|---|", flush=True)
for f in sorted((ROOT / "models").glob("*")):
    size = int(re.search(r"_(\d{3})", f.name).group(1))
    r = YOLO(str(f), task="detect").val(data=DATA, imgsz=size, batch=1, device="cpu", plots=False,
                                         verbose=False, project="/tmp/claude-0/val", name=f.name)
    print(f"| {f.name} | {r.box.map50:.3f} | {r.box.map:.3f} | {r.speed['inference']:.1f} |", flush=True)
