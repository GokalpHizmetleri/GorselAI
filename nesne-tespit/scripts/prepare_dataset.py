"""COCO val2017 (80 sinif = 'her sey') icinden YOLO formatinda 2.500 gorsellik alt kume hazirlar.
Kaynak: images.cocodataset.org (val2017) + Ultralytics coco2017labels (YOLO etiketleri)."""
import random, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "datasets"
RAW = ROOT / "raw"
OUT = ROOT / "coco-mini"
N_TRAIN, N_VAL, SEED = 2000, 500, 42

labels = sorted(p for p in (RAW / "coco/labels/val2017").glob("*.txt") if p.stat().st_size > 0)
random.Random(SEED).shuffle(labels)
splits = {"train": labels[:N_TRAIN], "val": labels[N_TRAIN:N_TRAIN + N_VAL]}
for split, items in splits.items():
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    for lb in items:
        img = RAW / "val2017" / (lb.stem + ".jpg")
        shutil.copy(img, OUT / "images" / split / img.name)
        shutil.copy(lb, OUT / "labels" / split / lb.name)
    print(split, len(items))

names = [l.strip() for l in (Path(__file__).parent / "coco_names.txt").read_text().splitlines() if l.strip()]
yaml = f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
(OUT / "data.yaml").write_text(yaml)
print("data.yaml yazildi:", len(names), "sinif")
