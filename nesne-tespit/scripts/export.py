"""Tam COCO (118k gorsel) uzerinde egitilmis resmi YOLO26 modellerini mobil formatlara aktarir.
  - ONNX fp32           -> tarayici, WebGPU
  - ONNX int8 (QDQ)     -> tarayici, WASM/CPU yedek yolu
  - TFLite/LiteRT int8 + w8a16 -> Android / iOS uygulamalari
Kullanim: python scripts/export.py
"""
import glob, shutil
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quant_pre_process, quantize_static)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets/coco-mini/data.yaml"
OUT = ROOT / "models"; OUT.mkdir(exist_ok=True)
WEB = ROOT / "web/model"; WEB.mkdir(parents=True, exist_ok=True)

# (agirlik, giris boyutu, web modu)
MODELS = [("yolo26n.pt", 416, "hizli"), ("yolo26s.pt", 416, "dengeli"), ("yolo26m.pt", 320, "hassas")]
TFLITE = {"yolo26n.pt", "yolo26s.pt"}
CALIB = sorted(glob.glob(str(ROOT / "datasets/coco-mini/images/train/*.jpg")))[:200]


def letterbox(path, size):
    im = Image.open(path).convert("RGB"); w, h = im.size; s = min(size / w, size / h)
    nw, nh = round(w * s), round(h * s)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(im.resize((nw, nh), Image.BILINEAR), ((size - nw) // 2, (size - nh) // 2))
    return (np.asarray(canvas, np.float32) / 255.0).transpose(2, 0, 1)[None]


class Reader(CalibrationDataReader):
    def __init__(self, files, size): self.it, self.size = iter(files), size
    def get_next(self):
        f = next(self.it, None)
        return None if f is None else {"images": letterbox(f, self.size)}


for weights, size, mode in MODELS:
    stem = f"{Path(weights).stem}_{size}"
    fp32 = OUT / f"{stem}.onnx"
    shutil.copy(YOLO(weights).export(format="onnx", imgsz=size, opset=17, simplify=True), fp32)

    # Sadece Conv katmanlari kuantize edilir; cikis basligi float kalir (kutu 0-416 ve
    # skor 0-1 ayni uint8 olcegine sikisirsa skorlar sifirlanir).
    pre = OUT / "_pre.onnx"
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
    quantize_static(str(pre), str(OUT / f"{stem}_int8.onnx"), Reader(CALIB, size),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["Conv"])
    pre.unlink()
    for f in (fp32, OUT / f"{stem}_int8.onnx"):
        shutil.copy(f, WEB / f"{mode}{'_int8' if 'int8' in f.name else ''}.onnx")

    if weights in TFLITE:
        for q, tag in ((8, "int8"), ("w8a16", "w8a16")):
            p = YOLO(weights).export(format="litert", imgsz=size, quantize=q, data=str(DATA), fraction=0.1)
            shutil.copy(p, OUT / f"{stem}_{tag}.tflite")

for p in sorted(OUT.iterdir()):
    print(f"{p.name:30s} {p.stat().st_size / 1e6:7.2f} MB")
