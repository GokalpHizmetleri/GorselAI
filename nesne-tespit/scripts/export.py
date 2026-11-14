"""En iyi modeli mobil formatlara aktarir:
  - ONNX fp32 (WebGPU)  + ONNX int8 statik kuantize (QDQ, tarayici WASM/CPU)
  - TFLite float16 ve int8 (Android / iOS uygulamalarina gomulebilir)
Kullanim: python scripts/export.py [weights.pt]
"""
import shutil, sys, glob
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
W = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/yolo11n_320/weights/best.pt"
DATA = ROOT / "datasets/coco-mini/data.yaml"
OUT = ROOT / "models"; OUT.mkdir(exist_ok=True)
WEB = ROOT / "web/model"; WEB.mkdir(parents=True, exist_ok=True)
IMG = 320

# 1) ONNX fp32 (opset 17, sadelestirilmis)
onnx_path = Path(YOLO(W).export(format="onnx", imgsz=IMG, opset=17, simplify=True, dynamic=False))
fp32 = OUT / "yolo11n_320.onnx"; shutil.copy(onnx_path, fp32)

# 2) ONNX int8 statik kuantizasyon (QDQ) - kalibrasyon: dogrulama gorselleri
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType, quant_pre_process

def letterbox(p):
    im = Image.open(p).convert("RGB"); w, h = im.size; s = min(IMG / w, IMG / h)
    nw, nh = round(w * s), round(h * s)
    canvas = Image.new("RGB", (IMG, IMG), (114, 114, 114))
    canvas.paste(im.resize((nw, nh), Image.BILINEAR), ((IMG - nw) // 2, (IMG - nh) // 2))
    return (np.asarray(canvas, np.float32) / 255.0).transpose(2, 0, 1)[None]

class Reader(CalibrationDataReader):
    def __init__(self, files): self.it = iter(files)
    def get_next(self):
        f = next(self.it, None)
        return None if f is None else {"images": letterbox(f)}

calib = sorted(glob.glob(str(ROOT / "datasets/coco-mini/images/train/*.jpg")))[:200]
pre = OUT / "_pre.onnx"
quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
int8 = OUT / "yolo11n_320_int8.onnx"
quantize_static(str(pre), str(int8), Reader(calib), quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                # Sadece Conv katmanlari kuantize edilir; cikis basligi (DFL/sigmoid/concat) float kalir.
                # Aksi halde kutu (0-320) ve skor (0-1) ayni uint8 olcegine sikisir ve skorlar sifirlanir.
                op_types_to_quantize=["Conv"], extra_options={"ActivationSymmetric": False})
pre.unlink()

# 3) TFLite / LiteRT (Ultralytics 8.4+: format="litert")
#    - int8   : tam INT8 (NPU/DSP icin en hizli, kalibrasyonlu)
#    - w8a16  : INT8 agirlik + 16-bit aktivasyon (GPU delegate icin float16 muadili)
tfl8 = Path(YOLO(W).export(format="litert", imgsz=IMG, quantize=8, data=str(DATA), fraction=0.1))
shutil.copy(tfl8, OUT / "yolo11n_320_int8.tflite")
tfl16 = Path(YOLO(W).export(format="litert", imgsz=IMG, quantize="w8a16", data=str(DATA), fraction=0.1))
shutil.copy(tfl16, OUT / "yolo11n_320_w8a16.tflite")
shutil.copy(W, OUT / "yolo11n_320_best.pt")

# Web uygulamasina kopyala
for f in ("yolo11n_320.onnx", "yolo11n_320_int8.onnx"):
    shutil.copy(OUT / f, WEB / f)
for p in sorted(OUT.iterdir()):
    print(f"{p.name:32s} {p.stat().st_size/1e6:6.2f} MB")
