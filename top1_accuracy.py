import os
import shutil
from pathlib import Path
from collections import Counter

import cv2
import yaml
from ultralytics import YOLO

# ----------------------------
# Ayarlar
# ----------------------------
MODEL_PATH = "best.pt"
DATA_YAML = r"Camera Occlusion Detection v1.v5i.yolov11/data.yaml"  # sende bu yol doğru
SPLIT = "val"  # "val" veya "test" (Roboflow genelde val)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

OUT_DIR = Path("mistakes_dump")
OUT_DIR.mkdir(exist_ok=True, parents=True)

COPY_IMAGES = True            # yanlış görselleri kopyala
SAVE_ANNOTATED = True         # üstüne yazı basıp kaydet
MAX_SAVE = 200                # maksimum kaç yanlış kaydedilsin

# ----------------------------
# data.yaml oku (path + class names)
# ----------------------------
with open(DATA_YAML, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

# Roboflow data.yaml bazen relative path içerir:
base_path = Path(data.get("path", Path(DATA_YAML).parent))
val_rel = data.get("val", "valid/images")  # bazıları "valid/images" döndürür
val_images_dir = Path("Camera Occlusion Detection v1.v5i.yolov11/valid/images")
labels_dir = Path("Camera Occlusion Detection v1.v5i.yolov11/valid/labels")


names = data.get("names")
if isinstance(names, dict):
    # {0:"a",1:"b"} gibi gelebilir
    names = [names[i] for i in sorted(names.keys())]
elif isinstance(names, list):
    pass
else:
    raise ValueError("data.yaml içinde 'names' bulunamadı veya formatı tanınmadı.")

# labels klasörünü bulalım: .../valid/labels veya .../val/labels
# val_images_dir = .../valid/images -> labels = .../valid/labels
labels_dir = val_images_dir.parent / "labels"

if not val_images_dir.exists():
    raise FileNotFoundError(f"Val images dir bulunamadı: {val_images_dir}")
if not labels_dir.exists():
    raise FileNotFoundError(f"Labels dir bulunamadı: {labels_dir}")

# ----------------------------
# Yardımcı: YOLO label oku -> class id
# (Bu projede her görsel tek sınıf gibi kullanılıyor: ilk satırın class'ı GT sayacağız)
# ----------------------------
def read_gt_class(label_path: Path):
    if not label_path.exists():
        return None
    txt = label_path.read_text(encoding="utf-8").strip().splitlines()
    if not txt:
        return None
    # YOLO format: class x y w h
    first = txt[0].split()
    if not first:
        return None
    return int(float(first[0]))

# ----------------------------
# Model
# ----------------------------
model = YOLO(MODEL_PATH)

# ----------------------------
# Döngü
# ----------------------------
total = 0
correct = 0
no_det = 0

confusions = Counter()
mistake_records = []

image_paths = sorted([p for p in val_images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])

for img_path in image_paths:
    total += 1
    label_path = labels_dir / (img_path.stem + ".txt")
    gt = read_gt_class(label_path)
    if gt is None:
        # GT yoksa pas geçelim
        continue

    # predict
    r = model.predict(source=str(img_path), conf=0.001, verbose=False)[0]

    # sınıflandırma mantığı (object detection çıktısından "en iyi class" seçiyoruz)
    if r.boxes is None or len(r.boxes) == 0:
        no_det += 1
        pred = None
    else:
        # en yüksek confidence'lı kutunun class'ı
        confs = r.boxes.conf.cpu().numpy()
        idx = int(confs.argmax())
        pred = int(r.boxes.cls[idx].cpu().item())

    if pred is not None and pred == gt:
        correct += 1
    else:
        # hata
        gt_name = names[gt] if 0 <= gt < len(names) else str(gt)
        pred_name = names[pred] if (pred is not None and 0 <= pred < len(names)) else ("NO_DET" if pred is None else str(pred))
        confusions[(gt_name, pred_name)] += 1
        mistake_records.append((img_path, gt_name, pred_name))

# ----------------------------
# Rapor
# ----------------------------
used = total  # bu örnekte total içinden GT olmayanları da saydık; istersen refine edebiliriz
acc = (correct / total) * 100 if total else 0.0
acc_used = (correct / (total - 0)) * 100 if total else 0.0  # placeholder

print("=" * 34)
print(f"Top-1 Accuracy (Validation): {acc:.2f}% ({correct}/{total})")
print(f"No-detection count: {no_det} ({(no_det/total*100 if total else 0):.2f}%)")
print("=" * 34)

print("\nTop confusions (gt -> pred):")
for (gt_name, pred_name), c in confusions.most_common(20):
    print(f"  {gt_name} -> {pred_name} : {c}")

# ----------------------------
# Yanlışları dışarı al (kopyala + annotate)
# ----------------------------
if mistake_records:
    dump_dir = OUT_DIR / "images"
    ann_dir = OUT_DIR / "annotated"
    dump_dir.mkdir(exist_ok=True, parents=True)
    ann_dir.mkdir(exist_ok=True, parents=True)

    saved = 0
    for img_path, gt_name, pred_name in mistake_records[:MAX_SAVE]:
        if COPY_IMAGES:
            shutil.copy2(img_path, dump_dir / img_path.name)

        if SAVE_ANNOTATED:
            img = cv2.imread(str(img_path))
            if img is not None:
                cv2.putText(img, f"GT: {gt_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(img, f"PRED: {pred_name}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imwrite(str(ann_dir / img_path.name), img)

        saved += 1

    # kayıt dosyası
    with open(OUT_DIR / "mistakes_list.txt", "w", encoding="utf-8") as f:
        for img_path, gt_name, pred_name in mistake_records:
            f.write(f"{img_path.name}\tGT={gt_name}\tPRED={pred_name}\n")

    print(f"\n✅ Mistakes saved: {saved} -> {OUT_DIR.resolve()}")
else:
    print("\n✅ Hiç yanlış yok (veya GT okunamadı).")
