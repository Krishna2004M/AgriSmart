# analyze_predictions.py
import csv, sys, json, shutil
from pathlib import Path
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Inputs
VAL_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/e/capston P/Agri/PlantVillage_split/val")
CSV_PATH = VAL_DIR / "predictions.csv"
CLASSES_JSON = Path("/mnt/e/capston P/Agri/Results/class_names.json")
OUT_DIR = VAL_DIR / "analysis_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load class list (order matters)
with open(CLASSES_JSON, "r", encoding="utf-8") as f:
    class_names = json.load(f)
name_to_idx = {n:i for i,n in enumerate(class_names)}

files, y_true, y_pred, confs = [], [], [], []
mis = []  # misclassifications

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        fn = r["file"]
        pred = r["prediction"]
        conf = float(r["confidence"])
        true = r["label_from_name"]
        correct = (r["correct"].strip().lower() == "true")
        if true not in name_to_idx or pred not in name_to_idx:
            continue
        files.append(fn)
        y_true.append(name_to_idx[true])
        y_pred.append(name_to_idx[pred])
        confs.append(conf)
        if not correct:
            mis.append((fn, true, pred, conf))

y_true = np.array(y_true, dtype=int)
y_pred = np.array(y_pred, dtype=int)

acc = (y_true == y_pred).mean() * 100
print(f"Accuracy from CSV: {acc:.2f}% on {len(y_true)} images")

# Per-class accuracy
per_class_acc = {}
for i, cname in enumerate(class_names):
    idx = (y_true == i)
    if idx.sum() > 0:
        per_class_acc[cname] = float((y_pred[idx] == i).mean() * 100)

# Confusion matrix
cm = np.zeros((len(class_names), len(class_names)), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1

# Save confusion matrix plot
plt.figure(figsize=(10, 9))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=7)
plt.yticks(range(len(class_names)), class_names, fontsize=7)
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=200)
plt.close()

# Save per-class accuracy + summary
with open(OUT_DIR / "report.txt", "w", encoding="utf-8") as f:
    f.write(f"Overall accuracy: {acc:.2f}% on {len(y_true)} images\n\n")
    f.write("Per-class accuracy:\n")
    for cname, a in sorted(per_class_acc.items(), key=lambda x: x[0]):
        f.write(f"{cname:40s} {a:6.2f}%\n")

# Top-N highest-confidence mistakes
mis.sort(key=lambda x: x[3], reverse=True)
topN = 30
with open(OUT_DIR / "top_mistakes.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "true", "pred", "confidence"])
    w.writerows(mis[:topN])

# Optionally copy those mistakes for quick visual inspection
copy_dir = OUT_DIR / "top_mistakes_imgs"
copy_dir.mkdir(exist_ok=True)
for fn, true, pred, conf in mis[:topN]:
    src = Path(fn)
    if src.exists():
        dst = copy_dir / f"{Path(fn).stem}__TRUE={true}__PRED={pred}__CONF={conf:.3f}{src.suffix}"
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass

print(f"Saved: {OUT_DIR/'confusion_matrix.png'}")
print(f"Saved: {OUT_DIR/'report.txt'}")
print(f"Saved: {OUT_DIR/'top_mistakes.csv'} and images in {copy_dir}")
