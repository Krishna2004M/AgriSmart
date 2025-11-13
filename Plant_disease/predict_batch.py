# predict_batch.py
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore

MODEL_PATH   = r"/mnt/e/capston P/Agri/Results/plantvillage_b4_best.keras"
CLASSES_JSON = r"/mnt/e/capston P/Agri/Results/class_names.json"
IMG_SIZE     = (380, 380)

def imgs(folder: Path):
    exts = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts: yield p

def load(path: Path):
    x = np.array(Image.open(path).convert("RGB").resize(IMG_SIZE), dtype=np.float32)
    return preprocess_input(x)[None, ...]  # (1,H,W,3)

def main():
    if len(sys.argv) < 2: 
        print("Usage: python predict_batch.py <folder_path>"); return
    folder = Path(sys.argv[1]); assert folder.exists(), f"Folder not found: {folder}"

    model = keras.models.load_model(MODEL_PATH)
    class_names = json.loads(Path(CLASSES_JSON).read_text(encoding="utf-8"))

    files = list(imgs(folder))
    out_csv = folder / "predictions.csv"
    if not files:
        out_csv.write_text("file,prediction,confidence,label_from_name,correct\n"); print(f"Saved: {out_csv}"); return

    rows, ok, fail = [], 0, 0
    for p in files:
        try:
            probs = model.predict(load(p), verbose=0)[0]
            i = int(np.argmax(probs)); pred, conf = class_names[i], float(probs[i])
            true = p.parent.name; rows.append([str(p), pred, f"{conf:.6f}", true, str(pred==true)]); ok += 1
        except Exception: fail += 1

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("file,prediction,confidence,label_from_name,correct\n")
        f.writelines(",".join(r) + "\n" for r in rows)

    acc = sum(r[4]=="True" for r in rows)/len(rows)
    print(f"Saved: {out_csv}\nSummary: ok={ok}, failed={fail}, accuracy={acc*100:.2f}% ({sum(r[4]=='True' for r in rows)}/{len(rows)})")

if __name__ == "__main__":
    main()
