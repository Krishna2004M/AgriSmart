# plot_test_metrics.py
# Usage:
#   python plot_test_metrics.py "/mnt/e/capston P/Agri/PlantVillage_split/test"
#   # or use your val split if you don't have test

import sys, json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

# ---- paths from your training run ----
MODEL_PATH   = r"/mnt/e/capston P/Agri/Results/plantvillage_b4_best.keras"
CLASSES_JSON = r"/mnt/e/capston P/Agri/Results/class_names.json"
IMG_SIZE = (380, 380)
BATCH = 16
OUT_DIR = r"/mnt/e/capston P/Agri/Results/analysis"

def load_dataset(data_dir: str, class_names):
    ds = keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=False,
        color_mode="rgb",
        class_names=class_names,   # force same order as training
    )
    def prep(x,y):
        return preprocess_input(tf.cast(x, tf.float32)), y
    return ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE), ds.file_paths

def plot_confusion(cm, classes, out_path, normalize=True):
    if normalize:
        cm = cm.astype(np.float32) / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix (normalized)" if normalize else "Confusion Matrix")
    thresh = cm.max()*0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:.2f}" if normalize else int(cm[i,j]),
                    ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black", fontsize=7)
    fig.tight_layout(); fig.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close(fig)

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_test_metrics.py <test_or_val_folder>")
        sys.exit(1)
    data_dir = sys.argv[1]
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # load model + class order used at train time
    model = keras.models.load_model(MODEL_PATH)
    class_names = json.loads(Path(CLASSES_JSON).read_text(encoding="utf-8"))

    # dataset
    ds, file_paths = load_dataset(data_dir, class_names)

    # predict
    y_true, probs = [], []
    for xb, yb in ds:
        p = model.predict(xb, verbose=0)
        probs.append(p)
        y_true.append(np.argmax(yb.numpy(), axis=1))
    probs = np.concatenate(probs); y_true = np.concatenate(y_true); y_pred = probs.argmax(1)

    # metrics
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    df.to_csv(Path(OUT_DIR, "per_class_metrics.csv"), index=True)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plot_confusion(cm, class_names, Path(OUT_DIR, "confusion_matrix.png"), normalize=True)

    # brief txt summary
    with open(Path(OUT_DIR, "summary_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Dataset: {data_dir}\n")
        f.write(f"Accuracy: {acc*100:.2f}%\n")
        f.write(f"Macro Precision: {report['macro avg']['precision']:.4f}\n")
        f.write(f"Macro Recall:    {report['macro avg']['recall']:.4f}\n")
        f.write(f"Macro F1-score:  {report['macro avg']['f1-score']:.4f}\n")
        f.write(f"Weighted Precision: {report['weighted avg']['precision']:.4f}\n")
        f.write(f"Weighted Recall:    {report['weighted avg']['recall']:.4f}\n")
        f.write(f"Weighted F1-score:  {report['weighted avg']['f1-score']:.4f}\n")

    print("\n✅ Saved to:", OUT_DIR)
    print(" - per_class_metrics.csv (precision/recall/F1 per class)")
    print(" - confusion_matrix.png (normalized)")
    print(" - summary_metrics.txt (accuracy + macro/weighted P/R/F1)")

if __name__ == "__main__":
    main()
