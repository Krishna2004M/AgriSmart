# plot_final_accuracy.py
import json, numpy as np, matplotlib.pyplot as plt, tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore

MODEL_PATH = r"/mnt/e/capston P/Agri/Results/plantvillage_b4_best.keras"
DATA_DIR   = r"/mnt/e/capston P/Agri/PlantVillage_split/val"  # <-- use val (exists)
CLASSES_JSON = r"/mnt/e/capston P/Agri/Results/class_names.json"
IMG_SIZE = (380, 380); BATCH = 16

# Load model and class order from training
model = keras.models.load_model(MODEL_PATH)
with open(CLASSES_JSON, "r", encoding="utf-8") as f:
    class_names_saved = json.load(f)

# Build dataset with EXACT same class order/count
ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=False,
    class_names=class_names_saved,   # <- this prevents (None,16) vs (None,15)
)

def prep(x, y):
    return preprocess_input(tf.cast(x, tf.float32)), y

ds = ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

# Evaluate true accuracy on this split
loss, acc = model.evaluate(ds, verbose=1)
effnet_acc = acc * 100.0
print(f"\n✅ Accuracy (on {DATA_DIR}): {effnet_acc:.2f}%")

# ---- Bar chart (replace baselines if you have them) ----
models = ["Decision Tree","Naive Bayes","SVM","Logistic Regression",
          "Random Forest","XGBoost","KNN","EfficientNetB4"]
accuracies = [90.00, 99.09, 10.68, 95.23, 99.55, 99.09, 97.50, effnet_acc]

plt.figure(figsize=(10,6))
bars = plt.bar(models, np.array(accuracies)/100.0)
for b, a in zip(bars, accuracies):
    y = b.get_height()
    plt.text(b.get_x()+b.get_width()/2, y+0.01, f"{a:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.ylim(0,1.1); plt.ylabel("Accuracy"); plt.title("Model Accuracy Comparison")
plt.grid(axis="y", linestyle="--", alpha=0.4); plt.tight_layout()
plt.savefig(r"/mnt/e/capston P/Agri/Results/analysis/final_accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
