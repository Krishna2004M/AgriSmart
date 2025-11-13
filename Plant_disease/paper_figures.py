# paper_figures.py — final, with named-input fix for Grad-CAM
import os, json, math, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.manifold import TSNE
import pandas as pd
from PIL import Image
from tensorflow.keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D # type: ignore

# -------------------- CONFIG --------------------
MODEL_PATH   = r"/mnt/e/capston P/Agri/Results/plantvillage_b4_best.keras"
CLASSES_JSON = r"/mnt/e/capston P/Agri/Results/class_names.json"
DATA_DIR     = r"/mnt/e/capston P/Agri/PlantVillage_split/val"   # use test/ if available
OUT_DIR      = r"/mnt/e/capston P/Agri/Results/analysis"
IMG_SIZE     = (380, 380)
BATCH        = 16
SEED         = 42
# ------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)
random.seed(SEED)

# -------------------- DATA ----------------------
with open(CLASSES_JSON, "r", encoding="utf-8") as f:
    class_names = json.load(f)

ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=False,
    color_mode="rgb",
    class_names=class_names,  # force same order/count as training
)
file_paths = ds.file_paths

def _prep(x, y):
    return preprocess_input(tf.cast(x, tf.float32)), y

ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

# -------------------- MODEL + PREDICTIONS ----------------------
model = keras.models.load_model(MODEL_PATH)

probs_list, ytrue_list = [], []
for xb, yb in ds:
    p = model.predict(xb, verbose=0)
    probs_list.append(p)
    ytrue_list.append(np.argmax(yb.numpy(), axis=1))

probs = np.concatenate(probs_list, axis=0)
y_true = np.concatenate(ytrue_list, axis=0)
y_pred = probs.argmax(axis=1)
assert probs.size > 0 and y_true.size > 0, "Dataset produced no samples—check DATA_DIR and class_names."

# -------------------- PER-CLASS METRICS + F1 BAR ----------------
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
pd.DataFrame(report).transpose().to_csv(Path(OUT_DIR, "per_class_metrics.csv"))

f1s = [report[c]["f1-score"] for c in class_names]
plt.figure(figsize=(11, 4))
plt.bar(range(len(class_names)), f1s)
plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
plt.ylabel("F1-score")
plt.title("Per-class F1-score")
plt.tight_layout()
plt.savefig(Path(OUT_DIR, "per_class_f1_bar.png"), dpi=300, bbox_inches="tight")
plt.close()

# -------------------- PR / ROC ----------------
yb = keras.utils.to_categorical(y_true, num_classes=len(class_names))

plt.figure(figsize=(8, 6))
ap_vals = []
for c in range(len(class_names)):
    precision, recall, _ = precision_recall_curve(yb[:, c], probs[:, c])
    ap = average_precision_score(yb[:, c], probs[:, c])
    ap_vals.append(ap)
    plt.plot(recall, precision, label=f"{class_names[c]} (AP={ap:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Per-class Precision–Recall Curves")
plt.legend(fontsize=7, ncol=2); plt.tight_layout()
plt.savefig(Path(OUT_DIR, "pr_curves.png"), dpi=300, bbox_inches="tight"); plt.close()

plt.figure(figsize=(8, 6))
auc_vals = []
for c in range(len(class_names)):
    fpr, tpr, _ = roc_curve(yb[:, c], probs[:, c])
    roc_auc = auc(fpr, tpr); auc_vals.append(roc_auc)
    plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={roc_auc:.3f})")
plt.plot([0,1],[0,1],'--', lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("Per-class ROC Curves")
plt.legend(fontsize=7, ncol=2); plt.tight_layout()
plt.savefig(Path(OUT_DIR, "roc_curves.png"), dpi=300, bbox_inches="tight"); plt.close()

# -------------------- t-SNE (Penultimate Features) -------------
penultimate = model.layers[-1].input  # input to final Dense
feat_model = keras.Model(model.input, penultimate)
feats = []
for xb, _ in ds:
    feats.append(feat_model.predict(xb, verbose=0))
feats = np.concatenate(feats, axis=0)
n = min(2000, feats.shape[0])
idx = rng.choice(feats.shape[0], size=n, replace=False)
emb = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, random_state=SEED).fit_transform(feats[idx])
plt.figure(figsize=(7, 6))
plt.scatter(emb[:,0], emb[:,1], c=y_true[idx], s=8, alpha=0.85, cmap="tab20")
plt.title("t-SNE of Penultimate Features")
plt.tight_layout()
plt.savefig(Path(OUT_DIR, "tsne_features.png"), dpi=300, bbox_inches="tight"); plt.close()

# -------------------- Grad-CAM (handles named input & submodels) ----------------
def find_last_conv_recursive(m):
    """Return the last convolutional layer object, searching recursively."""
    for layer in reversed(m.layers):
        if hasattr(layer, "layers") and isinstance(layer, (keras.Model, keras.Sequential)):
            found = find_last_conv_recursive(layer)
            if found is not None:
                return found
        if isinstance(layer, (Conv2D, SeparableConv2D, DepthwiseConv2D)):
            return layer
    return None

target_layer = find_last_conv_recursive(model)
assert target_layer is not None, "No conv layer found (cannot compute Grad-CAM)."

# Build grad model (same input as original)
grad_model = keras.Model(inputs=model.inputs, outputs=[target_layer.output, model.output])
input_key = model.input_names[0] if hasattr(model, "input_names") else None  # typically 'image'

def grad_cam_from_array(img_array, class_index):
    # Ensure float32 tensor
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
    # Feed as dict if model expects named input
    feed = {input_key: img_array} if input_key else img_array
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(feed)
        class_channel = preds[:, class_index]
    grads = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(1, 2))     # (B, C)
    conv_out = conv_out[0]                          # (H, W, C)
    weights = pooled[0]                             # (C,)
    heatmap = tf.reduce_sum(conv_out * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def load_rgb(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    return np.array(img)

def to_model_input(rgb_uint8):
    return preprocess_input(rgb_uint8.astype(np.float32))[None, ...]

correct_idx = np.where(y_true == y_pred)[0]
wrong_idx   = np.where(y_true != y_pred)[0]
pick_c = correct_idx[:4] if len(correct_idx) >= 4 else correct_idx
pick_w = wrong_idx[:4]   if len(wrong_idx)   >= 4 else wrong_idx
picks = list(pick_c) + list(pick_w)

if len(picks) > 0:
    cols = 4
    rows = math.ceil(len(picks)/cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols*3.0, rows*3.0))
    axs = np.array(axs).reshape(rows, cols)
    for ax in axs.ravel(): ax.axis("off")

    for ax, i in zip(axs.ravel(), picks):
        rgb = load_rgb(file_paths[i])
        x = to_model_input(rgb)
        heat = grad_cam_from_array(x, class_index=int(y_pred[i]))
        heat = np.array(Image.fromarray((heat*255).astype(np.uint8)).resize(IMG_SIZE[::-1]))/255.0
        overlay = (0.6*rgb + 0.4*(plt.cm.jet(heat)[..., :3]*255)).astype(np.uint8)
        ax.imshow(overlay)
        ax.set_title(f"T:{class_names[y_true[i]]}\nP:{class_names[y_pred[i]]}", fontsize=8)

    fig.suptitle("Grad-CAM (top: correct, bottom: incorrect if available)", y=0.98, fontsize=10)
    fig.tight_layout()
    fig.savefig(Path(OUT_DIR, "gradcam_gallery.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

# -------------------- SUMMARY -------------------
acc = (y_true == y_pred).mean()
with open(Path(OUT_DIR, "figures_summary.txt"), "w", encoding="utf-8") as f:
    f.write(f"Data: {DATA_DIR}\n")
    f.write(f"Images: {len(y_true)} | Classes: {len(class_names)}\n")
    f.write(f"Accuracy: {acc*100:.2f}%\n")
    f.write(f"Mean AP (macro): {np.mean(ap_vals):.4f}\n")
    f.write(f"Mean ROC-AUC (macro): {np.mean(auc_vals):.4f}\n")

print("\n✅ Saved figures to:", OUT_DIR)
print(" - per_class_metrics.csv")
print(" - per_class_f1_bar.png")
print(" - pr_curves.png")
print(" - roc_curves.png")
print(" - tsne_features.png")
print(" - gradcam_gallery.png")
print(" - figures_summary.txt")
