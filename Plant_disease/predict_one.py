"""
PlantVillage Disease Prediction — GPU / WSL (EfficientNetB4)
Author: Krishna M (AgriSens)
"""

import os, json, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input  # type: ignore # same as training
from PIL import Image

# ==============================
# CONFIGURATION
# ==============================
MODEL_PATH = r"/mnt/e/capston P/Agri/Results/plantvillage_b4_best.keras"
CLASSES_JSON = r"/mnt/e/capston P/Agri/Results/class_names.json"
IMG_PATH = r"/mnt/e/capston P/Agri/test_images/t1.JPG"
IMG_SIZE = (380, 380)

# ==============================
# ENVIRONMENT SETUP
# ==============================
print("\n[Env] TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("[Env] GPU(s) available:", gpus)
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        from tensorflow.keras import mixed_precision # type: ignore
        mixed_precision.set_global_policy("mixed_float16")
        print("[Env] Mixed precision:", mixed_precision.global_policy())
    except Exception as e:
        print("[Env] Could not enable mixed precision:", e)
else:
    print("[Env] ⚠️ No GPU detected — running on CPU")

# ==============================
# LOAD MODEL AND CLASS NAMES
# ==============================
print("\n🔄 Loading model and class labels...")
model = keras.models.load_model(MODEL_PATH)

with open(CLASSES_JSON, "r", encoding="utf-8") as f:
    class_names = json.load(f)

print(f"[Model] Loaded {len(class_names)} classes.")

# ==============================
# LOAD AND PREPROCESS IMAGE
# ==============================
print(f"\n🖼️ Loading image from: {IMG_PATH}")
if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(f"❌ Image not found: {IMG_PATH}")

img = Image.open(IMG_PATH).convert("RGB").resize(IMG_SIZE)
x = np.array(img, dtype=np.float32)
x = preprocess_input(x)[None, ...]  # Add batch dimension

# ==============================
# PREDICT
# ==============================
print("🤖 Running inference...")
probs = model.predict(x, verbose=0)[0]
pred_idx = int(np.argmax(probs))
pred_class = class_names[pred_idx]
confidence = probs[pred_idx] * 100

# ==============================
# DISPLAY RESULT
# ==============================
print(f"\n🧠 Prediction: {pred_class}")
print(f"📊 Confidence: {confidence:.2f}%\n")
