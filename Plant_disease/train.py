"""
PlantVillage Disease Classification — GPU / WSL (EfficientNetB4)
Author: Krishna M (AgriSens)
"""

import os, json, time, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers, models # type: ignore
from tensorflow.keras.applications import efficientnet             # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input  # type: ignore

# ----------------------- USER CONFIG -----------------------
DATA_DIR = r"/mnt/e/capston P/Agri/PlantVillage"
IMG_SIZE = (380, 380)
BATCH = 16
VAL_SPLIT = 0.20
SEED = 42
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 25
UNFREEZE_LAST_N_LAYERS = 100

OUT_DIR = "/mnt/e/capston P/Agri/Results"
MODEL_BEST = os.path.join(OUT_DIR, "plantvillage_b4_best.keras")
CLASSES_JSON = os.path.join(OUT_DIR, "class_names.json")
LOG_DIR = os.path.join(OUT_DIR, "tb_logs", time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(OUT_DIR, exist_ok=True)
# -----------------------------------------------------------

print("\n[Env] TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("[Env] GPU(s):", gpus)
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    try:
        from tensorflow.keras import mixed_precision    # type: ignore
        mixed_precision.set_global_policy("mixed_float16")
        print("[Env] Mixed precision:", mixed_precision.global_policy())
    except Exception as e:
        print("[Env] Could not enable mixed precision:", e)
else:
    print("[Env] ⚠️ No GPU detected — training on CPU (float32).")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ----------------------- DATA PIPELINE ---------------------
print("\n[Data] Loading datasets from:", DATA_DIR)

train_raw = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="categorical",
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="training",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=True,
    color_mode="rgb",
)

val_raw = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="categorical",
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="validation",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=False,
    color_mode="rgb",
)

class_names = train_raw.class_names
num_classes = len(class_names)
print(f"[Data] {num_classes} classes detected:", class_names)

with open(CLASSES_JSON, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

# ----- Augment + Preprocess -----
augment = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ],
    name="augment",
)

def prep_train(x, y):
    x = tf.cast(x, tf.float32)
    x = preprocess_input(x)
    x = augment(x, training=True)
    return x, y

def prep_val(x, y):
    x = tf.cast(x, tf.float32)
    x = preprocess_input(x)
    return x, y

AUTOTUNE = tf.data.AUTOTUNE
opt = tf.data.Options()
opt.experimental_optimization.apply_default_optimizations = True
opt.experimental_optimization.map_parallelization = True
opt.deterministic = False

train_ds = train_raw.map(prep_train, num_parallel_calls=AUTOTUNE).with_options(opt).prefetch(AUTOTUNE)
val_ds   = val_raw.map(prep_val,   num_parallel_calls=AUTOTUNE).with_options(opt).prefetch(AUTOTUNE)

# ----- Compute class weights -----
print("\n[Data] Computing class weights...")
counts = np.zeros(num_classes, dtype=np.int64)
for _, yb in train_raw.unbatch():
    counts[np.argmax(yb.numpy())] += 1
total = counts.sum()
class_weights = {i: float(total / (num_classes * max(1, counts[i]))) for i in range(num_classes)}
print("[Data] Counts:", counts.tolist())
print("[Data] Class weights:", class_weights)

# ----------------------- MODEL ------------------------------
def build_model(train_backbone: bool, lr: float):
    input_shape = IMG_SIZE + (3,)
    inputs = layers.Input(shape=input_shape, name="image")

    # Build EfficientNetB4 WITH imagenet weights but WITHOUT passing shapes/tensors;
    # then call it on our 3-channel 'inputs'. This avoids any accidental 1-channel graph.
    try:
        backbone = efficientnet.EfficientNetB4(include_top=False, weights="imagenet")
        x = backbone(inputs, training=False)
    except ValueError as e:
        # Safety net: if some Keras build still mismatches, fall back to random init.
        print("[Warn] Imagenet weight load failed, falling back to weights=None:", e)
        backbone = efficientnet.EfficientNetB4(include_top=False, weights=None)
        x = backbone(inputs, training=False)

    backbone.trainable = train_backbone

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)  # ensure fp32 head

    model = models.Model(inputs, outputs, name="EffNetB4_PlantVillage")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, backbone

# ---------------------- CALLBACKS ---------------------------
ckpt = callbacks.ModelCheckpoint(MODEL_BEST, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1)
reduce = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
early = callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1)
tb = callbacks.TensorBoard(LOG_DIR, write_graph=False, profile_batch=0)

TOTAL_EPOCHS = EPOCHS_STAGE1 + EPOCHS_STAGE2
def cosine_lr(epoch):
    base_lr = 3e-4
    scale = 0.5 * (1 + tf.math.cos(np.pi * epoch / TOTAL_EPOCHS))
    return float(base_lr * scale)
cosine = callbacks.LearningRateScheduler(cosine_lr, verbose=0)

# ---------------------- TRAINING ----------------------------
print("\n[Train] Stage 1 — Train classifier head (frozen backbone)")
model, base = build_model(train_backbone=False, lr=3e-4)
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weights,
    callbacks=[ckpt, reduce, cosine, tb],
    verbose=1,
)

print("\n[Train] Stage 2 — Fine-tune top layers")
for layer in base.layers[:-UNFREEZE_LAST_N_LAYERS]:
    layer.trainable = False
for layer in base.layers[-UNFREEZE_LAST_N_LAYERS:]:
    layer.trainable = True

model.compile(
    optimizer=optimizers.Adam(learning_rate=3e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weights,
    callbacks=[ckpt, reduce, early, tb, cosine],
    verbose=1,
)

# ---------------------- EVALUATION --------------------------
print("\n[Eval] Loading best checkpoint:", MODEL_BEST)
best = keras.models.load_model(MODEL_BEST)
val_loss, val_acc = best.evaluate(val_ds, verbose=0)
print(f"[Eval] Final Validation Accuracy: {val_acc*100:.2f}%")

# Collect predictions
y_true, y_pred = [], []
for xb, yb in val_ds:
    probs = best.predict(xb, verbose=0)
    y_pred.extend(np.argmax(probs, axis=1))
    y_true.extend(np.argmax(yb.numpy(), axis=1))

np.save(os.path.join(OUT_DIR, "y_true.npy"), np.array(y_true))
np.save(os.path.join(OUT_DIR, "y_pred.npy"), np.array(y_pred))
print("[Eval] Saved y_true/y_pred arrays in:", OUT_DIR)

print("\n✅ Training complete!")
print("[Model] Best:", MODEL_BEST)
print("[Classes] JSON:", CLASSES_JSON)
print("[Logs] TensorBoard:", LOG_DIR)
