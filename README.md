🌾 AgriSmart – AI-Powered Crop & Plant Disease Identification
Built by Krishna M (AgriSens)

AgriSmart is an AI-driven agricultural support system designed to help farmers with:

🍃 Plant Disease Detection using a deep learning EfficientNetB4 model

🌱 Crop Recommendation using a Random Forest classifier

📊 High-accuracy, field-ready predictions

⚙️ Compatible with CPU, GPU, and WSL environments

Both ML & DL models in this project were trained using high-quality Kaggle datasets and optimized for real-world usage.

📁 Project Structure
AGRI/
│
├── crop_recommadtion/
│   ├── Crop_recommendation.csv
│   ├── crop_recommender_rf.joblib
│   ├── train_crop_model.py
│   ├── use_model.py
│
├── Plant_disease/
│   ├── analyze_predictions.py
│   ├── Make_Split.py
│   ├── paper_figures.py
│   ├── plot_final_accuracy.py
│   ├── plot_test_metrics.py
│   ├── predict_batch.py
│   ├── predict_one.py
│   ├── train.py
│
├── PlantVillage/                  # Original Kaggle dataset
├── PlantVillage_split/
│   ├── train/
│   ├── val/
│
├── Results/
│   ├── analysis/
│   │   └── report.txt
│   ├── tb_logs/
│   ├── class_names.json
│   ├── y_pred.npy
│   ├── y_true.npy
│
├── test_images/
│   └── t1.JPG
│
└── README.md

🚨 Important Note — Model File Not Included

GitHub does NOT allow large files in the repository.
Therefore, the trained plant disease model:

plantvillage_b4_best.keras


is not uploaded here.

👉 A cloud storage link will be added to download the .keras model.

After downloading, place it here:

AGRI/Results/plantvillage_b4_best.keras

🍃 1. Plant Disease Classification (EfficientNetB4)
📌 Dataset

Used: PlantVillage Dataset (Kaggle)
https://www.kaggle.com/datasets/emmarex/plantdisease

📌 Features Of the Model

EfficientNetB4 backbone (ImageNet pretrained)

Optimized for GPU / WSL

RandomFlip, Rotation, Zoom, Contrast augmentations

Two-stage training:

Stage 1: Train classification head

Stage 2: Fine-tune last 100 layers

Mixed precision enabled

Computes class weights

Saves predictions + labels for metric analysis

🚀 Training
python Plant_disease/train.py

📈 Plant Disease Model Performance
🔥 Overall Accuracy: 99.66%

(4127 validation images)

📊 Per-Class Accuracy
Class	Accuracy
Pepper_bell___Bacterial_spot	99.50%
Pepper_bell___healthy	99.66%
Potato___Early_blight	100.00%
Potato___Late_blight	100.00%
Potato___healthy	100.00%
Tomato___Bacterial_spot	99.76%
Tomato___Early_blight	99.50%
Tomato___Late_blight	99.48%
Tomato___Leaf_Mold	99.47%
Tomato___Septoria_leaf_spot	100.00%
Tomato___Spider_mites___Two_spotted_spider_mite	98.81%
Tomato___Target_Spot	99.29%
Tomato___Tomato_YellowLeaf___Curl_Virus	99.84%
Tomato___Tomato_mosaic_virus	100.00%
Tomato___healthy	100.00%

This performance is comparable to state-of-the-art research benchmarks.

🌱 2. Crop Recommendation System (Random Forest)
📌 Dataset

Used: Crop Recommendation Dataset (Kaggle)
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

📌 Input Features

Nitrogen (N)

Phosphorus (P)

Potassium (K)

Temperature

Humidity

pH

Rainfall

📌 Model Workflow

Train-test split

Pipeline with StandardScaler

RandomForestClassifier (300 trees)

Exports final .joblib model

🚀 Training
python crop_recommadtion/train_crop_model.py

📈 Crop Recommendation Model Performance
🔥 Overall Accuracy: 99.55%
📊 Classification Report (Summary)

Almost all 22 crop classes achieved precision & recall of 1.00

Few crops had slight variation:

blackgram (F1 = 0.97)

jute (F1 = 0.98)

maize (F1 = 0.98)

rice (F1 = 0.97)

Weighted F1-Score = 1.00

💾 Saved Model
crop_recommender_rf.joblib

🛠️ Installation
1. Clone the repository
git clone https://github.com/Krishna2004M/AgriSmart
cd AgriSmart

2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

3. Install dependencies
pip install -r requirements.txt

📦 requirements.txt
numpy
pandas
scikit-learn
matplotlib
tensorflow
keras
opencv-python
Pillow
joblib
python-dotenv
fastapi
uvicorn[standard]
streamlit

🧪 Prediction Examples
1️⃣ Predict Plant Disease
from tensorflow.keras.models import load_model
import cv2, json, numpy as np

model = load_model("Results/plantvillage_b4_best.keras")
labels = json.load(open("Results/class_names.json"))

img = cv2.imread("test_images/t1.JPG")
img = cv2.resize(img, (380, 380))
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
print(labels[np.argmax(pred)])

2️⃣ Predict Recommended Crop
import joblib
import pandas as pd

model = joblib.load("crop_recommadtion/crop_recommender_rf.joblib")

sample = pd.DataFrame([{
    "N": 90,
    "P": 40,
    "K": 40,
    "temperature": 24,
    "humidity": 80,
    "ph": 6.5,
    "rainfall": 200
}])

print(model.predict(sample)[0])

🚀 Future Scope

Add multilingual support (Hindi, Tamil, English)

Deploy API (FastAPI / Flask)

Mobile app (React Native / Flutter)

Fertilizer recommendation module

Weather-based insights

Region-specific crop recommendations

🤝 Contributing

Contributions, ideas, and enhancements are welcome!

📄 License

MIT License © 2025 Krishna M
