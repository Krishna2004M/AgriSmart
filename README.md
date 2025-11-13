````markdown
# 🌾 AgriSmart – AI-Powered Crop & Plant Disease Identification

**Author:** Krishna M (AgriSens)  

AgriSmart is an AI-driven agricultural support system that helps farmers with:

- 🍃 **Plant Disease Detection** using a deep learning EfficientNetB4 model  
- 🌱 **Crop Recommendation** using a Random Forest classifier  
- 📊 **High-accuracy, field-ready predictions**  
- ⚙️ **Compatible with CPU, GPU, and WSL** environments  

Both ML & DL models are trained using high-quality Kaggle datasets and optimized for real-world usage.

---

## 🚀 Features

- **Plant Disease Classification**
  - EfficientNetB4 backbone with ImageNet weights  
  - Strong data augmentation (flip, rotation, zoom, contrast)  
  - Two-stage training (frozen backbone → fine-tuning)  
  - Mixed precision & GPU-optimized input pipeline  

- **Crop Recommendation**
  - RandomForestClassifier wrapped in a scikit-learn Pipeline  
  - Uses soil & weather features (N, P, K, temperature, humidity, pH, rainfall)  
  - Achieves >99% accuracy on the Kaggle crop dataset  

- **Analysis & Visualization**
  - Per-class accuracy report for PlantVillage  
  - Saved `y_true` and `y_pred` for metric analysis  
  - Scripts to plot test metrics and final accuracy  

---

## 📊 Datasets

- **Plant Disease Detection**

  - **Dataset:** PlantVillage (Kaggle)  
  - **Link:** <https://www.kaggle.com/datasets/emmarex/plantdisease>

- **Crop Recommendation**

  - **Dataset:** Crop Recommendation (Kaggle)  
  - **Link:** <https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset>

You should download these datasets manually and place them in the appropriate folders as described below.

---

## 📁 Project Structure

```text
AGRI/
│
├── crop_recommadtion/
│   ├── Crop_recommendation.csv
│   ├── crop_recommender_rf.joblib          # Saved RandomForest model
│   ├── train_crop_model.py                 # Train crop recommendation model
│   └── use_model.py                        # Example usage / inference script
│
├── Plant_disease/
│   ├── analyze_predictions.py              # Analyze y_true / y_pred
│   ├── Make_Split.py                       # Create train/val splits
│   ├── paper_figures.py                    # Helper for paper plots (optional)
│   ├── plot_final_accuracy.py              # Plot final accuracy curves
│   ├── plot_test_metrics.py                # Plot test metrics
│   ├── predict_batch.py                    # Run batch predictions on a folder
│   ├── predict_one.py                      # Run prediction on a single image
│   └── train.py                            # EfficientNetB4 training script
│
├── PlantVillage/                           # Original PlantVillage dataset
│   ...                                     # (raw images from Kaggle)
│
├── PlantVillage_split/                     # Split dataset
│   ├── train/
│   └── val/
│
├── Results/
│   ├── analysis/
│   │   └── report.txt                      # Per-class accuracy report
│   ├── tb_logs/                            # TensorBoard logs
│   ├── class_names.json                    # Mapping of class indices to names
│   ├── y_pred.npy                          # Predicted labels on val set
│   └── y_true.npy                          # Ground truth labels on val set
│
├── test_images/
│   └── t1.JPG                              # Sample test image
│
└── README.md
````

---

## 📦 Model Files & Large File Note

The trained **Keras model** for plant disease classification is **not** stored in the GitHub repository because of its large size.

* Expected file path:

  ```text
  Results/plantvillage_b4_best.keras
  ```

* This file will be shared via **cloud storage (e.g., Google Drive)**:

  > 🔗 **Model Download Link:** *to be added*

After downloading, place it at:

```text
AGRI/Results/plantvillage_b4_best.keras
```

The crop recommendation model **is small** and is stored as:

```text
crop_recommadtion/crop_recommender_rf.joblib
```

---

## 🛠 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Krishna2004M/AgriSmart.git
cd AgriSmart
```

### 2️⃣ Create & Activate a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🌱 A. Crop Recommendation

#### 1. Train the Model

Make sure `Crop_recommendation.csv` is inside `crop_recommadtion/`, then run:

```bash
cd crop_recommadtion
python train_crop_model.py
```

This will:

* Train a RandomForest model inside a scikit-learn Pipeline
* Print the **accuracy** and **classification report**
* Save the model as `crop_recommender_rf.joblib`

#### 2. Use the Trained Model

Example (inside `crop_recommadtion/use_model.py`):

```python
import joblib
import pandas as pd

model = joblib.load("crop_recommender_rf.joblib")

sample = pd.DataFrame([{
    "N": 90,
    "P": 40,
    "K": 40,
    "temperature": 24,
    "humidity": 80,
    "ph": 6.5,
    "rainfall": 200
}])

print("Recommended crop:", model.predict(sample)[0])
```

Run:

```bash
python use_model.py
```

---

### 🍃 B. Plant Disease Classification

#### 1. Prepare the Dataset

* Download the **PlantVillage** dataset from Kaggle.
* Keep raw images under `PlantVillage/`.
* Use `Make_Split.py` (if needed) to create `PlantVillage_split/train` and `PlantVillage_split/val`.

#### 2. Train the EfficientNetB4 Model

From the project root:

```bash
cd Plant_disease
python train.py
```

This script will:

* Build the data pipeline with augmentation
* Train in **two stages** (head → fine-tune)
* Save:

  * `Results/plantvillage_b4_best.keras`
  * `Results/class_names.json`
  * `Results/y_true.npy`
  * `Results/y_pred.npy`
  * TensorBoard logs under `Results/tb_logs/`

#### 3. Run Inference on a Single Image

```bash
cd Plant_disease
python predict_one.py --image_path ../test_images/t1.JPG
```

*(You can adapt this script to your own paths.)*

---

## 📈 Results

### 🍃 Plant Disease Classification (EfficientNetB4)

From `Results/analysis/report.txt`:

* **Overall accuracy:** **99.66%** on **4127 validation images**

#### Per-Class Accuracy

| Class                                           | Accuracy |
| ----------------------------------------------- | -------- |
| Pepper_bell___Bacterial_spot                    | 99.50%   |
| Pepper_bell___healthy                           | 99.66%   |
| Potato___Early_blight                           | 100.00%  |
| Potato___Late_blight                            | 100.00%  |
| Potato___healthy                                | 100.00%  |
| Tomato___Bacterial_spot                         | 99.76%   |
| Tomato___Early_blight                           | 99.50%   |
| Tomato___Late_blight                            | 99.48%   |
| Tomato___Leaf_Mold                              | 99.47%   |
| Tomato___Septoria_leaf_spot                     | 100.00%  |
| Tomato___Spider_mites___Two_spotted_spider_mite | 98.81%   |
| Tomato___Target_Spot                            | 99.29%   |
| Tomato___Tomato_YellowLeaf___Curl_Virus         | 99.84%   |
| Tomato___Tomato_mosaic_virus                    | 100.00%  |
| Tomato___healthy                                | 100.00%  |

---

### 🌱 Crop Recommendation (Random Forest)

From `crop_recommadtion/train_crop_model.py` output:

* **Overall accuracy:** **99.55%**

Most classes achieve precision & recall of **1.00**, with only minor drops (F1 ≈ 0.97–0.98) for a few crops such as **blackgram, jute, maize, rice**.
Overall macro and weighted F1-scores are effectively **1.00**.

---

## 🔮 Future Work

* FastAPI / Streamlit web interface for farmers
* Mobile app integration (Flutter / React Native)
* Multilingual support (e.g., Hindi, Tamil, English)
* Fertilizer & pesticide recommendations
* Weather-aware crop suggestions
* Region-specific fine-tuning for Indian states

---

## 📄 License

This project is released under the **MIT License**.
© 2025 **Krishna M (AgriSens)**


