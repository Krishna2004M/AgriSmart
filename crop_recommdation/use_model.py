import joblib
import pandas as pd

# Load trained pipeline
model = joblib.load("crop_recommender_rf.joblib")

feature_order = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

def recommend_crops(features_dict, top_k=3):
    """
    features_dict = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 24.0,
        "humidity": 80.0,
        "ph": 6.5,
        "rainfall": 200.0
    }
    """
    # Convert dict → DataFrame (keeps feature names)
    x = pd.DataFrame([features_dict], columns=feature_order)

    # Predict probabilities
    probs = model.predict_proba(x)[0]
    classes = model.classes_

    # Top-k crops
    top_idx = probs.argsort()[::-1][:top_k]

    return [
        {
            "crop": classes[i],
            "score": round(float(probs[i] * 100), 2)
        }
        for i in top_idx
    ]

if __name__ == "__main__":
    sample_input = {
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.87,
  "humidity": 82.00,
  "ph": 6.50,
  "rainfall": 202.93
}


    recs = recommend_crops(sample_input, top_k=3)
    print("Top recommendations:")
    for r in recs:
        print(f"{r['crop']} — {r['score']}%")
