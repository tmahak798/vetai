import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────
UPLOAD_FOLDER   = "static/uploads"
MODEL_PATH      = "best_model.h5"
LABELS_PATH     = "class_labels.json"
MEDICATION_PATH = "medication.csv"
IMG_SIZE        = (224, 224)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ── Load model, labels, medication CSV once at startup ────
print("Loading model...")
model = load_model(MODEL_PATH)
print("✓ Model loaded")

with open(LABELS_PATH, "r") as f:
    # keys are strings when loaded from JSON — convert to int
    raw_labels = json.load(f)
    class_labels = {int(k): v for k, v in raw_labels.items()}
print("✓ Labels loaded:", class_labels)

medication_df = pd.read_csv(MEDICATION_PATH)
medication_df["disease"] = medication_df["disease"].str.strip()
print("✓ Medication CSV loaded")
print("✓ App ready\n")

# ── Helper: preprocess uploaded image ─────────────────────
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ── Helper: get medication info for predicted disease ──────
def get_medication(disease_name):
    row = medication_df[medication_df["disease"] == disease_name]
    if row.empty:
        return {
            "medicine":    "Information not available",
            "treatment":   "Please consult a veterinarian",
            "precautions": "Isolate the animal and seek professional help"
        }
    return {
        "medicine":    row.iloc[0]["medicine"],
        "treatment":   row.iloc[0]["treatment"],
        "precautions": row.iloc[0]["precautions"]
    }

# ── Routes ─────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save uploaded image
    img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(img_path)

    # Preprocess and predict
    img_array    = preprocess_image(img_path)
    predictions  = model.predict(img_array)
    predicted_idx = int(np.argmax(predictions[0]))
    confidence    = float(np.max(predictions[0])) * 100
    disease_name  = class_labels[predicted_idx]

    # Get medication info
    med_info = get_medication(disease_name)

    return jsonify({
        "disease":     disease_name,
        "confidence":  round(confidence, 2),
        "medicine":    med_info["medicine"],
        "treatment":   med_info["treatment"],
        "precautions": med_info["precautions"],
        "image_path":  f"static/uploads/{file.filename}"
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "MobileNetV2", "classes": list(class_labels.values())})

if __name__ == "__main__":
    app.run(debug=True)