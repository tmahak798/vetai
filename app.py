import os
import gc
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from PIL import Image

# Use tflite-runtime instead of full tensorflow
from tensorflow import lite as tflite
app = Flask(__name__)

# ── Config ─────────────────────────────────────────────────
UPLOAD_FOLDER   = "static/uploads"
MODEL_PATH      = "model.tflite"
LABELS_PATH     = "class_labels.json"
MEDICATION_PATH = "medication.csv"
IMG_SIZE        = (224, 224)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Load TFLite model ───────────────────────────────────────
print("Loading TFLite model...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✓ TFLite model loaded")

# ── Load labels ─────────────────────────────────────────────
with open(LABELS_PATH, "r") as f:
    raw_labels = json.load(f)
    class_labels = {int(k): v for k, v in raw_labels.items()}
print("✓ Labels loaded:", class_labels)

# ── Load medication CSV ─────────────────────────────────────
medication_df = pd.read_csv(MEDICATION_PATH)
medication_df["disease"] = medication_df["disease"].str.strip()
print("✓ Medication CSV loaded")
print("✓ App ready\n")

# ── Helpers ─────────────────────────────────────────────────
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_tflite(img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]

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

# ── Routes ──────────────────────────────────────────────────
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

    img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(img_path)

    try:
        img_array     = preprocess_image(img_path)
        predictions   = predict_tflite(img_array)
        predicted_idx = int(np.argmax(predictions))
        confidence    = float(np.max(predictions)) * 100
        disease_name  = class_labels[predicted_idx]
        med_info      = get_medication(disease_name)

        del img_array, predictions
        gc.collect()

        return jsonify({
            "disease":     disease_name,
            "confidence":  round(confidence, 2),
            "medicine":    med_info["medicine"],
            "treatment":   med_info["treatment"],
            "precautions": med_info["precautions"],
            "image_path":  f"static/uploads/{file.filename}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({
        "status":  "ok",
        "model":   "MobileNetV2 TFLite",
        "classes": list(class_labels.values())
    })

if __name__ == "__main__":
    app.run(debug=True)