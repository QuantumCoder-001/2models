from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- LOAD SYMPTOM MODEL ---
try:
    # Use consistent variable names to avoid overwriting
    symptom_model = joblib.load(os.path.join(BASE_DIR, 'symptom_model.pkl'))
    symptom_encoder = joblib.load(os.path.join(BASE_DIR, 'symptom_encoder.pkl'))
    with open(os.path.join(BASE_DIR, 'symptom_features.json'), 'r') as f:
        features = json.load(f)
    print("✅ Symptom Model Loaded")
except Exception as e:
    print(f"❌ Symptom Model Error: {e}")

# --- LOAD BLOOD REPORT MODEL ---
try:
    # UPDATED: Matches your uploaded filename 'health_model.pkl'
    blood_model = joblib.load(os.path.join(BASE_DIR, 'health_model.pkl'))
    # UPDATED: You uploaded 'disease_encoder.pkl', so we must load and use it
    blood_encoder = joblib.load(os.path.join(BASE_DIR, 'disease_encoder.pkl'))
    print("✅ Blood Report Model Loaded")
except Exception as e:
    print(f"❌ Blood Model Error: {e}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "endpoints": ["/health", "/predict", "/predict-report", "/symptoms"]
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_symptoms = data.get('symptoms', [])
        input_vector = np.zeros(len(features))
        for s in user_symptoms:
            if s in features:
                input_vector[features.index(s)] = 1

        probs = symptom_model.predict_proba([input_vector])[0]
        top_3_idx = np.argsort(probs)[-3:][::-1]

        results = []
        for i in top_3_idx:
            results.append({
                "disease": symptom_encoder.inverse_transform([i])[0],
                "confidence": f"{round(probs[i] * 100, 2)}%"
            })
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict-report', methods=['POST'])
@app.route('/predict-report', methods=['POST'])
def predict_report():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        feature_order = [
            'glucose', 'cholesterol', 'hemoglobin', 'platelets',
            'wbc', 'rbc', 'hematocrit', 'mcv', 'mch', 'mchc',
            'insulin', 'bmi', 'systolic', 'diastolic', 'triglycerides',
            'hba1c', 'ldl', 'hdl', 'alt', 'ast', 'heartRate',
            'creatinine', 'troponin', 'crp'
        ]

        # MODIFIED: If key is missing or None, use np.nan instead of 0
        input_values = []
        for key in feature_order:
            val = data.get(key)
            if val is None or val == "":
                input_values.append(np.nan) # XGBoost recognizes np.nan as a missing value
            else:
                input_values.append(float(val))

        final_features = np.array([input_values])

        # Get probabilities
        probs = blood_model.predict_proba(final_features)[0]

        # Get all predictions sorted by confidence
        sorted_indices = np.argsort(probs)[::-1]

        results = []
        for i in sorted_indices:
            if probs[i] > 0.01: # Only include results > 1% confidence
                results.append({
                    "disease": blood_encoder.inverse_transform([i])[0],
                    "confidence": f"{round(probs[i] * 100, 2)}%",
                    "confidence_level": "High" if probs[i] >= 0.7 else "Medium" if probs[i] >= 0.4 else "Low"
                })

        return jsonify(results)
    except Exception as e:
        print(f"❌ Blood Prediction Error: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify({"success": True, "symptoms": features})

if __name__ == '__main__':
    # Use the PORT environment variable for Render
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, port=port, host='0.0.0.0')

