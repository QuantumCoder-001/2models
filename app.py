from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- FULL 42-DISEASE RECOMMENDATION MAPPING ---
DISEASE_RECS = {
    "Metabolic/Endocrine": {
        "diseases": ["Diabetes", "Hypoglycemia", "Hypertension", "Hyperthyroidism", "Hypothyroidism"],
        "diet": "Low glycemic index foods, lean proteins, and high-fiber vegetables. Limit sodium/salt.",
        "lifestyle": "Regular 30-min cardio, daily blood sugar/BP monitoring, and consistent sleep schedule."
    },
    "Infectious (Viral/Bacterial/Parasitic)": {
        "diseases": ["Dengue", "Malaria", "Typhoid", "Chicken pox", "Common Cold", "Pneumonia", "Tuberculosis", "AIDS", "Common Cold"],
        "diet": "High-calorie, high-protein diet (eggs, pulses). Stay hydrated with ORS and coconut water.",
        "lifestyle": "Complete bed rest, isolation to prevent spread, and frequent temperature monitoring."
    },
    "Digestive/Hepatic": {
        "diseases": [
            "GERD", "Peptic ulcer diseae", "Gastroenteritis", "Jaundice", "Hepatitis A", 
            "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E", "Alcoholic hepatitis", 
            "Chronic cholestasis", "Dimorphic hemmorhoids(piles)"
        ],
        "diet": "Bland diet (BRAT: Bananas, Rice, Applesauce, Toast). Avoid spice, caffeine, and alcohol.",
        "lifestyle": "Avoid lying down after meals, eat smaller portions, and maintain strict hand hygiene."
    },
    "Dermatological": {
        "diseases": ["Acne", "Psoriasis", "Impetigo", "Fungal infection", "Drug Reaction"],
        "diet": "Hydrate (3L+ water/day). Include Vitamin E and Omega-3 rich foods like nuts and seeds.",
        "lifestyle": "Keep skin clean/dry, use non-comedogenic products, and avoid sharing personal items."
    },
    "Respiratory": {
        "diseases": ["Bronchial Asthma", "Allergy"],
        "diet": "Anti-inflammatory foods (ginger, turmeric). Avoid cold/processed dairy if it triggers mucus.",
        "lifestyle": "Avoid dust/smoke/pollen. Practice breathing exercises (Pranayama) and keep rescue inhalers ready."
    },
    "Neurological/Musculoskeletal": {
        "diseases": [
            "Migraine", "Arthritis", "Osteoarthristis", "Cervical spondylosis", 
            "Paralysis (brain hemorrhage)", "(vertigo) Paroymsal  Positional Vertigo"
        ],
        "diet": "Magnesium-rich foods (spinach, pumpkin seeds). Avoid aged cheese/processed meats for Migraines.",
        "lifestyle": "Maintain ergonomic posture, gentle physiotherapy, and ensure a dark, quiet room for attacks."
    },
    "Vascular/Systemic": {
        "diseases": ["Varicose veins", "Urinary tract infection", "Heart attack"],
        "diet": "Low saturated fats (Heart), Cranberry juice (UTI), High fiber (Varicose veins).",
        "lifestyle": "Avoid long periods of standing/sitting. Regular walking to improve circulation."
    }
}

def get_recommendation(disease_name):
    for category, data in DISEASE_RECS.items():
        if disease_name in data["diseases"]:
            return data
    return {
        "diet": "Maintain a balanced diet and stay hydrated.",
        "lifestyle": "Consult a physician for a specific recovery plan."
    }

# --- LOAD MODELS ---
try:
    symptom_model = joblib.load(os.path.join(BASE_DIR, 'symptom_model.pkl'))
    symptom_encoder = joblib.load(os.path.join(BASE_DIR, 'symptom_encoder.pkl'))
    with open(os.path.join(BASE_DIR, 'symptom_features.json'), 'r') as f:
        features = json.load(f)
    print("✅ Symptom Model Loaded")
except Exception as e:
    print(f"❌ Symptom Model Error: {e}")

try:
    blood_model = joblib.load(os.path.join(BASE_DIR, 'health_model.pkl'))
    blood_encoder = joblib.load(os.path.join(BASE_DIR, 'disease_encoder.pkl'))
    print("✅ Blood Report Model Loaded")
except Exception as e:
    print(f"❌ Blood Model Error: {e}")

# --- ENDPOINTS ---
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
            disease_name = symptom_encoder.inverse_transform([i])[0]
            conf_val = float(probs[i] * 100)
            results.append({
                "disease": disease_name,
                "confidence": f"{round(conf_val, 2)}%",
                "prob_value": conf_val,
                "recommendation": get_recommendation(disease_name) if i == top_3_idx[0] else None
            })
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict-report', methods=['POST'])
def predict_report():
    try:
        data = request.get_json()
        feature_order = [
            'glucose', 'cholesterol', 'hemoglobin', 'platelets', 'wbc', 'rbc', 'hematocrit', 
            'mcv', 'mch', 'mchc', 'insulin', 'bmi', 'systolic', 'diastolic', 'triglycerides',
            'hba1c', 'ldl', 'hdl', 'alt', 'ast', 'heartRate', 'creatinine', 'troponin', 'crp'
        ]
        input_values = [float(data.get(key, 0)) for key in feature_order]
        probs = blood_model.predict_proba(np.array([input_values]))[0]
        sorted_indices = np.argsort(probs)[::-1]

        results = []
        for i in sorted_indices[:3]:  # Top 3
            disease_name = blood_encoder.inverse_transform([i])[0]
            conf_val = float(probs[i] * 100)
            results.append({
                "disease": disease_name,
                "confidence": f"{round(conf_val, 2)}%",
                "prob_value": conf_val,
                "recommendation": get_recommendation(disease_name) if i == sorted_indices[0] else None
            })
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, port=port, host='0.0.0.0')
