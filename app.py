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
# Note: Spellings here must match your model's encoder exactly.
DISEASE_RECS = {
    "Metabolic/Endocrine": {
        "diseases": ["Diabetes", "Hypoglycemia", "Hypertension", "Hyperthyroidism", "Hypothyroidism"],
        "diet": "Low glycemic index foods, lean proteins, and high-fiber vegetables. Limit sodium/salt.",
        "lifestyle": "Regular 30-min cardio, daily blood sugar/BP monitoring, and consistent sleep schedule."
    },
    "Infectious (Viral/Bacterial/Parasitic)": {
        "diseases": ["Dengue", "Malaria", "Typhoid", "Chicken pox", "Common Cold", "Pneumonia", "Tuberculosis", "AIDS"],
        "diet": "High-calorie, high-protein diet (eggs, pulses). Stay hydrated with ORS and coconut water.",
        "lifestyle": "Complete bed rest, isolation to prevent spread, and frequent temperature monitoring."
    },
    "Digestive/Hepatic": {
        "diseases": [
            "GERD", "Peptic ulcer disease", "Gastroenteritis", "Jaundice", "Hepatitis A", 
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
            "Migraine", "Arthritis", "Osteoarthritis", "Cervical spondylosis", 
            "Paralysis (brain hemorrhage)", "(vertigo) Paroxysmal Positional Vertigo"
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
    """Searches for a recommendation based on the disease name."""
    for category, data in DISEASE_RECS.items():
        if disease_name in data["diseases"]:
            return {
                "category": category,
                "diet": data["diet"],
                "lifestyle": data["lifestyle"]
            }
    return {
        "category": "General",
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

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "message": "ML Service is running"})

@app.route('/symptoms', methods=['GET'])
def get_symptoms_list():
    return jsonify(features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_symptoms = data.get('symptoms', [])
        
        # Create input vector
        input_vector = np.zeros(len(features))
        for s in user_symptoms:
            if s in features:
                input_vector[features.index(s)] = 1

        # Get probabilities
        probs = symptom_model.predict_proba([input_vector])[0]
        top_3_idx = np.argsort(probs)[-3:][::-1]

        results = []
        for rank, i in enumerate(top_3_idx):
            disease_name = symptom_encoder.inverse_transform([i])[0]
            conf_val = float(probs[i] * 100)
            
            # Attach recommendation only to the top (first) result
            rec = get_recommendation(disease_name) if rank == 0 else None
            
            results.append({
                "disease": disease_name,
                "confidence": f"{round(conf_val, 2)}%",
                "prob_value": conf_val,
                "recommendation": rec
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
        for rank, i in enumerate(sorted_indices[:3]):
            disease_name = blood_encoder.inverse_transform([i])[0]
            conf_val = float(probs[i] * 100)
            
            # Attach recommendation only to the top result
            rec = get_recommendation(disease_name) if rank == 0 else None
            
            results.append({
                "disease": disease_name,
                "confidence": f"{round(conf_val, 2)}%",
                "prob_value": conf_val,
                "recommendation": rec
            })
            
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, port=port, host='0.0.0.0')
