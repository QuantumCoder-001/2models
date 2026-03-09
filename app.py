from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- RECOMMENDATION ENGINE DATA ---
# Categorized into 8 Clinical Groups for scalability
DISEASE_ADVICE = {
    "Infectious": {
        "diseases": ["AIDS", "Chicken pox", "Dengue", "Malaria", "Typhoid", "hepatitis A", "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E", "Tuberculosis"],
        "diet": ["High protein (eggs, lean meat)", "Hydrating fluids (ORS, Coconut water)", "Soft foods like Khichdi"],
        "lifestyle": ["Complete bed rest", "Isolation to prevent spread", "Monitor body temperature"],
        "precaution": ["Avoid physical exertion", "Stay away from public places"]
    },
    "Respiratory": {
        "diseases": ["Bronchial Asthma", "Common Cold", "Pneumonia"],
        "diet": ["Warm liquids (soups, herbal tea)", "Ginger and Honey", "Avoid cold/processed foods"],
        "lifestyle": ["Steam inhalation 2x daily", "Avoid dust and smoke", "Keep the room ventilated"],
        "precaution": ["Use a mask in crowded areas", "Avoid sudden temperature changes"]
    },
    "Digestive": {
        "diseases": ["GERD", "Gastroenteritis", "Peptic ulcer diseae", "Jaundice", "Chronic cholestasis", "Dimorphic hemmorhoids(piles)"],
        "diet": ["Bland diet (Banana, Rice, Toast)", "Avoid spicy and oily food", "Small, frequent meals"],
        "lifestyle": ["Stay upright for 2 hours after eating", "Stay hydrated", "Avoid alcohol and caffeine"],
        "precaution": ["Avoid self-medicating with painkillers", "Check for signs of dehydration"]
    },
    "Metabolic": {
        "diseases": ["Diabetes", "Hypoglycemia", "Hypothyroidism", "Hyperthyroidism"],
        "diet": ["Low Glycemic Index foods", "High fiber (oats, sprouts)", "Consistent meal timings"],
        "lifestyle": ["Regular 30-min brisk walk", "Stress management", "Adequate sleep (7-8 hours)"],
        "precaution": ["Regularly monitor blood sugar levels", "Carry a sugar snack for emergencies"]
    },
    "Musculoskeletal": {
        "diseases": ["Arthritis", "Osteoarthristis", "Cervical spondylosis"],
        "diet": ["Calcium-rich foods (dairy, ragi)", "Anti-inflammatory (Turmeric, Ginger)", "Omega-3 (walnuts)"],
        "lifestyle": ["Low-impact exercises (Yoga, Swimming)", "Correct posture while sitting", "Warm compress on joints"],
        "precaution": ["Avoid lifting heavy weights", "Avoid long periods of inactivity"]
    },
    "Dermatological": {
        "diseases": ["Acne", "Fungal infection", "Psoriasis", "Impetigo", "Drug Reaction"],
        "diet": ["Drink 3-4 liters of water", "Avoid high-sugar foods", "Include Vitamin C and E"],
        "lifestyle": ["Maintain skin hygiene", "Use mild, soap-free cleansers", "Keep affected area dry"],
        "precaution": ["Do not scratch or pop lesions", "Avoid sharing personal items like towels"]
    },
    "Cardiovascular": {
        "diseases": ["Heart attack", "Hypertension", "Varicose veins"],
        "diet": ["Low sodium (salt) diet", "Heart-healthy fats (olive oil)", "Limit red meat"],
        "lifestyle": ["Weight management", "Quit smoking and limit alcohol", "Daily light cardio"],
        "precaution": ["Emergency contact number on speed dial", "Regular BP monitoring"]
    },
    "Neurological": {
        "diseases": ["(vertigo) Paroymsal  Positional Vertigo", "Migraine", "Paralysis (brain hemorrhage)"],
        "diet": ["Magnesium-rich foods", "Consistent hydration", "Avoid triggers like aged cheese"],
        "lifestyle": ["Dark, quiet room rest during episodes", "Regular sleep cycle", "Physical therapy"],
        "precaution": ["Avoid sudden head movements", "Identify and avoid sensory triggers"]
    }
}

def get_recommendations(disease_name, confidence_val):
    """Helper to fetch advice based on clinical group and confidence level."""
    # Find the group the disease belongs to
    selected_group = None
    for group, details in DISEASE_ADVICE.items():
        if disease_name in details["diseases"]:
            selected_group = details
            break
    
    if not selected_group:
        return {"general": "Consult a healthcare professional for personalized advice."}

    # Dynamic urgency based on your dashboard logic (<30 Green, 30-60 Orange, >60 Red)
    if confidence_val < 30:
        return {
            "urgency": "Low (Wellness Focus)",
            "advice": ["Maintain general hydration", "Monitor symptoms", "Prioritize sleep"]
        }
    elif 30 <= confidence_val <= 60:
        return {
            "urgency": "Moderate (Targeted Prevention)",
            "diet": selected_group["diet"][:2],
            "lifestyle": selected_group["lifestyle"][:1]
        }
    else:
        return {
            "urgency": "High (Clinical Urgent)",
            "diet": selected_group["diet"],
            "lifestyle": selected_group["lifestyle"],
            "precaution": selected_group["precaution"] + ["CONSULT A DOCTOR IMMEDIATELY"]
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
            confidence_pct = round(probs[i] * 100, 2)
            
            results.append({
                "disease": disease_name,
                "confidence": f"{confidence_pct}%",
                "recommendations": get_recommendations(disease_name, confidence_pct)
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
        input_values = [float(data.get(key)) if data.get(key) else np.nan for key in feature_order]
        final_features = np.array([input_values])

        probs = blood_model.predict_proba(final_features)[0]
        sorted_indices = np.argsort(probs)[::-1]

        results = []
        for i in sorted_indices:
            if probs[i] > 0.01:
                disease_name = blood_encoder.inverse_transform([i])[0]
                confidence_pct = round(probs[i] * 100, 2)
                results.append({
                    "disease": disease_name,
                    "confidence": f"{confidence_pct}%",
                    "confidence_level": "High" if probs[i] >= 0.7 else "Medium" if probs[i] >= 0.4 else "Low",
                    "recommendations": get_recommendations(disease_name, confidence_pct)
                })
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify({"success": True, "symptoms": features})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, port=port, host='0.0.0.0')
