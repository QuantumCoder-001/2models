import os
import json
import joblib
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

app = Flask(__name__)
CORS(app)

# 5. Add Request Size Limit (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- RECOMMENDATION MAPPING (Kept for Logic) ---
DISEASE_RECS = {
    "Metabolic/Endocrine": {
        "diseases": ["Diabetes", "Hypoglycemia", "Hypertension", "Hyperthyroidism", "Hypothyroidism"],
        "diet": "Low glycemic index foods, lean proteins. Limit sodium.",
        "lifestyle": "Regular 30-min cardio, daily monitoring.",
        "precautions": "Avoid skipping meals. Carry a sugar source."
    },
    "Infectious (Viral/Bacterial/Parasitic)": {
        "diseases": ["Dengue", "Malaria", "Typhoid", "Chicken pox", "Common Cold", "Pneumonia", "Tuberculosis", "AIDS", "Covid"],
        "diet": "High-calorie, high-protein diet. Stay hydrated.",
        "lifestyle": "Complete bed rest, isolation to prevent spread.",
        "precautions": "Avoid crowded places. Do not self-medicate."
    },
    "Digestive/Hepatic": {
        "diseases": [
            "GERD", "Peptic ulcer disease", "Gastroenteritis", "Jaundice", "Hepatitis A", 
            "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E", "Alcoholic hepatitis", 
            "Chronic cholestasis", "Dimorphic hemmorhoids(piles)"
        ],
        "diet": "Bland diet (Bananas, Rice, Toast). Avoid spice and alcohol.",
        "lifestyle": "Smaller portions, avoid lying down after meals.",
        "precautions": "Drink filtered water. Stop alcohol immediately."
    },
    "Dermatological": {
        "diseases": ["Acne", "Psoriasis", "Impetigo", "Fungal infection", "Drug Reaction"],
        "diet": "Hydrate (3L+ water). Vitamin E rich foods.",
        "lifestyle": "Keep skin clean/dry, use non-comedogenic products.",
        "precautions": "Do not scratch lesions. Avoid sharing towels."
    },
    "Respiratory": {
        "diseases": ["Bronchial Asthma", "Allergy", "Normal"],
        "diet": "Anti-inflammatory foods (ginger, turmeric).",
        "lifestyle": "Avoid dust/smoke. Practice breathing exercises.",
        "precautions": "Keep environment dust-free. Carry an inhaler if asthmatic."
    },
    "Neurological/Musculoskeletal": {
        "diseases": [
            "Migraine", "Arthritis", "Osteoarthritis", "Cervical spondylosis", 
            "Paralysis (brain hemorrhage)", "(vertigo) Paroxysmal Positional Vertigo"
        ],
        "diet": "Magnesium-rich foods (spinach, seeds).",
        "lifestyle": "Maintain ergonomic posture, physiotherapy.",
        "precautions": "Avoid sudden head movements. Do not drive during dizzy spells."
    },
    "Vascular/Systemic": {
        "diseases": ["Varicose veins", "Urinary tract infection", "Heart attack"],
        "diet": "Low saturated fats, Cranberry juice, High fiber.",
        "lifestyle": "Avoid long standing. Regular walking.",
        "precautions": "For chest pain, seek emergency help. Drink plenty of water for UTI."
    }
}

def get_recommendation(disease_name):
    for category, data in DISEASE_RECS.items():
        if disease_name in data["diseases"]:
            return {
                "category": category,
                "diet": data["diet"],
                "lifestyle": data["lifestyle"],
                "precautions": data["precautions"]
            }
    return {
        "category": "General",
        "diet": "Maintain a balanced diet and stay hydrated.",
        "lifestyle": "Consult a physician for a specific recovery plan.",
        "precautions": "Monitor symptoms closely. Seek professional medical advice."
    }

# --- 1. GRANULAR MODEL LOADING ---

# Global variables initialized to None
symptom_model = symptom_encoder = features = None
blood_model = blood_encoder = None
mobilenet_extractor = xray_svm = xray_scaler = xray_classes = None

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

try:
    mobilenet_extractor = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    # 3. Freeze weights for inference
    mobilenet_extractor.trainable = False 
    
    xray_svm = joblib.load(os.path.join(BASE_DIR, 'svm_model.pkl'))
    xray_scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    xray_classes = joblib.load(os.path.join(BASE_DIR, 'categories.pkl'))
    print("✅ X-ray Hybrid Model (MobileNetV2) Loaded")
except Exception as e:
    print(f"❌ X-ray Model Error: {e}")

# --- 2. MODEL AVAILABILITY ENDPOINT ---
@app.route('/model-status', methods=['GET'])
def model_status():
    return jsonify({
        "symptom_model": symptom_model is not None,
        "blood_model": blood_model is not None,
        "xray_model": xray_svm is not None,
        "feature_extractor": mobilenet_extractor is not None
    })

@app.route('/')
def home():
    return "Health Prediction API (MobileNetV2) is Running", 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "online"})

@app.route('/symptoms', methods=['GET'])
def get_symptoms_list():
    if not features:
        return jsonify({"error": "Symptom features not loaded"}), 500
    return jsonify({"symptoms": features})

# --- PREDICTION ENDPOINTS ---

@app.route('/predict', methods=['POST'])
def predict():
    if not symptom_model:
        return jsonify({"error": "Symptom model is unavailable"}), 503
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
        for rank, i in enumerate(top_3_idx):
            disease_name = symptom_encoder.inverse_transform([i])[0]
            conf_val = float(probs[i] * 100)
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
    if not blood_model:
        return jsonify({"error": "Blood model is unavailable"}), 503
    try:
        data = request.get_json()
        feature_order = ['glucose', 'cholesterol', 'hemoglobin', 'platelets', 'wbc', 'rbc', 'hematocrit', 'mcv', 'mch', 'mchc', 'insulin', 'bmi', 'systolic', 'diastolic', 'triglycerides', 'hba1c', 'ldl', 'hdl', 'alt', 'ast', 'heartRate', 'creatinine', 'troponin', 'crp']
        input_values = [float(data.get(key, 0)) for key in feature_order]
        probs = blood_model.predict_proba(np.array([input_values]))[0]
        sorted_indices = np.argsort(probs)[::-1]
        results = []
        for rank, i in enumerate(sorted_indices[:3]):
            disease_name = blood_encoder.inverse_transform([i])[0]
            conf_val = float(probs[i] * 100)
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

@app.route('/predict-xray', methods=['POST'])
def predict_xray():
    # 4. Input Validation
    if not xray_svm or not mobilenet_extractor:
        return jsonify({"error": "X-ray model is unavailable"}), 503
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
            
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        
        cnn_features = mobilenet_extractor.predict(img)
        features_scaled = xray_scaler.transform(cnn_features)
        probs = xray_svm.predict_proba(features_scaled)[0]
        
        pred_idx = np.argmax(probs)
        disease_name = xray_classes[pred_idx]
        conf_val = float(probs[pred_idx] * 100)

        return jsonify({
            "disease": disease_name,
            "confidence": f"{round(conf_val, 2)}%",
            "recommendation": get_recommendation(disease_name)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode, port=port, host='0.0.0.0')
