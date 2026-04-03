import os
import json
import joblib
import numpy as np
import cv2
import gc
import psutil
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- TENSORFLOW MEMORY OPTIMIZATION ---
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.set_visible_devices([], 'GPU')
try:
    tf.config.experimental.set_memory_growth = True
except:
    pass

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- FULL 42-DISEASE PROFESSIONAL RECOMMENDATION ENGINE ---
DETAILED_RECS = {
    "Endocrine/Metabolic": {
        "diseases": ["Diabetes", "Hypoglycemia", "Hyperthyroidism", "Hypothyroidism"],
        "clinical_context": "Focus on stabilizing blood glucose fluctuations and improving insulin sensitivity through low-glycemic indexing.",
        "diet": "Prioritize soluble fiber (steel-cut oats, legumes) and lean alpha-proteins. Replace simple sugars with complex carbohydrates.",
        "lifestyle": "Post-prandial (after meal) 15-minute walks are critical. Maintain a strict circadian rhythm for hormonal balance.",
        "contraindications": "Avoid high-fructose corn syrups, processed 'white' flours, and erratic fasting windows.",
        "next_steps": "HbA1c screening and fasting lipid profile recommended."
    },
    "Cardiovascular": {
        "diseases": ["Hypertension", "Heart attack", "Varicose veins", "Heart Di"],
        "clinical_context": "Management of systemic vascular resistance and reduction of cardiac workload.",
        "diet": "Adhere to the DASH protocol: Reduce sodium intake to <2g/day. Increase Omega-3 via flaxseeds or walnuts.",
        "lifestyle": "Incorporate low-impact steady-state (LISS) cardio. Implement Progressive Muscle Relaxation (PMR) for stress reduction.",
        "contraindications": "Strictly avoid trans-fats, excessive caffeine, and nicotine. Avoid isometric exercises that spike BP.",
        "next_steps": "Daily BP charting and Cardiology consultation for an Echocardiogram."
    },
    "Infectious (Systemic)": {
        "diseases": ["Dengue", "Malaria", "Typhoid", "Chicken pox", "AIDS", "hepatitis A", "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E", "Infectious (Viral/Bacterial/Parasitic)"],
        "clinical_context": "Supporting the immune system during acute viral/bacterial load and preventing dehydration.",
        "diet": "Transition to a high-calorie, soft-residue diet. Focus on electrolyte-rich fluids (ORS, Coconut water).",
        "lifestyle": "Phase 1: Absolute bed rest. Phase 2: Gradual mobilization only after fever subsidence.",
        "contraindications": "Avoid NSAIDs (like Aspirin/Ibuprofen) due to bleeding risks in Dengue. Avoid heavy, oily meals.",
        "next_steps": "Complete Blood Count (CBC) monitoring, specifically Platelet trends."
    },
    "Respiratory": {
        "diseases": ["Bronchial Asthma", "Pneumonia", "Tuberculosis", "Common Cold", "Covid", "Allergy", "covid", "tb", "pnumonia", "pneumothorax"],
        "clinical_context": "Reduction of airway inflammation and improvement of pulmonary ventilation.",
        "diet": "Anti-inflammatory focus: Turmeric, ginger, and Vitamin D rich foods. Ensure high fluid intake to thin mucus.",
        "lifestyle": "Incorporate Diaphragmatic breathing. Practice steam inhalation (40-45C).",
        "contraindications": "Avoid environmental triggers (smoke, dander). Avoid mucus-forming dairy if congestion is high.",
        "next_steps": "Spirometry (PFT) or Chest X-ray follow-up as advised by a Pulmonologist."
    },
    "Gastrointestinal": {
        "diseases": ["GERD", "Peptic ulcer diseae", "Gastroenteritis", "Dimorphic hemmorhoids(piles)", "Peptic ulcer disease"],
        "clinical_context": "Mucosal protection and regulation of gastric acid secretion/bowel motility.",
        "diet": "Adopt the BRAT-P diet (Banana, Rice, Applesauce, Toast, Probiotics). Small, frequent alkaline meals.",
        "lifestyle": "Elevate head of bed by 6 inches. Avoid restrictive clothing around the abdomen.",
        "contraindications": "Avoid late-night meals (within 3 hours of sleep). Eliminate spicy peppers, citrus, and carbonated drinks.",
        "next_steps": "Gastroenterology review; consider H. Pylori testing if ulcers are suspected."
    },
    "Hepatobiliary": {
        "diseases": ["Jaundice", "Alcoholic hepatitis", "Chronic cholestasis"],
        "clinical_context": "Minimizing hepatic metabolic load to facilitate cellular regeneration.",
        "diet": "Ultra-low-fat diet. Primary energy from simple glucose (sugarcane juice, honey). Boiled vegetables only.",
        "lifestyle": "Absolute physical rest for liver cellular repair.",
        "contraindications": "Zero alcohol or hepatotoxic over-the-counter drugs. Avoid all fried/fatty foods.",
        "next_steps": "Liver Function Test (LFT) and Viral Load markers."
    },
    "Neurological": {
        "diseases": ["Migraine", "Paralysis (brain hemorrhage)", "(vertigo) Paroymsal Positional Vertigo"],
        "clinical_context": "Neuro-vascular stabilization and management of sensory triggers.",
        "diet": "Ensure adequate Magnesium and B2 (Riboflavin). Maintain consistent hydration.",
        "lifestyle": "Maintain a 'Headache Diary' to identify triggers. Practice strict sleep hygiene.",
        "contraindications": "Avoid aged cheeses, MSG, and artificial sweeteners.",
        "next_steps": "Neurological physical exam and potential MRI/CT follow-up."
    },
    "Musculoskeletal": {
        "diseases": ["Arthritis", "Osteoarthristis", "Cervical spondylosis", "Osteoarthritis"],
        "clinical_context": "Reduction of joint inflammation and preservation of articular cartilage.",
        "diet": "Turmeric, walnuts, calcium-rich foods. Maintain optimal vitamin D levels.",
        "lifestyle": "Low-impact exercise (swimming). Use heat packs.",
        "contraindications": "Avoid jerky movements. Maintain ergonomic posture.",
        "next_steps": "Rheumatoid factor test and orthopedic evaluation."
    },
    "Dermatological": {
        "diseases": ["Acne", "Psoriasis", "Impetigo", "Fungal infection", "Drug Reaction"],
        "clinical_context": "Restoring epidermal barrier function and regulating sebum/inflammatory responses.",
        "diet": "High water intake. Limit dairy and processed sugar. Zinc-rich foods.",
        "lifestyle": "Use mild cleansers. Change linens frequently.",
        "contraindications": "Do not scratch lesions. Avoid sharing towels.",
        "next_steps": "Dermatological assessment for topical or systemic therapy."
    },
    "Urological": {
        "diseases": ["Urinary tract infection"],
        "clinical_context": "Eradication of pathogenic bacteria from the urinary tract and mucosal soothing.",
        "diet": "Unsweetened cranberry juice, high water intake.",
        "lifestyle": "Strict personal hygiene. Frequent voiding.",
        "contraindications": "Complete the full antibiotic course. Avoid holding urine.",
        "next_steps": "Urine routine and microscopy; culture sensitivity test."
    },
    "Hematological": {
        "diseases": ["Anemia", "Thalasse", "Thromboc"],
        "clinical_context": "Optimization of red blood cell indices, hemoglobin synthesis, and platelet function.",
        "diet": "Iron-rich foods (spinach, red meat) paired with Vitamin C for absorption. Folate-rich greens.",
        "lifestyle": "Pace daily activities to manage fatigue. Prevent physical trauma if platelets are low.",
        "contraindications": "Avoid consuming calcium/dairy simultaneously with iron supplements.",
        "next_steps": "Comprehensive peripheral blood smear and ferritin level check."
    },
    "General/Healthy": {
        "diseases": ["Healthy"],
        "clinical_context": "Maintenance of current physiological homeostasis and preventive care.",
        "diet": "Balanced macronutrients following the Mediterranean or standard balanced plate model.",
        "lifestyle": "150 minutes of moderate aerobic activity weekly. 7-9 hours of sleep.",
        "contraindications": "Avoid sedentary habits and ultra-processed foods.",
        "next_steps": "Annual routine health checkups."
    }
}

def get_detailed_rec(disease_name):
    for category, data in DETAILED_RECS.items():
        if disease_name in data["diseases"]:
            return {
                "category": category,
                "summary": f"Professional management plan for suspected {disease_name}.",
                "clinical_context": data["clinical_context"],
                "personalized_diet": data["diet"],
                "lifestyle_adjustments": data["lifestyle"],
                "strict_contraindications": data["contraindications"],
                "recommended_follow_up": data["next_steps"]
            }
    return {
        "category": "General",
        "summary": f"Observation plan for {disease_name}.",
        "clinical_context": "Generic physiological support during recovery.",
        "personalized_diet": "Balanced macronutrients with high hydration.",
        "lifestyle_adjustments": "Rest and avoidance of strenuous activity.",
        "strict_contraindications": "Avoid self-medication.",
        "recommended_follow_up": "Consult Physician."
    }

# --- GLOBAL MODELS (Baseline) ---
symptom_model = symptom_encoder = features = blood_model = blood_encoder = None

try:
    symptom_model = joblib.load(os.path.join(BASE_DIR, 'symptom_model.pkl'))
    symptom_encoder = joblib.load(os.path.join(BASE_DIR, 'symptom_encoder.pkl'))
    with open(os.path.join(BASE_DIR, 'symptom_features.json'), 'r') as f:
        features = json.load(f)
    blood_model = joblib.load(os.path.join(BASE_DIR, 'health_model.pkl'))
    blood_encoder = joblib.load(os.path.join(BASE_DIR, 'disease_encoder.pkl'))
    print("✅ Base Models Loaded")
except Exception as e:
    print(f"❌ Startup Load Error: {e}")

# --- ENDPOINTS ---

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "memory": f"{psutil.virtual_memory().percent}%"}), 200

# FIXED: This was missing and caused the 404 in your frontend
@app.route('/symptoms', methods=['GET'])
def get_symptoms_list():
    if not features:
        return jsonify({"error": "Features not loaded"}), 500
    return jsonify({"symptoms": features})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_symptoms = data.get('symptoms', [])
        input_vector = np.zeros(len(features))
        for s in user_symptoms:
            if s in features: input_vector[features.index(s)] = 1
        
        probs = symptom_model.predict_proba([input_vector])[0]
        top_3_idx = np.argsort(probs)[-3:][::-1]
        
        results = []
        for rank, i in enumerate(top_3_idx):
            disease_name = symptom_encoder.inverse_transform([i])[0]
            conf_val = float(probs[i] * 100)
            rec = get_detailed_rec(disease_name) if rank == 0 else None
            results.append({
                "disease": disease_name,
                "confidence": f"{round(conf_val, 2)}%",
                "recommendation": rec
            })
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict-report', methods=['POST'])
def predict_report():
    try:
        data = request.get_json()
        # Ensure this order matches your blood_model training order
        feature_order = [
            'glucose', 'cholesterol', 'hemoglobin', 'platelets', 'wbc', 'rbc', 'hematocrit', 
            'mcv', 'mch', 'mchc', 'insulin', 'bmi', 'systolic', 'diastolic', 'triglycerides',
            'hba1c', 'ldl', 'hdl', 'alt', 'ast', 'heartRate', 'creatinine', 'troponin', 'crp'
        ]
        input_values = [float(data.get(key, 0)) for key in feature_order]
        probs = blood_model.predict_proba(np.array([input_values]))[0]
        top_idx = np.argsort(probs)[-1]
        disease = blood_encoder.inverse_transform([top_idx])[0]
        
        return jsonify([{
            "disease": disease,
            "confidence": f"{round(float(probs[top_idx]*100), 2)}%",
            "recommendation": get_detailed_rec(disease)
        }])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict-xray', methods=['POST'])
def predict_xray():
    extractor = svm = scaler = img = None
    try:
        if psutil.virtual_memory().available / (1024 * 1024) < 150:
            return jsonify({"error": "Low memory, please wait"}), 503
        
        # LAZY LOAD HEAVY MODELS
        extractor = tf.keras.models.load_model(os.path.join(BASE_DIR, 'mobilenet_extractor.h5'), compile=False)
        svm = joblib.load(os.path.join(BASE_DIR, 'svm_model.pkl'))
        scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
        classes = joblib.load(os.path.join(BASE_DIR, 'categories.pkl'))
        
        file = request.files['file']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) / 127.5) - 1.0
        img = np.expand_dims(img, axis=0)
        
        feats = extractor.predict(img, verbose=0)
        feats_scaled = scaler.transform(feats)
        probs = svm.predict_proba(feats_scaled)[0]
        res_idx = np.argmax(probs)
        disease = classes[res_idx]
        
        return jsonify({
            "disease": disease,
            "confidence": f"{round(float(probs[res_idx]*100), 2)}%",
            "recommendation": get_detailed_rec(disease)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    finally:
        if extractor: del extractor
        if svm: del svm
        if scaler: del scaler
        if img is not None: del img
        tf.keras.backend.clear_session()
        gc.collect()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, port=port, host='0.0.0.0')
