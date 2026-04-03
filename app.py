import os
import json
import joblib
import numpy as np
import cv2
import gc
import psutil
import re
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None

# --- TENSORFLOW CONFIG ---
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

# --- NORMALIZATION RANGES (Real World -> 0-1 Scale) ---
BLOOD_RANGES = {
    'glucose': [60, 200], 'cholesterol': [120, 300], 'hemoglobin': [8, 18],
    'platelets': [100, 450], 'wbc': [3, 15], 'rbc': [3, 7],
    'hematocrit': [30, 55], 'mcv': [70, 110], 'mch': [20, 40],
    'mchc': [28, 38], 'insulin': [2, 30], 'bmi': [15, 45],
    'systolic': [90, 180], 'diastolic': [60, 110], 'triglycerides': [50, 250],
    'hba1c': [4, 12], 'ldl': [50, 200], 'hdl': [20, 100],
    'alt': [5, 60], 'ast': [5, 60], 'heartRate': [50, 120],
    'creatinine': [0.5, 2.0], 'troponin': [0, 0.5], 'crp': [0, 20]
}

def normalize_input(data):
    feature_order = [
        'glucose', 'cholesterol', 'hemoglobin', 'platelets', 'wbc', 'rbc', 'hematocrit', 
        'mcv', 'mch', 'mchc', 'insulin', 'bmi', 'systolic', 'diastolic', 'triglycerides',
        'hba1c', 'ldl', 'hdl', 'alt', 'ast', 'heartRate', 'creatinine', 'troponin', 'crp'
    ]
    normalized = []
    for key in feature_order:
        val = float(data.get(key, 0))
        r_min, r_max = BLOOD_RANGES[key]
        norm = (val - r_min) / (r_max - r_min)
        normalized.append(np.clip(norm, 0, 1))
    return np.array([normalized])

# --- FULL PROFESSIONAL RECOMMENDATION ENGINE ---
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
    name_clean = str(disease_name).strip().lower()
    for category, data in DETAILED_RECS.items():
        if any(d.lower() in name_clean for d in data["diseases"]):
            return {
                "category": category,
                "summary": f"Professional management plan for suspected {disease_name}.",
                "clinical_context": data["clinical_context"],
                "diet": data["diet"],
                "lifestyle": data["lifestyle"],
                "precautions": data["contraindications"],
                "next_steps": data["next_steps"]
            }
    return {
        "category": "General",
        "diet": "Maintain a balanced diet and stay hydrated.",
        "lifestyle": "Consult a physician for a specific recovery plan.",
        "precautions": "Monitor symptoms. Seek professional advice.",
        "next_steps": "Physician consultation."
    }

# --- LOAD MODELS ---
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
    print(f"❌ Startup Error: {e}")

@app.route('/')
def home():
    return "Health API Online", 200

@app.route('/health')
def health():
    return jsonify({"status": "online", "mem": f"{psutil.virtual_memory().percent}%"}), 200

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify({"symptoms": features})

# --- NEW: AUTO-FILL EXTRACTION ROUTE ---
@app.route('/extract-report', methods=['POST'])
def extract_report():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    img = Image.open(file.stream)
    text = pytesseract.image_to_string(img)
    
    # Simple regex parser to find values next to keywords
    extracted_values = {}
    for key in BLOOD_RANGES.keys():
        # Look for the keyword followed by optional colon/spaces and a number
        pattern = re.compile(rf"{key}[:\s]*(\d+\.?\d*)", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            extracted_values[key] = match.group(1)
            
    return jsonify(extracted_values)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_symptoms = data.get('symptoms', [])
        vec = np.zeros(len(features))
        for s in user_symptoms:
            if s in features: vec[features.index(s)] = 1
        probs = symptom_model.predict_proba([vec])[0]
        idx = np.argsort(probs)[-1]
        name = symptom_encoder.inverse_transform([idx])[0]
        return jsonify([{
            "disease": name,
            "confidence": f"{round(float(probs[idx]*100), 2)}%",
            "recommendation": get_detailed_rec(name)
        }])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict-report', methods=['POST'])
def predict_report():
    try:
        data = request.get_json()
        # Automatically normalize real values to 0-1 scale before predicting
        vals = normalize_input(data)
        probs = blood_model.predict_proba(vals)[0]
        idx = np.argsort(probs)[-1]
        name = blood_encoder.inverse_transform([idx])[0]
        return jsonify([{
            "disease": name,
            "confidence": f"{round(float(probs[idx]*100), 2)}%",
            "recommendation": get_detailed_rec(name)
        }])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict-xray', methods=['POST'])
def predict_xray():
    extractor = None
    m_path = os.path.join(BASE_DIR, 'mobilenet_extractor.h5')
    if not os.path.exists(m_path):
        return jsonify({"error": f"Model missing at {m_path}"}), 500
    try:
        if psutil.virtual_memory().available / (1024 * 1024) < 100:
            return jsonify({"error": "Low memory"}), 503
        extractor = tf.keras.models.load_model(m_path, compile=False)
        svm = joblib.load(os.path.join(BASE_DIR, 'svm_model.pkl'))
        scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
        classes = joblib.load(os.path.join(BASE_DIR, 'categories.pkl'))
        
        file = request.files['file']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5) - 1.0
        
        feats = extractor.predict(np.expand_dims(img, axis=0), verbose=0)
        probs = svm.predict_proba(scaler.transform(feats))[0]
        res_idx = np.argmax(probs)
        name = classes[res_idx]
        
        return jsonify({
            "disease": name,
            "confidence": f"{round(float(probs[res_idx]*100), 2)}%",
            "recommendation": get_detailed_rec(name)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    finally:
        if extractor: del extractor
        tf.keras.backend.clear_session()
        gc.collect()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, port=port, host='0.0.0.0')
