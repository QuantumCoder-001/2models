from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json

app = Flask(__name__)
CORS(app) 

# Load the model, encoder, and feature list
model = joblib.load('symptom_model.pkl')
encoder = joblib.load('symptom_encoder.pkl')
with open('symptom_features.json', 'r') as f:
    features = json.load(f)

# Load report prediction model
report_model = joblib.load("health_model.pkl")
disease_encoder = joblib.load("disease_encoder.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_symptoms = data.get('symptoms', [])

        # Create a 132-length input vector of zeros
        input_vector = np.zeros(len(features))
        
        # Set 1 for every symptom the user selected
        for s in user_symptoms:
            if s in features:
                idx = features.index(s)
                input_vector[idx] = 1

        # Get probabilities for all diseases
        probs = model.predict_proba([input_vector])[0]
        
        # Pick the top 3 highest probabilities
        top_3_idx = np.argsort(probs)[-3:][::-1]
        
        results = []
        for i in top_3_idx:
            results.append({
                "disease": encoder.inverse_transform([i])[0],
                "confidence": f"{round(probs[i] * 100, 2)}%"
            })

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict_report', methods=['POST'])
def predict_report():
    try:
        data = request.get_json()

        # Convert input values to numpy array
        input_data = np.array([list(data.values())])

        # Predict probabilities
        probs = report_model.predict_proba(input_data)[0]

        results = []
        for i, prob in enumerate(probs):
            results.append({
                "disease": disease_encoder.inverse_transform([i])[0],
                "confidence": f"{round(prob*100,2)}%"
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)