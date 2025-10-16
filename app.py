# app.py
from flask import Flask, request, jsonify
import joblib
import os

MODEL_PATH = 'model.pkl'

app = Flask(__name__)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Run `python train_model.py` first to create model.pkl")

model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return jsonify({"message": "Sentiment Analysis API. POST /predict with JSON {'text': '...'}"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "JSON with 'text' field required"}), 400

    text = data['text']
    probs = model.predict_proba([text])[0]
    pred = int(model.predict([text])[0])
    confidence = float(probs[pred])
    label = "Positive" if pred == 1 else "Negative"
    return jsonify({
        "text": text,
        "prediction": label,
        "confidence": round(confidence, 3)
    }), 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)
