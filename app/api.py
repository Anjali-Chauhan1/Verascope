import os
import sys
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Ensure project root in path
sys.path.insert(0, os.path.abspath("."))
from src.predict import FakeNewsPredictor

app = Flask(__name__, static_folder="app/static", template_folder="app")
CORS(app)

# Load predictor once at startup
predictor = FakeNewsPredictor(models_dir="models")

# ─── Serve the HTML frontend ────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("app", "index.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("app/static", filename)

# ─── REST Prediction Endpoint ───────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    title  = data.get("title", "").strip()
    text   = data.get("text", "").strip()
    model  = data.get("model", "Logistic Regression")

    full_text = (title + " " + text).strip()
    if len(full_text) < 15:
        return jsonify({"error": "Please enter at least 15 characters of content."}), 400

    try:
        result = predictor.predict(full_text, model_choice=model)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── Health check ───────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "models_ready": predictor.is_ready(),
        "models_dir": os.path.abspath("models")
    })

if __name__ == "__main__":
    print("\nVerascope API starting at http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000, host="0.0.0.0")
