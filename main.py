import joblib
from pathlib import Path

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from tokenization import tokenize

app = Flask(__name__)
CORS(app)

PROJECT_DIR = Path(__file__).resolve().parent
model = joblib.load(PROJECT_DIR / 'currentModel.pkl')
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    user_input = data.get('input_value')
    if not isinstance(user_input, str) or not user_input.strip():
        return jsonify({'error': 'Enter review text before generating a rating.'}), 400

    token_ids = tokenize(user_input)[:2475]
    token_ids.extend([0] * (2475 - len(token_ids)))
    model_input = torch.tensor([token_ids], dtype=torch.long)

    with torch.inference_mode():
        prediction = model(model_input).squeeze().item()
    return jsonify({'result': prediction})

if __name__ == '__main__':
    app.run(debug=True, port=5000)