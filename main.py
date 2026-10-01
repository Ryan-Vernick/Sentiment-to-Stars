import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model once when the server starts
model = joblib.load('currentModel.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_input = data.get('input_value')
    
    prediction = model.predict([user_input])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, port=3000)