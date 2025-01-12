import pickle
import numpy as np
from flask import Flask, request, jsonify
from urllib.parse import quote as url_quote  # Updated import for URL quoting

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = "model.pkl"
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

@app.route("/")
def index():
    return jsonify({"message": "Model API is running"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict endpoint for the model
    Expects JSON input: {"features": [value1, value2, ...]}
    """
    try:
        # Get the input data
        data = request.get_json()
        features = data["features"]

        # Convert input to numpy array and make a prediction
        prediction = model.predict(np.array(features).reshape(1, -1))

        # Return prediction as JSON
        return jsonify({"prediction": prediction.tolist()}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
