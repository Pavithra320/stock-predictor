from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model
with open("stock_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Stock Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        close_price = float(data["close_price"])  # Get input
        prediction = model.predict(np.array([[close_price]]))[0]  # Predict
        return jsonify({"predicted_price": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
