from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("car_encoded_model.pkl")

@app.route("/")
def home():
    return "Model API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json 
    X_test = np.array(data["features"]).reshape(1, -1) 

    predicted = model.predict(X_test)
    return jsonify({"prediction": predicted.tolist()})

if __name__ == "__main__":
    app.run(debug=False)
