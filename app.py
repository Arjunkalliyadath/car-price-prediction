from flask import Flask, render_template, request
import pickle
import numpy as np
import yaml

app = Flask(__name__)

# Load config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load model and scaler using config
model = pickle.load(open(config["model_path"], "rb"))
scaler = pickle.load(open(config["scaler_path"], "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get values from form
    present_price = float(request.form["present_price"])
    kms_driven = float(request.form["kms_driven"])
    owner = int(request.form["owner"])
    car_age = int(request.form["car_age"])
    fuel_type = request.form["fuel_type"]
    seller_type = request.form["seller_type"]
    transmission = request.form["transmission"]

    # Encoding manually (must match training)
    fuel_diesel = 1 if fuel_type == "Diesel" else 0
    fuel_petrol = 1 if fuel_type == "Petrol" else 0
    seller_individual = 1 if seller_type == "Individual" else 0
    transmission_manual = 1 if transmission == "Manual" else 0

    input_data = np.array([[present_price,
                             kms_driven,
                             owner,
                             car_age,
                             fuel_diesel,
                             fuel_petrol,
                             seller_individual,
                             transmission_manual]])

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    return render_template(
        "index.html",
        prediction_text=f"Estimated Car Price: ₹ {round(prediction, 2)} Lakhs"
    )

if __name__ == "__main__":
    app.run()

