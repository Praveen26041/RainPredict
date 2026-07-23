from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("rainfall_model.pkl")

FEATURES = [
    "Temperature_C",
    "Humidity_Percent",
    "Wind_Speed_kmh",
    "Pressure_hPa",
    "Cloud_Cover_Percent",
    "Previous_Rainfall_mm"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_data = pd.DataFrame([{
            "Temperature_C": float(data["temperature"]),
            "Humidity_Percent": float(data["humidity"]),
            "Wind_Speed_kmh": float(data["wind_speed"]),
            "Pressure_hPa": float(data["pressure"]),
            "Cloud_Cover_Percent": float(data["cloud_cover"]),
            "Previous_Rainfall_mm": float(data["previous_rainfall"])
        }])

        prediction = int(model.predict(input_data[FEATURES])[0])
        probabilities = model.predict_proba(input_data[FEATURES])[0]
        rain_probability = round(float(probabilities[1]) * 100, 2)

        if prediction == 1:
            result = "Rain Likely"
            advice = "Delay irrigation if possible because rainfall is likely."
        else:
            result = "Rain Unlikely"
            advice = "Irrigation may be needed depending on soil moisture and crop requirements."

        return jsonify({
            "success": True,
            "prediction": result,
            "rain_probability": rain_probability,
            "irrigation_advice": advice
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
