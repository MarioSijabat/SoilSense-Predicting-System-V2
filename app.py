from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load semua model dan scaler
try:
    model_clf = joblib.load("model_classifier.pkl")
    model_reg = joblib.load("model_regressor.pkl")   # Load model regresi
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
except Exception as e:
    print(f"[ERROR] Gagal memuat file: {e}")
    exit()

# Route halaman utama
@app.route('/')
def index():
     return render_template('index.html')

@app.route('/input')
def input_data():
    return render_template('input_data.html')

# Route prediksi
@app.route('/predict', methods=["POST"])
def predict():
    data = request.get_json()
    
    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    try:
        # Buat dataframe input dan scaling
        df_input = pd.DataFrame([data], columns=features)
        df_scaled = scaler.transform(df_input)

        # Prediksi klasifikasi tanaman
        prediction_class = model_clf.predict(df_scaled)[0]
        predicted_label = le.inverse_transform([prediction_class])[0]

        # Prediksi fertility_score menggunakan model regresi
        fertility_score_pred = model_reg.predict(df_scaled)[0]
        fertility_score_pred = round(fertility_score_pred, 2)

        return jsonify({
            "predicted_crop": predicted_label,
            "fertility_score": fertility_score_pred
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)