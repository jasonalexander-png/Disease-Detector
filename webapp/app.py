from flask import Flask, render_template, jsonify, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# ---- Load semua artefak sekali saat server start ----
model = joblib.load(os.path.join(MODEL_DIR, "disease_model.pkl"))
symptom_columns = joblib.load(os.path.join(MODEL_DIR, "symptom_columns.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
lookup = pd.read_pickle(os.path.join(MODEL_DIR, "disease_lookup.pkl"))


def format_symptom_label(s):
    """'skin_rash' -> 'Skin Rash' supaya enak dibaca di form"""
    return s.replace("_", " ").title()


@app.route("/")
def index():
    symptoms = sorted(symptom_columns)
    symptom_options = [{"value": s, "label": format_symptom_label(s)} for s in symptoms]
    return render_template("index.html", symptoms=symptom_options)


@app.route("/api/symptoms")
def get_symptoms():
    """Endpoint terpisah kalau nanti frontend mau di-fetch via JS murni, bukan render Jinja"""
    symptoms = sorted(symptom_columns)
    return jsonify([{"value": s, "label": format_symptom_label(s)} for s in symptoms])


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    gejala_user = data.get("symptoms", [])

    if not gejala_user:
        return jsonify({"error": "Pilih minimal 1 gejala"}), 400

    # validasi: hanya terima gejala yang memang dikenal model
    gejala_user = [g for g in gejala_user if g in symptom_columns]
    if not gejala_user:
        return jsonify({"error": "Gejala yang dikirim tidak dikenali sistem"}), 400

    input_vector = pd.DataFrame(
        [[1 if col in gejala_user else 0 for col in symptom_columns]],
        columns=symptom_columns
    )

    pred_encoded = model.predict(input_vector)[0]
    pred_disease = label_encoder.inverse_transform([pred_encoded])[0]

    proba = model.predict_proba(input_vector)[0]
    top3_idx = np.argsort(proba)[-3:][::-1]
    top3 = [
        {
            "penyakit": label_encoder.inverse_transform([i])[0],
            "probabilitas": round(float(proba[i]) * 100, 2)
        }
        for i in top3_idx
    ]

    info_rows = lookup[lookup['disease_clean'] == pred_disease.strip().lower()]
    info_records = info_rows.drop(columns=['disease_clean'], errors='ignore').to_dict('records')
    info = info_records[0] if info_records else {}

    # NaN tidak valid di JSON, ganti jadi None
    info_clean = {k: (v if pd.notna(v) else None) for k, v in info.items()}

    profil = data.get("profil", {})  # umur/BB/TB/tekanan darah/durasi -> hanya diteruskan balik, TIDAK dipakai model

    return jsonify({
        "prediksi_utama": pred_disease,
        "top3_kemungkinan": top3,
        "info_tambahan": info_clean,
        "profil_user": profil
    })


if __name__ == "__main__":
    app.run(debug=True)
