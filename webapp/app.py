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


CATEGORY_ORDER = [
    "Kulit & Kuku",
    "Kepala & Saraf",
    "Mata",
    "THT & Pernapasan",
    "Pencernaan & Perut",
    "Jantung & Sirkulasi",
    "Otot & Sendi",
    "Saluran Kemih & Reproduksi",
    "Kondisi Umum & Lainnya",
]

# key = nama gejala ASLI di dataset (huruf kecil semua), value = (label Indonesia, kategori)
SYMPTOM_ID = {
    "itching": ("Gatal-gatal", "Kulit & Kuku"),
    "skin_rash": ("Ruam kulit", "Kulit & Kuku"),
    "nodal_skin_eruptions": ("Bintil/benjolan pada kulit", "Kulit & Kuku"),
    "continuous_sneezing": ("Bersin terus-menerus", "THT & Pernapasan"),
    "shivering": ("Menggigil", "Kondisi Umum & Lainnya"),
    "chills": ("Kedinginan/meriang", "Kondisi Umum & Lainnya"),
    "joint_pain": ("Nyeri sendi", "Otot & Sendi"),
    "stomach_pain": ("Sakit perut (lambung)", "Pencernaan & Perut"),
    "acidity": ("Asam lambung naik", "Pencernaan & Perut"),
    "ulcers_on_tongue": ("Sariawan di lidah", "Pencernaan & Perut"),
    "muscle_wasting": ("Otot mengecil/menyusut", "Otot & Sendi"),
    "vomiting": ("Muntah", "Pencernaan & Perut"),
    "burning_micturition": ("Nyeri/panas saat buang air kecil", "Saluran Kemih & Reproduksi"),
    "spotting_urination": ("Buang air kecil menetes/tidak lancar", "Saluran Kemih & Reproduksi"),
    "fatigue": ("Kelelahan", "Kondisi Umum & Lainnya"),
    "weight_gain": ("Berat badan naik", "Kondisi Umum & Lainnya"),
    "anxiety": ("Cemas berlebihan", "Kepala & Saraf"),
    "cold_hands_and_feets": ("Tangan dan kaki dingin", "Jantung & Sirkulasi"),
    "mood_swings": ("Perubahan suasana hati", "Kepala & Saraf"),
    "weight_loss": ("Berat badan turun", "Kondisi Umum & Lainnya"),
    "restlessness": ("Gelisah", "Kepala & Saraf"),
    "lethargy": ("Lesu/lemas", "Kondisi Umum & Lainnya"),
    "patches_in_throat": ("Bercak putih di tenggorokan", "THT & Pernapasan"),
    "irregular_sugar_level": ("Kadar gula darah tidak stabil", "Kondisi Umum & Lainnya"),
    "cough": ("Batuk", "THT & Pernapasan"),
    "high_fever": ("Demam tinggi", "Kondisi Umum & Lainnya"),
    "sunken_eyes": ("Mata cekung", "Mata"),
    "breathlessness": ("Sesak napas", "THT & Pernapasan"),
    "sweating": ("Berkeringat berlebihan", "Kondisi Umum & Lainnya"),
    "dehydration": ("Dehidrasi", "Kondisi Umum & Lainnya"),
    "indigestion": ("Gangguan pencernaan", "Pencernaan & Perut"),
    "headache": ("Sakit kepala", "Kepala & Saraf"),
    "yellowish_skin": ("Kulit menguning", "Kulit & Kuku"),
    "dark_urine": ("Urine berwarna gelap", "Saluran Kemih & Reproduksi"),
    "nausea": ("Mual", "Pencernaan & Perut"),
    "loss_of_appetite": ("Nafsu makan menurun", "Pencernaan & Perut"),
    "pain_behind_the_eyes": ("Nyeri di belakang mata", "Mata"),
    "back_pain": ("Sakit punggung", "Otot & Sendi"),
    "constipation": ("Sembelit/susah BAB", "Pencernaan & Perut"),
    "abdominal_pain": ("Nyeri perut", "Pencernaan & Perut"),
    "diarrhoea": ("Diare", "Pencernaan & Perut"),
    "mild_fever": ("Demam ringan", "Kondisi Umum & Lainnya"),
    "yellow_urine": ("Urine kuning pekat", "Saluran Kemih & Reproduksi"),
    "yellowing_of_eyes": ("Mata menguning", "Mata"),
    "acute_liver_failure": ("Gagal hati akut", "Pencernaan & Perut"),
    "fluid_overload": ("Kelebihan cairan tubuh", "Jantung & Sirkulasi"),
    "swelling_of_stomach": ("Perut bengkak", "Pencernaan & Perut"),
    "swelled_lymph_nodes": ("Kelenjar getah bening bengkak", "Kondisi Umum & Lainnya"),
    "malaise": ("Badan tidak enak/lemas", "Kondisi Umum & Lainnya"),
    "blurred_and_distorted_vision": ("Penglihatan kabur/buram", "Mata"),
    "phlegm": ("Berdahak", "THT & Pernapasan"),
    "throat_irritation": ("Tenggorokan gatal/iritasi", "THT & Pernapasan"),
    "redness_of_eyes": ("Mata merah", "Mata"),
    "sinus_pressure": ("Tekanan di area sinus", "THT & Pernapasan"),
    "runny_nose": ("Hidung meler", "THT & Pernapasan"),
    "congestion": ("Hidung tersumbat", "THT & Pernapasan"),
    "chest_pain": ("Nyeri dada", "Jantung & Sirkulasi"),
    "weakness_in_limbs": ("Lemas pada tangan/kaki", "Otot & Sendi"),
    "fast_heart_rate": ("Detak jantung cepat", "Jantung & Sirkulasi"),
    "pain_during_bowel_movements": ("Nyeri saat BAB", "Pencernaan & Perut"),
    "pain_in_anal_region": ("Nyeri di area anus", "Pencernaan & Perut"),
    "bloody_stool": ("BAB berdarah", "Pencernaan & Perut"),
    "irritation_in_anus": ("Iritasi di anus", "Pencernaan & Perut"),
    "neck_pain": ("Nyeri leher", "Otot & Sendi"),
    "dizziness": ("Pusing", "Kepala & Saraf"),
    "cramps": ("Kram otot", "Otot & Sendi"),
    "bruising": ("Mudah memar", "Kulit & Kuku"),
    "obesity": ("Obesitas", "Kondisi Umum & Lainnya"),
    "swollen_legs": ("Kaki bengkak", "Jantung & Sirkulasi"),
    "swollen_blood_vessels": ("Pembuluh darah membengkak", "Jantung & Sirkulasi"),
    "puffy_face_and_eyes": ("Wajah dan mata bengkak", "Mata"),
    "enlarged_thyroid": ("Kelenjar tiroid membesar", "Kondisi Umum & Lainnya"),
    "brittle_nails": ("Kuku rapuh", "Kulit & Kuku"),
    "swollen_extremeties": ("Tangan/kaki bengkak", "Jantung & Sirkulasi"),
    "excessive_hunger": ("Rasa lapar berlebihan", "Pencernaan & Perut"),
    "extra_marital_contacts": ("Riwayat hubungan di luar nikah", "Saluran Kemih & Reproduksi"),
    "drying_and_tingling_lips": ("Bibir kering dan kesemutan", "Kulit & Kuku"),
    "slurred_speech": ("Bicara tidak jelas (pelo)", "Kepala & Saraf"),
    "knee_pain": ("Nyeri lutut", "Otot & Sendi"),
    "hip_joint_pain": ("Nyeri sendi panggul", "Otot & Sendi"),
    "muscle_weakness": ("Otot lemah", "Otot & Sendi"),
    "stiff_neck": ("Leher kaku", "Otot & Sendi"),
    "swelling_joints": ("Sendi bengkak", "Otot & Sendi"),
    "movement_stiffness": ("Gerakan tubuh kaku", "Otot & Sendi"),
    "spinning_movements": ("Sensasi berputar (vertigo)", "Kepala & Saraf"),
    "loss_of_balance": ("Kehilangan keseimbangan", "Kepala & Saraf"),
    "unsteadiness": ("Tubuh limbung/tidak stabil", "Kepala & Saraf"),
    "weakness_of_one_body_side": ("Lemas di satu sisi tubuh", "Kepala & Saraf"),
    "loss_of_smell": ("Kehilangan penciuman", "THT & Pernapasan"),
    "bladder_discomfort": ("Tidak nyaman pada kandung kemih", "Saluran Kemih & Reproduksi"),
    "foul_smell_of_urine": ("Urine berbau tidak sedap", "Saluran Kemih & Reproduksi"),
    "continuous_feel_of_urine": ("Terus-menerus ingin buang air kecil", "Saluran Kemih & Reproduksi"),
    "passage_of_gases": ("Sering kentut", "Pencernaan & Perut"),
    "internal_itching": ("Gatal di dalam tubuh", "Kondisi Umum & Lainnya"),
    "toxic_look_(typhos)": ("Tampak lemas parah", "Kondisi Umum & Lainnya"),
    "depression": ("Depresi", "Kepala & Saraf"),
    "irritability": ("Mudah marah/tersinggung", "Kepala & Saraf"),
    "muscle_pain": ("Nyeri otot", "Otot & Sendi"),
    "altered_sensorium": ("Kesadaran menurun/bingung", "Kepala & Saraf"),
    "red_spots_over_body": ("Bintik merah di tubuh", "Kulit & Kuku"),
    "belly_pain": ("Nyeri perut bagian bawah", "Pencernaan & Perut"),
    "abnormal_menstruation": ("Menstruasi tidak normal", "Saluran Kemih & Reproduksi"),
    "dischromic_patches": ("Bercak warna kulit tidak merata", "Kulit & Kuku"),
    "watering_from_eyes": ("Mata berair", "Mata"),
    "increased_appetite": ("Nafsu makan meningkat", "Pencernaan & Perut"),
    "polyuria": ("Sering buang air kecil (poliuria)", "Saluran Kemih & Reproduksi"),
    "family_history": ("Riwayat penyakit keluarga", "Kondisi Umum & Lainnya"),
    "mucoid_sputum": ("Dahak berlendir", "THT & Pernapasan"),
    "rusty_sputum": ("Dahak berwarna kemerahan", "THT & Pernapasan"),
    "lack_of_concentration": ("Sulit berkonsentrasi", "Kepala & Saraf"),
    "visual_disturbances": ("Gangguan penglihatan", "Mata"),
    "receiving_blood_transfusion": ("Riwayat transfusi darah", "Kondisi Umum & Lainnya"),
    "receiving_unsterile_injections": ("Riwayat suntikan tidak steril", "Kondisi Umum & Lainnya"),
    "coma": ("Koma", "Kepala & Saraf"),
    "stomach_bleeding": ("Perdarahan lambung", "Pencernaan & Perut"),
    "distention_of_abdomen": ("Perut kembung/membesar", "Pencernaan & Perut"),
    "history_of_alcohol_consumption": ("Riwayat konsumsi alkohol", "Kondisi Umum & Lainnya"),
    "blood_in_sputum": ("Dahak berdarah", "THT & Pernapasan"),
    "prominent_veins_on_calf": ("Urat menonjol di betis", "Jantung & Sirkulasi"),
    "palpitations": ("Jantung berdebar", "Jantung & Sirkulasi"),
    "painful_walking": ("Nyeri saat berjalan", "Otot & Sendi"),
    "pus_filled_pimples": ("Jerawat bernanah", "Kulit & Kuku"),
    "blackheads": ("Komedo", "Kulit & Kuku"),
    "scurring": ("Bekas luka", "Kulit & Kuku"),
    "skin_peeling": ("Kulit mengelupas", "Kulit & Kuku"),
    "silver_like_dusting": ("Sisik keperakan pada kulit", "Kulit & Kuku"),
    "small_dents_in_nails": ("Lekukan kecil pada kuku", "Kulit & Kuku"),
    "inflammatory_nails": ("Peradangan pada kuku", "Kulit & Kuku"),
    "blister": ("Lepuh/gelembung kulit", "Kulit & Kuku"),
    "red_sore_around_nose": ("Luka merah di sekitar hidung", "Kulit & Kuku"),
    "yellow_crust_ooze": ("Keluar cairan berkerak kuning", "Kulit & Kuku"),
}


def get_symptom_info(key):
    """Kembalikan (label_indonesia, kategori) untuk 1 nama gejala dari dataset.
    Kalau tidak ada di kamus (nama beda dari dugaan), fallback ke format judul biasa."""
    k = key.strip().lower()
    if k in SYMPTOM_ID:
        return SYMPTOM_ID[k]
    fallback_label = key.replace("_", " ").title()
    return fallback_label, "Kondisi Umum & Lainnya"


@app.route("/")
def index():
    grouped = {}
    for s in symptom_columns:
        label, category = get_symptom_info(s)
        grouped.setdefault(category, []).append({"value": s, "label": label})

    for cat in grouped:
        grouped[cat].sort(key=lambda x: x["label"])

    categories = [(cat, grouped[cat]) for cat in CATEGORY_ORDER if grouped.get(cat)]
    for cat in grouped:
        if cat not in CATEGORY_ORDER:
            categories.append((cat, grouped[cat]))

    return render_template(
        "index.html",
        categories=categories,
        total_symptoms=len(symptom_columns),
    )

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
