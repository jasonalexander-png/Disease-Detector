# 🩺 Disease Detector

Aplikasi web berbasis Machine Learning yang memprediksi kemungkinan penyakit berdasarkan gejala yang dialami pengguna, lengkap dengan rekomendasi dokter spesialis, deskripsi penyakit, dan saran pencegahan.

**🔗 Live Demo:** [jsnaldr.pythonanywhere.com](https://jsnaldr.pythonanywhere.com)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.2-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/status-live-brightgreen)

---

## 📖 Tentang Project

Disease Detector adalah sistem yang menggabungkan **Data Mining** dan **Machine Learning** untuk membantu pengguna memahami kemungkinan penyakit yang dialami berdasarkan gejala yang dipilih, sekaligus memberikan arah tindak lanjut berupa rekomendasi jenis dokter spesialis yang relevan.

Project ini dibangun dengan pendekatan metodologi yang disengaja: **model Machine Learning hanya dilatih menggunakan data gejala**, sementara data profil pengguna (usia, berat badan, tinggi badan, tekanan darah, durasi sakit) ditampilkan sebagai informasi pelengkap di antarmuka tanpa memengaruhi hasil prediksi. Pendekatan ini menjaga validitas model karena tidak ada asumsi/data buatan yang dipaksakan ke dalam proses pelatihan.

> ⚠️ Hasil prediksi bersifat informatif dan edukatif, **bukan pengganti diagnosis medis profesional.**

---

## ✨ Fitur

- **Input gejala interaktif** — pencarian dan pemilihan gejala lewat checkbox dengan fitur search
- **Prediksi penyakit** menggunakan model klasifikasi Machine Learning
- **Top-3 kemungkinan penyakit** beserta persentase probabilitasnya, tidak hanya satu hasil mutlak
- **Rekomendasi dokter spesialis** sesuai hasil prediksi
- **Deskripsi penyakit** dan **saran pencegahan** otomatis ditampilkan
- **Form data pasien** (usia, berat/tinggi badan, tekanan darah, durasi sakit) sebagai pelengkap konteks

---

## 🏗️ Arsitektur & Alur Sistem

```
                    FORM WEBSITE
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
     PROFIL PASIEN                  GEJALA
  (usia, BB, TB, tekanan         (checkbox, dicari
   darah, durasi — pelengkap)     lewat search box)
           │                           │
           │                           ▼
           │                  MACHINE LEARNING MODEL
           │                    (Random Forest)
           │                           │
           │                           ▼
           │                  PREDIKSI PENYAKIT
           │                           │
           │              ┌────────────┴────────────┐
           │              ▼                         ▼
           │      Dokter Spesialis         Deskripsi & Precaution
           │              │                         │
           └──────────────┴─────────────────────────┘
                           ▼
                    HASIL DITAMPILKAN
```

---

## 🛠️ Tech Stack

| Layer | Teknologi |
|---|---|
| **Machine Learning** | scikit-learn (Random Forest, Decision Tree, Naive Bayes, Logistic Regression — dibandingkan lewat F1-macro score) |
| **Data Processing** | pandas, numpy |
| **Backend** | Flask (Python) |
| **Frontend** | HTML, CSS, JavaScript (vanilla) |
| **Model Serialization** | joblib |
| **Deployment** | PythonAnywhere |
| **Version Control** | Git & GitHub |

---

## 🔬 Metodologi Machine Learning

1. **Data Preprocessing** — one-hot encoding gejala dari format teks (`Symptom_1`, `Symptom_2`, ...) menjadi fitur biner (0/1)
2. **Exploratory Data Analysis** — analisis distribusi kelas penyakit dan pengecekan keseimbangan data
3. **Train-Test Split** — pembagian data dengan `stratify` untuk menjaga proporsi tiap kelas penyakit
4. **Model Comparison** — melatih dan membandingkan 4 algoritma klasifikasi (Decision Tree, Random Forest, Naive Bayes, Logistic Regression)
5. **Evaluation** — classification report, feature importance, dan 5-fold cross-validation untuk memastikan hasil evaluasi kredibel dan tidak overfit pada satu split data
6. **Model Selection** — dipilih berdasarkan **F1-macro score**, bukan hanya accuracy, karena distribusi kelas penyakit dalam dataset tidak seimbang

---

## 📂 Struktur Project

```
Disease-Detector/
├── data/
│   ├── raw/                        # Dataset mentah (CSV)
│   └── processed/                  # Dataset hasil cleaning & encoding
├── notebooks/
│   └── eda_modelling.ipynb         # Notebook lengkap: EDA → preprocessing → modeling → evaluasi
├── models/                         # Model & artefak hasil training (.pkl)
└── webapp/
    ├── app.py                      # Backend Flask & API prediksi
    ├── templates/
    │   └── index.html              # Halaman utama
    ├── static/
    │   ├── style.css
    │   └── script.js
    ├── models/                     # Copy artefak model untuk backend
    └── requirements.txt
```

---

## 🚀 Menjalankan Secara Lokal

```bash
# Clone repository
git clone https://github.com/jasonalexander-png/Disease-Detector.git
cd Disease-Detector/webapp

# Buat virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Jalankan
python app.py
```

Buka `http://127.0.0.1:5000` (atau `8080`, tergantung konfigurasi) di browser.

---

## 📊 Dataset

Dataset gejala-penyakit yang digunakan bersumber dari Kaggle, mencakup relasi antara gejala klinis dengan diagnosis penyakit, dilengkapi dataset pendukung untuk mapping dokter spesialis, deskripsi penyakit, dan saran pencegahan.

---

## 🔭 Pengembangan Selanjutnya

- [ ] Menyimpan riwayat prediksi pengguna ke database untuk analisis lanjutan
- [ ] Menambahkan data profil pasien (usia, BB, TB, tekanan darah) sebagai fitur tambahan model versi berikutnya, dengan dataset yang memang mendukung kombinasi variabel tersebut
- [ ] Validasi model dengan data klinis riil (bekerja sama dengan tenaga medis)
- [ ] Custom domain untuk deployment

---

## 👤 Author

**Jason Alexander**
[GitHub](https://github.com/jasonalexander-png)

---

## 📄 License

Project ini dibuat untuk keperluan portofolio.
