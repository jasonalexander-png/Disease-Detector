# 🩺 Disease Detector

Aplikasi web berbasis Machine Learning yang memprediksi kemungkinan penyakit berdasarkan gejala yang dipilih pengguna, lengkap dengan rekomendasi dokter spesialis, deskripsi penyakit, dan saran pencegahan — seluruhnya dalam Bahasa Indonesia.

**🔗 Live Demo:** [jsnaldr.pythonanywhere.com](https://jsnaldr.pythonanywhere.com)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.2-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/status-live-brightgreen)

---

## 📖 Tentang Project

Disease Detector adalah sistem skrining gejala berbasis Machine Learning yang membantu pengguna memahami kemungkinan penyakit yang dialami, sekaligus memberi arah tindak lanjut berupa rekomendasi dokter spesialis yang relevan.

Project ini dibangun dengan keputusan metodologi yang disengaja: **model Machine Learning hanya dilatih menggunakan data gejala**, sementara data profil pengguna (usia, berat/tinggi badan, tekanan darah, durasi sakit) ditampilkan sebagai informasi pelengkap tanpa memengaruhi hasil prediksi — menjaga validitas model dari data yang tidak pernah benar-benar dipelajarinya.

> ⚠️ Hasil prediksi bersifat informatif dan edukatif, **bukan pengganti diagnosis medis profesional.**

---

## ✨ Fitur

- **131 gejala** dikelompokkan ke dalam 9 kategori tubuh (Kulit & Kuku, Kepala & Saraf, Mata, THT & Pernapasan, Pencernaan, Jantung & Sirkulasi, Otot & Sendi, Saluran Kemih, Kondisi Umum), lengkap dengan pencarian gejala
- **Antarmuka sepenuhnya Bahasa Indonesia** — seluruh nama gejala, deskripsi penyakit, saran pencegahan, dan nama spesialis diterjemahkan secara manual dari dataset asli berbahasa Inggris
- **Prediksi penyakit** dengan model klasifikasi terbaik (dipilih dari perbandingan Random Forest, Decision Tree, Naive Bayes, dan Logistic Regression berdasarkan F1-macro score)
- **Top-3 kemungkinan penyakit** beserta persentase probabilitas, bukan hanya satu jawaban mutlak
- **Indikator kepercayaan hasil** — sistem memberi peringatan ketika probabilitas prediksi rendah (gejala terlalu umum untuk dibedakan antar penyakit), disertai saran untuk menambah gejala yang lebih spesifik
- **Validasi input minimal** — mewajibkan minimal 3 gejala dipilih agar sinyal prediksi tidak terlalu tipis
- **Rekomendasi dokter spesialis**, deskripsi penyakit, dan saran pencegahan yang disesuaikan dengan hasil prediksi
- Form data pasien (usia, berat/tinggi badan, tekanan darah, durasi sakit) sebagai pelengkap konteks, menggunakan pilihan kategori (bukan input bebas) untuk kemudahan pengguna awam

---

## 🏗️ Arsitektur & Alur Sistem

```
                    FORM WEBSITE
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
     PROFIL PASIEN                  GEJALA
  (usia, BB, TB, tekanan       (checkbox per kategori
   darah, durasi — pelengkap)      + pencarian)
           │                           │
           │                           ▼
           │                  MACHINE LEARNING MODEL
           │                    (klasifikasi terbaik)
           │                           │
           │                           ▼
           │              CEK CONFIDENCE PREDIKSI
           │                    │           │
           │              rendah│           │tinggi
           │                    ▼           ▼
           │              tampilkan    PREDIKSI PENYAKIT
           │               warning           │
           │                    └─────┬──────┘
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
| **Machine Learning** | scikit-learn (Random Forest, Decision Tree, Naive Bayes, Logistic Regression) |
| **Data Processing** | pandas, numpy |
| **Backend** | Flask (Python) |
| **Frontend** | HTML, CSS, JavaScript (vanilla) |
| **Model Serialization** | joblib |
| **Deployment** | PythonAnywhere |
| **Version Control** | Git & GitHub |

---

## 🔬 Metodologi Machine Learning

1. **Data Preprocessing** — one-hot encoding gejala dari format teks (`Symptom_1`, `Symptom_2`, ...) menjadi fitur biner (0/1), termasuk pengecekan dan penanganan file dataset yang tumpang tindih/duplikat antar sumber
2. **Exploratory Data Analysis** — analisis distribusi kelas penyakit dan pengecekan keseimbangan data
3. **Train-Test Split** — pembagian data dengan `stratify`, serta penanganan kelas dengan jumlah sampel terlalu sedikit
4. **Model Comparison** — melatih dan membandingkan 4 algoritma klasifikasi
5. **Evaluation** — classification report, feature importance, dan 5-fold cross-validation
6. **Model Selection** — dipilih berdasarkan **F1-macro score**, bukan hanya accuracy, karena distribusi kelas penyakit dalam dataset tidak seimbang
7. **Post-processing** — perhitungan confidence score saat inferensi untuk mendeteksi prediksi yang kurang bisa diandalkan

---

## ⚠️ Keterbatasan

- Model hanya mencakup **41 jenis penyakit** dan **131 gejala** sesuai dataset yang digunakan — penyakit di luar cakupan ini (misalnya COVID-19, batu ginjal, kanker) tidak dapat diprediksi
- Beberapa gejala bersifat sangat umum dan dipakai di banyak penyakit sekaligus (misal mual, muntah, lelah), sehingga kombinasi gejala minim dapat menghasilkan prediksi dengan tingkat kepercayaan rendah — inilah alasan fitur validasi minimal gejala dan peringatan confidence ditambahkan
- Dataset bersifat edukasi/tidak sepenuhnya merepresentasikan data klinis dunia nyata dalam skala besar

---

## 💡 Challenges & Learnings

Beberapa tantangan nyata yang dihadapi selama pengembangan, dan bagaimana diselesaikan:

- **Struktur dataset tidak konsisten** — dua file yang namanya mengindikasikan fungsi berbeda (`Doctor_Specialist.csv` dan `Doctor_Versus_Disease.csv`) ternyata isinya tertukar dari dugaan awal; salah satunya bahkan tidak memiliki header row. Diselesaikan dengan validasi isi file secara langsung sebelum diasumsikan, bukan hanya mengandalkan nama file.
- **Kendala deployment di berbagai platform gratis** — sempat mencoba Render, PythonAnywhere, Fly.io, Hugging Face Spaces, dan Replit, masing-masing dengan batasan berbeda (kuota disk, kebutuhan kartu kredit, server yang tidak selalu aktif). Akhirnya dipilih PythonAnywhere karena tidak memerlukan kartu kredit dan mendukung server yang aktif 24/7.
- **Inkonsistensi versi library antar environment** — perbedaan versi `scikit-learn` antara environment training dan environment deployment sempat menyebabkan model gagal dimuat (`incompatible dtype`). Diselesaikan dengan menyamakan versi secara eksplisit di kedua sisi.
- **Menerjemahkan konten medis teknis** — menerjemahkan 131 nama gejala dan 41 deskripsi penyakit ke Bahasa Indonesia yang natural, sambil tetap mempertahankan akurasi istilah medis.

---

## 📂 Struktur Project

```
Disease-Detector/
├── data/
│   ├── raw/                        # Dataset mentah (CSV)
│   └── processed/                  # Dataset hasil cleaning & encoding
├── notebooks/
│   └── eda_modelling.ipynb         # EDA → preprocessing → modeling → evaluasi
├── models/                         # Artefak model hasil training (.pkl)
└── webapp/
    ├── app.py                      # Backend Flask, API prediksi, kamus terjemahan
    ├── templates/
    │   └── index.html
    ├── static/
    │   ├── style.css
    │   └── script.js
    ├── models/                     # Copy artefak model untuk backend
    └── requirements.txt
```

---

## 🚀 Menjalankan Secara Lokal

```bash
git clone https://github.com/jasonalexander-png/Disease-Detector.git
cd Disease-Detector/webapp

python -m venv venv
venv\Scripts\activate        # Windows

pip install -r requirements.txt
python app.py
```

Buka `http://127.0.0.1:8080` di browser.

---

## 🔭 Pengembangan Selanjutnya

- [ ] Menambahkan penyimpanan riwayat prediksi pengguna untuk analisis lanjutan
- [ ] Memperluas cakupan dataset (jumlah penyakit & gejala) untuk representasi yang lebih luas
- [ ] Otomasi deployment dengan CI/CD (GitHub Actions) agar update kode langsung ter-deploy tanpa langkah manual
- [ ] Validasi model dengan data klinis riil bersama tenaga medis
- [ ] Custom domain untuk deployment

---

## 👤 Author

**Jason Alexander**
[GitHub](https://github.com/jasonalexander-png)

---

## 📄 License

Project ini dibuat untuk keperluan pembelajaran (Data Mining & Machine Learning).
