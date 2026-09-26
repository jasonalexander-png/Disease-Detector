# 🩺 Disease Detector

A Machine Learning-powered web application that predicts possible diseases based on user-selected symptoms, complete with medical specialist recommendations, disease descriptions, and prevention tips — with a fully localized Indonesian interface.

**🔗 Live Demo:** [jsnaldr.pythonanywhere.com](https://jsnaldr.pythonanywhere.com)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.2-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/status-live-brightgreen)

---

## 📖 About the Project

Disease Detector is a Machine Learning-based symptom screening system that helps users understand possible diseases based on their symptoms, while also guiding them toward the appropriate type of medical specialist.

The project follows a deliberate methodological decision: **the Machine Learning model is trained exclusively on symptom data**, while patient profile information (age, weight, height, blood pressure, symptom duration) is displayed purely as supplementary context without influencing the prediction — preserving model validity by avoiding data it was never actually trained on.

> ⚠️ Prediction results are informational and educational only, **not a substitute for professional medical diagnosis.**

---

## ✨ Features

- **131 symptoms** organized into 9 body-based categories (Skin & Nails, Head & Nervous System, Eyes, ENT & Respiratory, Digestive, Cardiovascular, Muscles & Joints, Urinary System, General Conditions), with a built-in search function
- **Fully localized Indonesian interface** — all symptom names, disease descriptions, prevention tips, and specialist titles were manually translated from the original English-language dataset
- **Disease prediction** using the best-performing classification model, selected by comparing Random Forest, Decision Tree, Naive Bayes, and Logistic Regression based on F1-macro score
- **Top-3 probable diseases** with their respective probability percentages, rather than a single definitive answer
- **Prediction confidence indicator** — the system flags low-confidence predictions (when selected symptoms are too generic to distinguish between diseases) and suggests adding more specific symptoms
- **Minimum input validation** — requires at least 3 selected symptoms to avoid predictions based on insufficient signal
- **Specialist recommendations**, disease descriptions, and prevention tips tailored to the predicted result
- Patient profile form (age, weight, height, blood pressure, symptom duration) using categorical pill-selection inputs instead of free text, designed for ease of use by non-medical users

---

## 🏗️ System Architecture

```
                    WEB FORM
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
     PATIENT PROFILE                SYMPTOMS
  (age, weight, height,        (checkboxes grouped by
   BP, duration — context)      category + search)
           │                           │
           │                           ▼
           │                  MACHINE LEARNING MODEL
           │                   (best-performing model)
           │                           │
           │                           ▼
           │              PREDICTION CONFIDENCE CHECK
           │                    │           │
           │                 low│           │high
           │                    ▼           ▼
           │               show warning  DISEASE PREDICTION
           │                    └─────┬──────┘
           │              ┌────────────┴────────────┐
           │              ▼                         ▼
           │       Medical Specialist      Description & Precautions
           │              │                         │
           └──────────────┴─────────────────────────┘
                           ▼
                    RESULT DISPLAYED
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Machine Learning** | scikit-learn (Random Forest, Decision Tree, Naive Bayes, Logistic Regression) |
| **Data Processing** | pandas, numpy |
| **Backend** | Flask (Python) |
| **Frontend** | HTML, CSS, JavaScript (vanilla) |
| **Model Serialization** | joblib |
| **Deployment** | PythonAnywhere |
| **Version Control** | Git & GitHub |

---

## 🔬 Machine Learning Methodology

1. **Data Preprocessing** — one-hot encoding symptoms from text format (`Symptom_1`, `Symptom_2`, ...) into binary features (0/1), including validation and handling of overlapping/duplicate files across data sources
2. **Exploratory Data Analysis** — analyzing disease class distribution and checking for class imbalance
3. **Train-Test Split** — stratified splitting, along with handling of classes with too few samples
4. **Model Comparison** — training and comparing 4 classification algorithms
5. **Evaluation** — classification report, feature importance analysis, and 5-fold cross-validation
6. **Model Selection** — chosen based on **F1-macro score** rather than accuracy alone, due to class imbalance in the disease distribution
7. **Post-processing** — confidence score computation at inference time to flag unreliable predictions

---

## ⚠️ Limitations

- The model only covers **41 disease types** and **131 symptoms** based on the dataset used — diseases outside this scope (e.g., COVID-19, kidney stones, cancer) cannot be predicted
- Some symptoms are highly generic and shared across many diseases (e.g., nausea, vomiting, fatigue), meaning a minimal symptom combination can result in low-confidence predictions — this is why the minimum symptom requirement and confidence warning features were added
- The dataset is educational in nature and does not fully represent large-scale, real-world clinical data

---

## 💡 Challenges & Learnings

Real challenges encountered during development, and how they were resolved:

- **Inconsistent dataset structure** — two files whose names implied different purposes (`Doctor_Specialist.csv` and `Doctor_Versus_Disease.csv`) turned out to have swapped roles from initial assumptions; one of them didn't even have a header row. Resolved by validating actual file contents rather than relying on filenames alone.
- **Deployment challenges across free-tier platforms** — Render, PythonAnywhere, Fly.io, Hugging Face Spaces, and Replit were all evaluated, each with different constraints (disk quotas, credit card requirements, servers that sleep when idle). PythonAnywhere was ultimately chosen for requiring no credit card while still providing a 24/7 always-on server.
- **Library version inconsistency across environments** — a scikit-learn version mismatch between the training and deployment environments caused model loading to fail (`incompatible dtype`). Resolved by explicitly pinning matching versions on both sides.
- **Translating technical medical content** — translated 131 symptom names and 41 disease descriptions into natural Bahasa Indonesia while preserving medical accuracy.

---

## 📂 Project Structure

```
Disease-Detector/
├── data/
│   ├── raw/                        # Raw dataset (CSV)
│   └── processed/                  # Cleaned & encoded dataset
├── notebooks/
│   └── eda_modelling.ipynb         # EDA → preprocessing → modeling → evaluation
├── models/                         # Trained model artifacts (.pkl)
└── webapp/
    ├── app.py                      # Flask backend, prediction API, translation dictionaries
    ├── templates/
    │   └── index.html
    ├── static/
    │   ├── style.css
    │   └── script.js
    ├── models/                     # Model artifacts copy for the backend
    └── requirements.txt
```

---

## 🚀 Running Locally

```bash
git clone https://github.com/jasonalexander-png/Disease-Detector.git
cd Disease-Detector/webapp

python -m venv venv
venv\Scripts\activate        # Windows

pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:8080` in your browser.

---

## 🔭 Future Improvements

- [ ] Store user prediction history for further analysis
- [ ] Expand dataset coverage (more diseases & symptoms) for broader representation
- [ ] Automate deployment with CI/CD (GitHub Actions) for zero-touch updates on every push
- [ ] Validate the model against real clinical data with medical professionals
- [ ] Custom domain for deployment

---

## 👤 Author

**Jason Alexander**
[GitHub](https://github.com/jasonalexander-png)

---

## 📄 License

This project was built for educational purposes (Data Mining & Machine Learning).
