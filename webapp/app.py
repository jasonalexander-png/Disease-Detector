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


SPECIALIST_ID = {
    "allergist": "Ahli Alergi",
    "cardiologist": "Ahli Jantung (Kardiolog)",
    "dermatologist": "Dokter Kulit (Dermatolog)",
    "endocrinologist": "Ahli Hormon (Endokrinolog)",
    "gastroenterologist": "Ahli Pencernaan (Gastroenterolog)",
    "general physician": "Dokter Umum",
    "gynecologist": "Dokter Kandungan",
    "hepatologist": "Ahli Hati (Hepatolog)",
    "nephrologist": "Ahli Ginjal (Nefrolog)",
    "neurologist": "Ahli Saraf (Neurolog)",
    "osteopathic": "Ahli Osteopati",
    "otolaryngologist": "Dokter THT",
    "pediatrician": "Dokter Anak",
    "phlebologist": "Ahli Pembuluh Darah (Flebolog)",
    "psychiatrist": "Psikiater",
    "pulmonologist": "Ahli Paru (Pulmonolog)",
    "rheumatologist": "Ahli Reumatik (Reumatolog)",
    "tropical medicine specialist": "Spesialis Penyakit Tropis",
    "urologist": "Ahli Kemih (Urolog)",
}


def translate_specialist(name):
    if not name:
        return name
    key = str(name).strip().lower()
    return SPECIALIST_ID.get(key, name)


# key = nama penyakit (huruf kecil, di-strip), value = deskripsi & saran pencegahan dalam Bahasa Indonesia
DISEASE_ID = {
    "drug reaction": {
        "description": "Reaksi obat yang merugikan (ADR) adalah cedera yang disebabkan oleh konsumsi obat. ADR dapat terjadi setelah satu kali dosis atau penggunaan obat dalam jangka panjang, atau akibat kombinasi dua obat atau lebih.",
        "precautions": ["Hentikan pemicu iritasi", "Konsultasi ke rumah sakit terdekat", "Hentikan konsumsi obat pemicu", "Lakukan kontrol lanjutan ke dokter"],
    },
    "malaria": {
        "description": "Penyakit menular yang disebabkan oleh parasit protozoa dari keluarga Plasmodium, dapat ditularkan melalui gigitan nyamuk Anopheles atau melalui jarum suntik/transfusi yang terkontaminasi. Malaria falciparum adalah jenis yang paling mematikan.",
        "precautions": ["Konsultasi ke rumah sakit terdekat", "Hindari makanan berminyak", "Hindari makanan daging/non-vegetarian", "Jauhkan diri dari gigitan nyamuk"],
    },
    "allergy": {
        "description": "Alergi adalah respons sistem kekebalan tubuh terhadap zat asing yang sebenarnya tidak berbahaya, seperti makanan tertentu, serbuk sari, atau bulu hewan peliharaan. Tugas sistem imun adalah menjaga tubuh tetap sehat dengan melawan patogen berbahaya.",
        "precautions": ["Oleskan losion calamine", "Tutup area yang terkena dengan perban", "Kompres dengan es untuk meredakan gatal"],
    },
    "hypothyroidism": {
        "description": "Hipotiroidisme, juga disebut tiroid kurang aktif, adalah gangguan pada sistem endokrin di mana kelenjar tiroid tidak menghasilkan hormon tiroid yang cukup.",
        "precautions": ["Kurangi stres", "Olahraga teratur", "Konsumsi makanan sehat", "Tidur yang cukup"],
    },
    "psoriasis": {
        "description": "Psoriasis adalah gangguan kulit umum yang menyebabkan bercak tebal, merah, dan menonjol yang tertutup sisik keperakan. Bisa muncul di mana saja, tapi paling sering di kulit kepala, siku, lutut, dan punggung bawah. Tidak menular antar orang, meski terkadang muncul pada anggota keluarga yang sama.",
        "precautions": ["Cuci tangan dengan air hangat dan sabun", "Hentikan perdarahan dengan menekan area luka", "Konsultasi ke dokter", "Mandi air garam"],
    },
    "gerd": {
        "description": "Penyakit refluks gastroesofagus (GERD) adalah gangguan pencernaan yang memengaruhi sfingter esofagus bagian bawah (LES), cincin otot antara kerongkongan dan lambung. Banyak orang, termasuk ibu hamil, mengalami sensasi panas di dada atau gangguan asam lambung akibat GERD.",
        "precautions": ["Hindari makanan berlemak dan pedas", "Jangan berbaring setelah makan", "Jaga berat badan tetap ideal", "Olahraga teratur"],
    },
    "chronic cholestasis": {
        "description": "Penyakit kolestasis kronis, baik pada bayi, anak-anak, maupun dewasa, ditandai dengan gangguan transportasi asam empedu dari hati ke usus, sebagian besar disebabkan oleh kerusakan awal pada epitel saluran empedu.",
        "precautions": ["Mandi air dingin", "Gunakan obat anti-gatal", "Konsultasi ke dokter", "Konsumsi makanan sehat"],
    },
    "hepatitis a": {
        "description": "Hepatitis A adalah infeksi hati yang sangat menular, disebabkan oleh virus hepatitis A. Virus ini menyebabkan peradangan dan memengaruhi fungsi hati.",
        "precautions": ["Konsultasi ke rumah sakit terdekat", "Cuci tangan secara menyeluruh", "Hindari makanan berlemak dan pedas", "Konsumsi obat sesuai anjuran"],
    },
    "osteoarthristis": {
        "description": "Osteoartritis adalah bentuk artritis yang paling umum, memengaruhi jutaan orang di seluruh dunia. Terjadi ketika tulang rawan pelindung yang melapisi ujung tulang aus seiring waktu.",
        "precautions": ["Konsumsi acetaminophen (paracetamol)", "Konsultasi ke rumah sakit terdekat", "Lakukan kontrol lanjutan ke dokter", "Mandi air garam"],
    },
    "(vertigo) paroymsal  positional vertigo": {
        "description": "Vertigo posisi paroksismal jinak (BPPV) adalah salah satu penyebab vertigo yang paling umum — sensasi tiba-tiba seperti berputar atau seolah bagian dalam kepala berputar. Menyebabkan episode pusing singkat mulai dari ringan hingga berat.",
        "precautions": ["Berbaring", "Hindari perubahan posisi tubuh secara tiba-tiba", "Hindari gerakan kepala yang mendadak", "Beristirahat dan rileks"],
    },
    "hypoglycemia": {
        "description": "Hipoglikemia adalah kondisi ketika kadar gula darah (glukosa) lebih rendah dari normal. Sering berkaitan dengan pengobatan diabetes, namun obat lain dan berbagai kondisi lain juga bisa menyebabkan gula darah rendah pada orang tanpa diabetes.",
        "precautions": ["Berbaring miring ke samping", "Periksa denyut nadi", "Minum minuman manis", "Konsultasi ke dokter"],
    },
    "acne": {
        "description": "Jerawat (acne vulgaris) adalah pembentukan komedo, papula, pustula, nodul, dan/atau kista akibat penyumbatan dan peradangan pada folikel rambut beserta kelenjar minyaknya. Berkembang di wajah dan tubuh bagian atas, paling sering pada remaja.",
        "precautions": ["Mandi dua kali sehari", "Hindari makanan berlemak dan pedas", "Perbanyak minum air putih", "Hindari pemakaian terlalu banyak produk perawatan kulit"],
    },
    "diabetes": {
        "description": "Diabetes terjadi ketika kadar glukosa darah terlalu tinggi. Glukosa darah adalah sumber energi utama yang berasal dari makanan. Insulin, hormon dari pankreas, membantu glukosa masuk ke sel-sel tubuh untuk digunakan sebagai energi.",
        "precautions": ["Terapkan pola makan seimbang", "Olahraga teratur", "Konsultasi ke dokter", "Lakukan kontrol lanjutan ke dokter"],
    },
    "impetigo": {
        "description": "Impetigo adalah infeksi kulit yang umum dan sangat menular, terutama menyerang bayi dan anak-anak. Biasanya muncul sebagai luka kemerahan di wajah (terutama sekitar hidung dan mulut), tangan, dan kaki, yang bisa pecah dan membentuk kerak kekuningan.",
        "precautions": ["Rendam area yang terkena dengan air hangat", "Gunakan antibiotik sesuai resep", "Angkat keropeng dengan kain lembap", "Konsultasi ke dokter"],
    },
    "hypertension": {
        "description": "Hipertensi (tekanan darah tinggi) adalah kondisi medis jangka panjang di mana tekanan darah dalam arteri terus-menerus meningkat. Umumnya tidak menimbulkan gejala.",
        "precautions": ["Lakukan meditasi", "Mandi air garam", "Kurangi stres", "Tidur yang cukup"],
    },
    "peptic ulcer diseae": {
        "description": "Penyakit tukak peptik (PUD) adalah luka pada lapisan dalam lambung, bagian awal usus halus, atau kerongkongan bagian bawah. Luka pada lambung disebut tukak lambung, luka pada usus disebut tukak duodenum.",
        "precautions": ["Hindari makanan berlemak dan pedas", "Konsumsi makanan probiotik", "Hindari konsumsi susu", "Batasi konsumsi alkohol"],
    },
    "dimorphic hemorrhoids(piles)": {
        "description": "Wasir (hemoroid) adalah struktur pembuluh darah pada saluran anus. Juga dikenal dengan sebutan ambeien.",
        "precautions": ["Hindari makanan berlemak dan pedas", "Gunakan witch hazel", "Mandi air hangat dengan garam epsom", "Konsumsi jus lidah buaya"],
    },
    "dimorphic hemmorhoids(piles)": {
        "description": "Wasir (hemoroid) adalah struktur pembuluh darah pada saluran anus. Juga dikenal dengan sebutan ambeien.",
        "precautions": ["Hindari makanan berlemak dan pedas", "Gunakan witch hazel", "Mandi air hangat dengan garam epsom", "Konsumsi jus lidah buaya"],
    },
    "common cold": {
        "description": "Flu biasa adalah infeksi virus pada hidung dan tenggorokan (saluran pernapasan atas). Biasanya tidak berbahaya meskipun terasa tidak nyaman. Berbagai jenis virus dapat menyebabkannya.",
        "precautions": ["Minum minuman kaya vitamin C", "Lakukan terapi uap (inhalasi)", "Hindari makanan/minuman dingin", "Pantau dan kendalikan suhu tubuh"],
    },
    "chicken pox": {
        "description": "Cacar air adalah penyakit sangat menular yang disebabkan oleh virus varicella-zoster (VZV), menyebabkan ruam gatal berbentuk lepuhan. Ruam pertama muncul di dada, punggung, dan wajah, lalu menyebar ke seluruh tubuh.",
        "precautions": ["Gunakan daun mimba saat mandi", "Konsumsi daun mimba", "Lakukan vaksinasi", "Hindari tempat umum/keramaian"],
    },
    "cervical spondylosis": {
        "description": "Spondilosis servikal adalah istilah umum untuk keausan terkait usia pada bantalan tulang belakang di leher, disertai tanda-tanda osteoartritis termasuk taji tulang.",
        "precautions": ["Gunakan kompres hangat atau dingin", "Olahraga teratur", "Konsumsi obat pereda nyeri bebas (OTC)", "Konsultasi ke dokter"],
    },
    "hyperthyroidism": {
        "description": "Hipertiroidisme (tiroid terlalu aktif) terjadi ketika kelenjar tiroid memproduksi hormon tiroksin berlebihan, dapat mempercepat metabolisme tubuh, menyebabkan penurunan berat badan tak disengaja serta detak jantung cepat/tidak teratur.",
        "precautions": ["Konsumsi makanan sehat", "Lakukan pijat", "Gunakan lemon balm", "Jalani terapi yodium radioaktif"],
    },
    "urinary tract infection": {
        "description": "Infeksi saluran kemih adalah infeksi pada ginjal, ureter, kandung kemih, atau uretra. Gejala umum meliputi sering ingin buang air kecil serta nyeri/panas saat buang air kecil.",
        "precautions": ["Perbanyak minum air putih", "Tingkatkan asupan vitamin C", "Minum jus cranberry", "Konsumsi probiotik"],
    },
    "varicose veins": {
        "description": "Vena varises adalah pembuluh darah yang membesar dan berkelok, sering terlihat sebagai pembuluh darah biru menonjol di bawah kulit. Paling umum pada orang dewasa lanjut usia, terutama wanita, sering di kaki.",
        "precautions": ["Berbaring dan angkat kaki lebih tinggi", "Gunakan salep", "Gunakan stoking kompresi vena", "Jangan berdiri diam terlalu lama"],
    },
    "aids": {
        "description": "AIDS adalah kondisi kronis yang berpotensi mengancam jiwa, disebabkan oleh virus HIV. HIV merusak sistem kekebalan tubuh sehingga mengganggu kemampuan tubuh melawan infeksi dan penyakit.",
        "precautions": ["Hindari luka terbuka", "Gunakan alat pelindung diri (APD) bila memungkinkan", "Konsultasi ke dokter", "Lakukan kontrol lanjutan ke dokter"],
    },
    "paralysis (brain hemorrhage)": {
        "description": "Perdarahan intraserebral (ICH) terjadi ketika darah tiba-tiba pecah ke jaringan otak. Gejala meliputi sakit kepala, kelemahan, kebingungan, dan kelumpuhan, terutama pada satu sisi tubuh.",
        "precautions": ["Lakukan pijat", "Konsumsi makanan sehat", "Olahraga teratur", "Konsultasi ke dokter"],
    },
    "typhoid": {
        "description": "Penyakit akut yang ditandai demam akibat infeksi bakteri Salmonella typhi, dengan gejala demam, sakit kepala, sembelit, badan tidak enak, kedinginan, dan nyeri otot. Diare jarang terjadi, muntah biasanya tidak parah.",
        "precautions": ["Konsumsi sayuran tinggi kalori", "Jalani terapi antibiotik", "Konsultasi ke dokter", "Konsumsi obat sesuai anjuran"],
    },
    "hepatitis b": {
        "description": "Hepatitis B adalah infeksi hati yang dapat menyebabkan jaringan parut, gagal hati, dan kanker. Menular melalui kontak dengan darah, luka terbuka, atau cairan tubuh penderita.",
        "precautions": ["Konsultasi ke rumah sakit terdekat", "Lakukan vaksinasi", "Konsumsi makanan sehat", "Konsumsi obat sesuai anjuran"],
    },
    "fungal infection": {
        "description": "Infeksi jamur terjadi ketika jamur menyerang suatu area tubuh dan sistem kekebalan tidak mampu mengatasinya. Jamur dapat hidup di udara, tanah, air, tumbuhan, dan bahkan secara alami di tubuh manusia.",
        "precautions": ["Mandi dua kali sehari", "Gunakan cairan antiseptik/daun mimba saat mandi", "Jaga area yang terinfeksi tetap kering", "Gunakan pakaian yang bersih"],
    },
    "hepatitis c": {
        "description": "Peradangan hati akibat virus hepatitis C (HCV), umumnya menyebar melalui transfusi darah, hemodialisis, dan tusukan jarum. Dapat menyebabkan sirosis dan kanker hati.",
        "precautions": ["Konsultasi ke rumah sakit terdekat", "Lakukan vaksinasi", "Konsumsi makanan sehat", "Konsumsi obat sesuai anjuran"],
    },
    "migraine": {
        "description": "Migrain menyebabkan nyeri berdenyut yang parah, biasanya pada satu sisi kepala, sering disertai mual, muntah, dan sensitivitas ekstrem terhadap cahaya dan suara. Serangan bisa berlangsung berjam-jam hingga berhari-hari.",
        "precautions": ["Lakukan meditasi", "Kurangi stres", "Gunakan kacamata hitam saat di bawah sinar matahari", "Konsultasi ke dokter"],
    },
    "bronchial asthma": {
        "description": "Asma bronkial menyebabkan saluran udara paru-paru membengkak dan menyempit, menghasilkan lendir berlebih sehingga sulit bernapas — menyebabkan batuk, napas pendek, dan mengi. Bersifat kronis dan mengganggu aktivitas harian.",
        "precautions": ["Kenakan pakaian yang longgar", "Lakukan napas dalam", "Jauhi pemicu (alergen/asap/debu)", "Segera cari bantuan medis"],
    },
    "alcoholic hepatitis": {
        "description": "Hepatitis alkoholik adalah peradangan hati akibat konsumsi alkohol berat dalam jangka panjang, diperparah oleh kebiasaan minum berlebihan. Konsumsi alkohol harus dihentikan jika mengalami kondisi ini.",
        "precautions": ["Hentikan konsumsi alkohol", "Konsultasi ke dokter", "Konsumsi obat sesuai anjuran", "Lakukan kontrol lanjutan ke dokter"],
    },
    "jaundice": {
        "description": "Perubahan warna kuning pada kulit dan bagian putih mata akibat kadar bilirubin (pigmen empedu) yang tinggi dalam darah, dapat menyebar ke jaringan dan cairan tubuh lainnya.",
        "precautions": ["Perbanyak minum air putih", "Konsumsi milk thistle", "Konsumsi buah dan makanan tinggi serat", "Konsumsi obat sesuai anjuran"],
    },
    "hepatitis e": {
        "description": "Bentuk peradangan hati yang jarang terjadi, disebabkan oleh virus hepatitis E (HEV), menular melalui makanan/minuman atau air yang terkontaminasi. Tidak menyebabkan penyakit hati kronis.",
        "precautions": ["Hentikan konsumsi alkohol", "Istirahat yang cukup", "Konsultasi ke dokter", "Konsumsi obat sesuai anjuran"],
    },
    "dengue": {
        "description": "Penyakit menular akut disebabkan oleh virus dengue, ditularkan nyamuk Aedes, ditandai sakit kepala, nyeri sendi parah, dan ruam kulit. Dikenal juga sebagai demam berdarah.",
        "precautions": ["Minum jus daun pepaya", "Hindari makanan berlemak dan pedas", "Jauhkan diri dari nyamuk", "Jaga tubuh tetap terhidrasi"],
    },
    "hepatitis d": {
        "description": "Hepatitis D (virus hepatitis delta) adalah infeksi yang menyebabkan peradangan hati, dapat mengganggu fungsi hati dan menyebabkan masalah jangka panjang termasuk jaringan parut dan kanker hati.",
        "precautions": ["Konsultasi ke dokter", "Konsumsi obat sesuai anjuran", "Konsumsi makanan sehat", "Lakukan kontrol lanjutan ke dokter"],
    },
    "heart attack": {
        "description": "Kematian jaringan otot jantung akibat hilangnya suplai darah, biasanya disebabkan oleh penyumbatan total pada salah satu arteri koroner yang memasok darah ke otot jantung.",
        "precautions": ["Segera hubungi ambulans", "Kunyah atau telan aspirin (sesuai anjuran medis)", "Tetap tenang"],
    },
    "pneumonia": {
        "description": "Pneumonia adalah infeksi pada salah satu atau kedua paru-paru akibat bakteri, virus, atau jamur, menyebabkan peradangan pada kantung udara paru-paru yang terisi cairan/nanah sehingga menyulitkan pernapasan.",
        "precautions": ["Konsultasi ke dokter", "Konsumsi obat sesuai anjuran", "Istirahat yang cukup", "Lakukan kontrol lanjutan ke dokter"],
    },
    "arthritis": {
        "description": "Artritis adalah pembengkakan dan nyeri tekan pada satu atau lebih sendi. Gejala utamanya nyeri sendi dan kekakuan yang biasanya memburuk seiring usia. Jenis paling umum: osteoartritis dan artritis reumatoid.",
        "precautions": ["Olahraga teratur", "Gunakan terapi panas dan dingin", "Coba terapi akupunktur", "Lakukan pijat"],
    },
    "gastroenteritis": {
        "description": "Gastroenteritis adalah peradangan pada saluran pencernaan, khususnya lambung serta usus besar dan kecil, berkaitan dengan gejala diare, kram perut, mual, dan muntah.",
        "precautions": ["Hentikan makanan padat sementara waktu", "Minum air sedikit demi sedikit", "Istirahat yang cukup", "Kembali makan secara bertahap"],
    },
    "tuberculosis": {
        "description": "Tuberkulosis (TB) adalah penyakit menular yang umumnya disebabkan bakteri Mycobacterium tuberculosis, biasanya menyerang paru-paru namun bisa memengaruhi bagian tubuh lain. Sebagian besar infeksi tidak bergejala (TB laten).",
        "precautions": ["Tutup mulut saat batuk/bersin", "Konsultasi ke dokter", "Konsumsi obat sesuai anjuran", "Istirahat yang cukup"],
    },
}


def translate_disease_content(disease_name, info_clean):
    """Timpa Description & Precaution_x di info_clean dengan versi Bahasa Indonesia, kalau ada di kamus."""
    key = str(disease_name).strip().lower()
    if key not in DISEASE_ID:
        return info_clean
    translated = DISEASE_ID[key]
    if "Description" in info_clean:
        info_clean["Description"] = translated["description"]
    precautions = translated["precautions"]
    for i in range(4):
        col = f"Precaution_{i+1}"
        if col in info_clean:
            info_clean[col] = precautions[i] if i < len(precautions) else None
    return info_clean


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
    result = []
    for s in sorted(symptom_columns):
        label, category = get_symptom_info(s)
        result.append({"value": s, "label": label, "category": category})
    return jsonify(result)


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

    # terjemahkan nama spesialis ke Bahasa Indonesia
    if "Specialist" in info_clean:
        info_clean["Specialist"] = translate_specialist(info_clean["Specialist"])

    # terjemahkan deskripsi & saran pencegahan ke Bahasa Indonesia (kalau penyakitnya ada di kamus)
    info_clean = translate_disease_content(pred_disease, info_clean)

    profil = data.get("profil", {})  # umur/BB/TB/tekanan darah/durasi -> hanya diteruskan balik, TIDAK dipakai model

    return jsonify({
        "prediksi_utama": pred_disease,
        "top3_kemungkinan": top3,
        "info_tambahan": info_clean,
        "profil_user": profil
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)