# Crop Recommendation System

Sistem rekomendasi tanaman berbasis machine learning yang memprediksi jenis tanaman terbaik berdasarkan kondisi lingkungan dan tanah. Dataset mencakup 2.200 sampel dengan 22 jenis tanaman dan 23 fitur agronomis.

## Struktur Direktori

```
Project/
├── data/
│   ├── raw/
│   │   └── crop-recommendation.csv      # Dataset mentah (2200 baris, 23 kolom)
│   └── processed/
│       ├── train_data.csv               # Data latih (1760 baris, 26 fitur + label)
│       └── test_data.csv                # Data uji (440 baris, 26 fitur + label)
├── models/
│   ├── label_encoder.pkl                # Encoder label tanaman (22 kelas)
│   ├── scaler.pkl                       # StandardScaler untuk normalisasi fitur
│   └── random_forest_best.pkl           # Model Random Forest terbaik
├── notebooks/
│   ├── eda.ipynb                        # Exploratory Data Analysis
│   ├── preprocessing.ipynb              # Preprocessing & pembuatan data train/test
│   └── experiments/
│       └── 01_random_forest.ipynb       # Pelatihan & evaluasi model Random Forest
└── README.md
```

## Dataset

Dataset `data/raw/crop-recommendation.csv` berisi 2.200 baris dengan fitur-fitur berikut:

| Fitur                    | Deskripsi                                  |
| ------------------------ | ------------------------------------------ |
| `N`, `P`, `K`            | Kadar Nitrogen, Fosfor, Kalium dalam tanah |
| `temperature`            | Suhu udara (°C)                            |
| `humidity`               | Kelembapan udara (%)                       |
| `ph`                     | Tingkat keasaman tanah                     |
| `rainfall`               | Curah hujan (mm)                           |
| `soil_moisture`          | Kelembapan tanah                           |
| `soil_type`              | Jenis tanah (kategorikal)                  |
| `sunlight_exposure`      | Paparan sinar matahari (jam)               |
| `wind_speed`             | Kecepatan angin                            |
| `co2_concentration`      | Konsentrasi CO2                            |
| `organic_matter`         | Kandungan bahan organik                    |
| `irrigation_frequency`   | Frekuensi irigasi                          |
| `crop_density`           | Kepadatan tanaman                          |
| `pest_pressure`          | Tekanan hama                               |
| `fertilizer_usage`       | Penggunaan pupuk                           |
| `growth_stage`           | Tahap pertumbuhan                          |
| `urban_area_proximity`   | Jarak ke area urban                        |
| `water_source_type`      | Jenis sumber air                           |
| `frost_risk`             | Risiko embun beku                          |
| `water_usage_efficiency` | Efisiensi penggunaan air                   |
| `label`                  | Jenis tanaman (target, 22 kelas)           |

**22 Kelas Tanaman:** apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, watermelon

## Instalasi Library

Pastikan Python 3.8+ sudah terinstal, lalu install dependensi berikut:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib tqdm
```

**Daftar library yang digunakan:**

| Library        | Versi Minimum | Kegunaan                          |
| -------------- | ------------- | --------------------------------- |
| `pandas`       | 1.3+          | Manipulasi dan analisis data      |
| `numpy`        | 1.21+         | Operasi numerik dan array         |
| `scikit-learn` | 1.0+          | Preprocessing, model ML, evaluasi |
| `matplotlib`   | 3.4+          | Visualisasi data dan grafik       |
| `seaborn`      | 0.11+         | Visualisasi statistik             |
| `joblib`       | 1.0+          | Menyimpan dan memuat model (.pkl) |
| `tqdm`         | 4.60+         | Progress bar saat training        |

## Alur Kerja

```
data/raw/ → [preprocessing.ipynb] → data/processed/ → [01_random_forest.ipynb] → models/
```

## Cara Menjalankan

### 1. Exploratory Data Analysis (Opsional)

Jalankan notebook EDA untuk memahami distribusi data, korelasi fitur, dan separabilitas kelas:

```bash
jupyter notebook notebooks/eda.ipynb
```

### 2. Preprocessing — Menghasilkan Data Train & Test

Langkah ini **wajib dijalankan terlebih dahulu** sebelum melatih model. Notebook ini akan membaca data mentah, melakukan feature engineering, menangani outlier, encoding, scaling, lalu menyimpan hasilnya.

```bash
jupyter notebook notebooks/preprocessing.ipynb
```

**Jalankan semua sel dari atas ke bawah (Run All).** Setelah selesai, file-file berikut akan terbuat:

```
data/processed/train_data.csv    ← data latih (1760 sampel, 26 fitur)
data/processed/test_data.csv     ← data uji  (440 sampel,  26 fitur)
models/label_encoder.pkl         ← encoder label tanaman
models/scaler.pkl                ← scaler untuk normalisasi
```

**Tahapan preprocessing yang dilakukan:**

1. **Feature Engineering** — membuat 4 fitur turunan:
   - `N_P_ratio` = N / (P + 1e-5)
   - `P_K_ratio` = P / (K + 1e-5)
   - `N_K_ratio` = N / (K + 1e-5)
   - `temp_humidity_idx` = temperature × humidity
2. **Outlier Handling** — clipping dengan metode IQR (batas 1.5×IQR)
3. **Label Encoding** — mengubah label tanaman (string) menjadi integer (0–21)
4. **Train-Test Split** — rasio 80:20 dengan stratifikasi (random_state=42)
5. **Feature Scaling** — StandardScaler di-fit pada data train, lalu di-transform ke train & test

### 3. Melatih Model Random Forest

Setelah data processed tersedia, jalankan notebook eksperimen:

```bash
jupyter notebook notebooks/experiments/01_random_forest.ipynb
```

Notebook ini menjalankan:

- **Baseline model** — Random Forest dengan 100 estimator
- **Hyperparameter tuning** — RandomizedSearchCV (50 iterasi, 5-fold CV)
- **Evaluasi** — accuracy, classification report, confusion matrix, feature importance
- **Simpan model** — model terbaik disimpan ke `models/random_forest_best.pkl`

## Hasil Model

| Metrik                               | Nilai          |
| ------------------------------------ | -------------- |
| Baseline Accuracy (test set)         | 99.32%         |
| Cross-Validation Accuracy (5-fold)   | 99.26% ± 0.58% |
| Best Model Accuracy (setelah tuning) | 99.32%         |
| Best CV Accuracy (setelah tuning)    | 99.55%         |

**Hyperparameter terbaik:**

| Parameter           | Nilai |
| ------------------- | ----- |
| `n_estimators`      | 500   |
| `max_depth`         | 50    |
| `max_features`      | sqrt  |
| `min_samples_split` | 5     |
| `min_samples_leaf`  | 2     |
| `bootstrap`         | False |
