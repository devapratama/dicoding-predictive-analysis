# Laporan Proyek Machine Learning - Ahmad Sabil Deva Pratama

## 1. Domain Proyek

### Latar Belakang
Kualitas air merupakan aspek penting dalam kehidupan manusia, khususnya terkait dengan kesehatan. Air yang tercemar atau tidak layak konsumsi dapat menimbulkan berbagai penyakit. Oleh karena itu, penting untuk memantau kualitas air secara terus-menerus. Pemantauan kualitas air secara manual memerlukan banyak waktu dan biaya. Untuk itu, pendekatan otomatis menggunakan **machine learning (ML)** dapat menjadi solusi yang efisien untuk memprediksi potabilitas air berdasarkan data fisikokimia.

**Masalah yang Dihadapi:**  
Bagaimana cara memprediksi potabilitas air (layak atau tidak layak konsumsi) berdasarkan karakteristik fisikokimia air?

**Referensi:**
- World Health Organization. *Drinking-water*. [WHO Report](https://www.who.int/news-room/fact-sheets/detail/drinking-water)  
- Badan Pusat Statistik. *Akses Rumah Tangga terhadap Air Minum*. BPS 2022. [BPS 2023](https://www.bps.go.id/id/statistics-table/2/ODU0IzI=/persentase-rumah-tangga-menurut-provinsi-tipe-daerah-dan-sumber-air-minum-layak.html)

---

## 2. Business Understanding

### Problem Statements
1. Bagaimana cara menentukan apakah air layak konsumsi berdasarkan berbagai parameter fisikokimia?
2. Bagaimana memprediksi kualitas air dengan akurasi yang tinggi untuk memudahkan pemantauan kualitas air?

### Goals
1. Membangun model prediktif menggunakan machine learning untuk menentukan apakah air layak konsumsi atau tidak.
2. Mengidentifikasi fitur-fitur yang paling berpengaruh terhadap keputusan potabilitas air.

### Solution Statements
1. **Penggunaan Algoritma Random Forest, XGBoost, dan SVM** dapat digunakan untuk memprediksi potabilitas air. Ketiga algoritma ini cocok untuk dataset dengan banyak fitur dan hubungan non-linear.
2. **Hyperparameter tuning** akan dilakukan pada model Random Forest untuk meningkatkan akurasi model dan mengurangi overfitting.
3. **Pemilihan Model Terbaik**: Algoritma terbaik akan dipilih berdasarkan evaluasi performa menggunakan metrik yang relevan.

---

## 3. Data Understanding

### Informasi Data
Dataset yang digunakan adalah dataset kualitas air yang berisi berbagai parameter fisikokimia dari sampel air. Dataset ini dapat diunduh melalui [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Water+Quality).

#### Variabel pada Dataset:
Dataset ini memiliki 10 kolom (fitur), yang menjelaskan berbagai aspek kualitas air. Berikut adalah penjelasan untuk setiap variabel:

| **Variabel**         | **Deskripsi**                                                                                                    |
|----------------------|------------------------------------------------------------------------------------------------------------------|
| **`pH`**             | Tingkat keasaman atau kebasaan air, dengan skala 0-14. pH yang lebih rendah menunjukkan air yang lebih asam, sementara pH yang lebih tinggi menunjukkan air yang lebih basa. |
| **`Hardness`**       | Kekerasan air, mengukur kemampuan air untuk mengendapkan sabun. Dinyatakan dalam miligram per liter (mg/L), air dengan kekerasan tinggi bisa membentuk kerak pada pipa dan peralatan. |
| **`Solids`**         | Jumlah total padatan terlarut dalam air, dinyatakan dalam bagian per juta (ppm). Ini mencakup bahan-bahan yang terlarut dalam air seperti garam mineral, kotoran, dan bahan organik. |
| **`Chloramines`**    | Jumlah kloramin yang digunakan dalam pengolahan air sebagai disinfektan. Dinyatakan dalam ppm (bagian per juta), kloramin digunakan untuk membunuh mikroorganisme dalam air. |
| **`Sulfate`**        | Jumlah sulfat terlarut dalam air, dinyatakan dalam miligram per liter (mg/L). Sulfat adalah senyawa yang dapat mempengaruhi rasa dan kualitas air. |
| **`Conductivity`**   | Konduktivitas listrik air yang menunjukkan jumlah ion terlarut dalam air. Dinyatakan dalam mikrosiemens per sentimeter (μS/cm), semakin tinggi konduktivitas, semakin banyak ion terlarut dalam air. |
| **`Organic_carbon`** | Jumlah karbon organik terlarut dalam air, dinyatakan dalam ppm. Karbon organik dapat berasal dari sumber alami atau kontaminasi, dan mempengaruhi kualitas air. |
| **`Trihalomethanes`**| Jumlah trihalometana, produk sampingan yang terbentuk selama proses disinfeksi air. Dinyatakan dalam mikrogram per liter (μg/L), senyawa ini dapat berpotensi berbahaya bagi kesehatan. |
| **`Turbidity`**      | Tingkat kekeruhan air, yang mengukur seberapa jernih air. Dinyatakan dalam NTU (Nephelometric Turbidity Units), nilai yang tinggi menunjukkan adanya partikel terlarut atau padatan dalam air. |
| **`Potability`**     | Label biner yang menunjukkan apakah air dapat dikonsumsi atau tidak. `1`: Air layak konsumsi, `0`: Air tidak layak konsumsi  |

### **Exploratory Data Analysis (EDA)**  

Exploratory Data Analysis (EDA) bertujuan untuk memahami pola data, distribusi nilai, hubungan antar variabel, serta anomali yang ada. Berikut adalah langkah-langkah EDA yang telah dilakukan pada dataset kualitas air.

#### **1. Memuat Dataset**

Dataset yang digunakan memiliki 3276 baris dan 10 kolom, dengan kolom terakhir adalah `Potability`. Kolom ini digunakan sebagai target untuk model klasifikasi. Berikut adalah beberapa baris pertama dataset:

| pH   | Hardness | Solids | Chloramines | Sulfate | Conductivity | Organic_carbon | Trihalomethanes | Turbidity | Potability |
|------|----------|--------|-------------|---------|--------------|----------------|-----------------|-----------|------------|
| 7.1  | 116.0    | 356.0  | 6.1         | 196.0   | 290.0        | 2.15           | 41.5            | 1.2       | 1          |
| 6.8  | 134.0    | 312.0  | 4.5         | 212.0   | 284.0        | 1.91           | 38.0            | 0.9       | 0          |
| 7.4  | 124.0    | 375.0  | 5.8         | 198.0   | 289.0        | 2.02           | 42.0            | 1.5       | 1          |

#### **2. Analisis Missing Values**

Berdasarkan pemeriksaan, kolom yang memiliki nilai hilang adalah sebagai berikut:

| Kolom          | Jumlah Missing | Persentase Missing |
|----------------|----------------|--------------------|
| `ph`           | 28             | 0.86%              |
| `Sulfate`      | 782            | 23.84%             |
| `Trihalomethanes` |  169         | 5.16%              |

Kolom `Sulfate` memiliki persentase missing yang cukup tinggi, yaitu sekitar 23.84%. Missing values pada kolom ini perlu ditangani, mungkin dengan imputasi atau teknik lainnya seperti penghapusan baris dengan nilai hilang.

#### **3. Analisis Distribusi Kategori `Potability`**

Distribusi kelas `Potability` menunjukkan ketidakseimbangan yang cukup signifikan, dengan sekitar 61% data memiliki label `0` (tidak layak konsumsi) dan 39% lainnya memiliki label `1` (layak konsumsi). Ketidakseimbangan ini perlu diperhatikan, misalnya dengan menggunakan teknik *oversampling* atau *undersampling* pada saat pemodelan.

---

## 4. Data Preparation

Tahapan *Data Preparation* mempersiapkan dataset agar siap digunakan dalam pemodelan machine learning. Proses ini mencakup penanganan missing values, transformasi fitur, normalisasi, dan pembagian data untuk memastikan model dapat belajar secara efektif dan memberikan prediksi yang akurat.

#### **1. Penanganan Missing Values**

Beberapa fitur dalam dataset memiliki missing values. Untuk mengatasi hal ini, kita menggunakan **KNN Imputer**, yang mengisi nilai yang hilang dengan rata-rata dari `k` tetangga terdekat. Setelah imputasi, kita memverifikasi bahwa tidak ada lagi missing values.

```python
# Imputasi missing values
imputer = KNNImputer(n_neighbors=5)
data.iloc[:, :-1] = imputer.fit_transform(data.iloc[:, :-1])
```

Dengan imputasi ini, kita memastikan dataset siap untuk analisis lebih lanjut tanpa masalah missing values.

#### **2. Penanganan Skewness**

Beberapa fitur, seperti **`Solids`** dan **`Trihalomethanes`**, menunjukkan distribusi yang skewed. Untuk mengatasi hal ini, kita menerapkan **transformasi log** untuk mengurangi skewness dan mendekatkan distribusi ke bentuk normal, yang membantu model untuk belajar lebih baik.

```python
# Transformasi log untuk fitur skewed
skewed_features = ['Solids', 'Trihalomethanes']
data[skewed_features] = data[skewed_features].apply(lambda x: np.log1p(x))
```

Transformasi ini membuat distribusi data lebih seimbang, meminimalkan pengaruh nilai ekstrem pada model.

#### **3. Normalisasi Fitur**

Untuk memastikan setiap fitur berada pada skala yang seragam, kita melakukan **normalisasi** menggunakan **StandardScaler**. Hal ini penting karena algoritma seperti SVM peka terhadap skala fitur. Normalisasi memastikan model tidak terdistorsi oleh perbedaan skala antar fitur.

```python
# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
```

Setelah normalisasi, setiap fitur memiliki rata-rata 0 dan standar deviasi 1, memastikan keseragaman skala.

#### **4. Menangani Ketidakseimbangan Data**

Dataset memiliki ketidakseimbangan kelas yang signifikan, di mana kelas **`0`** (air tidak layak) jauh lebih banyak daripada kelas **`1`** (air layak). Untuk mengatasi ini, kita menggunakan **SMOTE (Synthetic Minority Over-sampling Technique) Oversampling** untuk menambah jumlah sampel pada kelas minoritas, sehingga kelas menjadi lebih seimbang.

```python
# Menangani ketidakseimbangan kelas dengan SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
```

SMOTE oversampling meningkatkan jumlah sampel kelas minoritas, membantu model untuk belajar lebih efektif dan mengurangi bias terhadap kelas mayoritas.

#### **5. Pembagian Data**

Data dibagi menjadi **80% untuk pelatihan** dan **20% untuk pengujian**. Pembagian ini memastikan model diuji pada data yang belum dilihat sebelumnya.

```python
# Membagi data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
```

Dengan pembagian ini, kita siap melatih model dan menguji kemampuannya untuk memprediksi kualitas air pada data yang tidak terlihat sebelumnya.

---

## 5. Modeling

### Algoritma yang Digunakan
1. **Random Forest:** Algoritma ensemble yang membangun banyak pohon keputusan untuk meningkatkan performa prediksi dan menghindari overfitting.
2. **XGBoost:** Algoritma boosting yang berfokus pada kesalahan model sebelumnya, sangat efektif pada dataset dengan dimensi tinggi.
3. **Support Vector Machine (SVM):** Mencari hyperplane optimal untuk memisahkan data antara dua kelas (layak dan tidak layak konsumsi).

### Kelebihan dan Kekurangan
| Algoritma       | Kelebihan                                                   | Kekurangan                                          |
|-----------------|-------------------------------------------------------------|-----------------------------------------------------|
| **Random Forest** | Memiliki kemampuan untuk menangani data yang tidak seimbang | Proses pelatihan yang lebih lama dengan banyak pohon |
| **XGBoost**       | Meningkatkan akurasi melalui boosting, menghindari overfitting | Rentan terhadap overfitting jika parameter tidak disesuaikan |
| **SVM**           | Efektif pada data dengan dimensi tinggi                      | Memerlukan waktu pelatihan lama untuk dataset besar |

### Tahapan Pemodelan
- **Random Forest:** Model ini diterapkan dengan parameter default dan kemudian dilakukan **hyperparameter tuning** untuk menentukan nilai terbaik untuk `n_estimators` dan `max_depth`.
- **XGBoost:** Penggunaan parameter default dan optimasi dengan `eval_metric='logloss'`.
- **SVM:** Diterapkan dengan kernel `rbf` untuk memisahkan data yang tidak linear.

## Evaluation

### Metrik Evaluasi
Evaluasi model dilakukan menggunakan beberapa metrik untuk menilai kinerja model:
- **Precision:** Mengukur seberapa tepat model dalam mengklasifikasikan data positif.
- **Recall:** Mengukur seberapa baik model menangkap seluruh data positif yang sebenarnya.
- **F1-Score:** Kombinasi precision dan recall, sangat berguna untuk dataset yang tidak seimbang.
- **Accuracy:** Proporsi prediksi yang benar dibandingkan dengan seluruh data.

### Hasil Evaluasi
Berikut adalah hasil evaluasi model berdasarkan metrik yang digunakan:

#### Random Forest:
|              | Precision | Recall | F1-Score | Accuracy |
|--------------|----------|--------|----------|----------|
| Class 0      | 0.71     | 0.74   | 0.72     | 0.72     |
| Class 1      | 0.73     | 0.70   | 0.71     |          |
| **Macro Avg**| 0.72     | 0.72   | 0.72     |          |
| **Weighted Avg** | 0.72 | 0.72 | 0.72     |          |

#### XGBoost:
|              | Precision | Recall | F1-Score | Accuracy |
|--------------|----------|--------|----------|----------|
| Class 0      | 0.69     | 0.66   | 0.67     | 0.68     |
| Class 1      | 0.67     | 0.70   | 0.69     |          |
| **Macro Avg**| 0.68     | 0.68   | 0.68     |          |
| **Weighted Avg** | 0.68 | 0.68 | 0.68     |          |

#### SVM:
|              | Precision | Recall | F1-Score | Accuracy |
|--------------|----------|--------|----------|----------|
| Class 0      | 0.65     | 0.66   | 0.66     | 0.65     |
| Class 1      | 0.65     | 0.64   | 0.65     |          |
| **Macro Avg**| 0.65     | 0.65   | 0.65     |          |
| **Weighted Avg** | 0.65 | 0.65 | 0.65     |          |

**Kesimpulan:**  
- **Random Forest** memberikan hasil terbaik di antara ketiga model, dengan **accuracy** mencapai 72%.
- **XGBoost** juga menunjukkan performa yang baik meskipun sedikit lebih rendah daripada Random Forest.
- **SVM** memiliki performa yang lebih rendah, dengan **accuracy** hanya 65%.

### **Hyperparameter Tuning untuk Random Forest**

Pada tahap **Hyperparameter Tuning** untuk model **Random Forest**, ruang pencarian melibatkan tiga parameter utama: **`n_estimators`** (diuji pada 100, 200, dan 300), **`max_depth`** (diuji pada None, 10, dan 20), dan **`min_samples_split`** (diuji pada 2, 5, dan 10). Tuning dilakukan menggunakan **GridSearchCV** dengan **3-fold cross-validation** untuk memilih kombinasi parameter yang memberikan akurasi terbaik.

Setelah melakukan **Hyperparameter Tuning** menggunakan **GridSearchCV**, ditemukan kombinasi parameter optimal untuk model **Random Forest** yang meningkatkan kinerjanya. Parameter terbaik yang diperoleh adalah:
- **`max_depth = 20`**
- **`min_samples_split = 2`**
- **`n_estimators = 300`**

#### **Random Forest Dengan Hyperparameter Tuning:**

|              | Precision | Recall | F1-Score | Accuracy |
|--------------|----------|--------|----------|----------|
| **Class 0**  | 0.72     | 0.74   | 0.73     | 0.73     |
| **Class 1**  | 0.73     | 0.72   | 0.72     |          |
| **Macro Avg**| 0.73     | 0.73   | 0.73     | 0.73     |
| **Weighted Avg** | 0.73  | 0.73   | 0.73     | 0.73     |

#### **Analisis Hasil:**

Setelah hyperparameter tuning, model Random Forest menunjukkan performa seimbang dengan **precision** dan **recall** yang hampir sama (sekitar 0.73), menghasilkan **F1-Score** yang cukup baik (0.73) di kedua kelas. **Akurasi** mencapai 73%, yang mencerminkan kinerja model yang cukup baik dalam klasifikasi kualitas air. Model ini berhasil menangkap mayoritas kasus air tidak layak konsumsi tanpa mengorbankan akurasi keseluruhan.

### Feature Importance
**Feature Importance** pada model Random Forest dengan Hyperparameter Tuning menunjukkan bahwa **pH**, **Sulfate**, dan **Hardness** adalah fitur yang paling berpengaruh dalam menentukan apakah air layak konsumsi.

## Kesimpulan

Proyek ini berhasil mengembangkan model machine learning untuk memprediksi potabilitas air dengan menggunakan **Random Forest**, **XGBoost**, dan **SVM**. Hasil eksperimen menunjukkan bahwa **Random Forest** adalah model yang paling efektif dalam hal akurasi dan keseimbangan antara precision dan recall, dengan **akurasi sebesar 73%** setelah dilakukan **hyperparameter tuning**. 

Model **XGBoost** memberikan performa yang baik, namun sedikit lebih rendah dibandingkan Random Forest, dengan akurasi sekitar **68%**. Sementara itu, **SVM** menunjukkan performa yang lebih rendah, dengan akurasi hanya **65%**.

Dari hasil analisis, model **Random Forest** dengan hyperparameter tuning menunjukkan bahwa fitur-fitur seperti **pH**, **Sulfate**, dan **Hardness** adalah yang paling berpengaruh dalam menentukan apakah air layak konsumsi. Ini menunjukkan bahwa model Random Forest dapat secara efektif memprediksi potabilitas air berdasarkan karakteristik fisikokimia yang relevan.

Secara keseluruhan, penerapan machine learning pada prediksi potabilitas air menunjukkan potensi besar untuk digunakan sebagai alat bantu dalam pemantauan kualitas air secara otomatis dan efisien, yang dapat mendukung upaya untuk menjaga kesehatan masyarakat.
