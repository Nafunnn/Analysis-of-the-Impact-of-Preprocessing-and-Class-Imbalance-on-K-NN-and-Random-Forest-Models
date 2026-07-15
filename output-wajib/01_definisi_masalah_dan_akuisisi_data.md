# Problem Definition and Data Acquisition

**Judul Proyek:** Eksperimen Klasifikasi Depresi pada Remaja: Perbandingan Metode Feature Selection untuk Identifikasi Fitur Gaya Hidup Paling Berpengaruh  

**Mahasiswa:** Naf’an Nur’Alim (A11.2024.15651)

---

## 1. Problem Statement

Di era digital, remaja menghabiskan sebagian besar waktu untuk berinteraksi melalui platform media sosial. Intensitas penggunaan yang berlebihan, paparan layar sebelum tidur, dan kurangnya aktivitas fisik berkorelasi dengan menurunnya kualitas kesehatan mental, termasuk stres, kecemasan, dan depresi. Bukti epidemiologis menunjukkan bahwa remaja dengan screen time tinggi lebih berisiko mengalami gejala depresi, gangguan tidur, dan aktivitas fisik yang rendah. Gejala depresi pada remaja pun tidak selalu diekspresikan secara emosional di media sosial, melainkan sering muncul lewat sinyal somatik seperti gangguan tidur, penurunan aktivitas fisik, dan pola penggunaan layar yang tidak sehat.

Permasalahan inti penelitian ini adalah bahwa **indikasi depresi tidak ditentukan oleh satu atribut secara terisolasi**, melainkan oleh kombinasi berbagai fitur gaya hidup dan perilaku digital. Misalnya, dua remaja dengan durasi media sosial enam jam per hari belum tentu memiliki risiko depresi yang sama, karena faktor pendukung seperti kualitas tidur, aktivitas fisik, tingkat stres, dan interaksi sosial dapat saling memoderasi. Analisis korelasi tunggal antar atribut dengan label depresi hanya bersifat eksplorasi statistik (EDA) dan belum cukup untuk menangkap hubungan multi-fitur yang kompleks serta non-linear.

Penelitian terdahulu telah membuktikan bahwa Machine Learning mampu menganalisis pola perilaku digital dan psikososial untuk memprediksi risiko depresi remaja. Prediktor yang sering muncul mencakup jam tidur, tingkat stres, tingkat kecemasan, dan durasi media sosial harian. Namun, masih jarang penelitian yang secara eksplisit **membandingkan metode feature selection** untuk mengidentifikasi subset fitur gaya hidup paling berpengaruh terhadap label depresi. Tanpa seleksi fitur yang tepat, model dapat terganggu oleh noise, redundansi (misalnya antara stres dan kecemasan), serta perbedaan skala antar atribut.

Oleh karena itu, proyek ini merancang eksperimen **supervised learning (klasifikasi biner)** dengan fokus utama pada perbandingan metode feature selection — PCA, Chi-Square, Mutual Information, serta kondisi tanpa seleksi (baseline) — menggunakan algoritma klasifikasi tetap (Random Forest). Pendekatan Explainable AI (SHAP) digunakan sebagai pelengkap untuk memverifikasi kontribusi tiap fitur terhadap prediksi. Tujuan praktisnya adalah menemukan fitur gaya hidup dan digital mana yang paling memengaruhi indikasi depresi remaja, sekaligus menilai metode seleksi fitur mana yang paling efektif pada dataset ini.

**Learning task:** Supervised Learning — Klasifikasi biner  
**Target variable:** `depression_label` (0 = tidak terindikasi depresi, 1 = terindikasi depresi)  
**Fokus eksperimen:** Perbandingan metode feature selection (bukan perbandingan algoritma klasifikasi)

---

## 2. Sumber Dataset

| Aspek | Keterangan |
| --- | --- |
| **Nama dataset** | Teen Mental Health Dataset |
| **Penyedia** | Kaggle (publik) |
| **Deskripsi singkat** | Kompilasi survei terkait kesehatan mental dan penggunaan media sosial pada remaja |
| **Link dataset** | [https://www.kaggle.com/datasets/algozee/teenager-menthal-healy](https://www.kaggle.com/datasets/algozee/teenager-menthal-healy) |
| **File lokal proyek** | `Teen_Mental_Health_Dataset.csv` (root) dan `eksperimen-klasifikasi-depresi/data/Teen_Mental_Health_Dataset.csv` |
| **Format** | CSV (tabular) |
| **Catatan augmentasi** | Dataset asli dilaporkan memiliki 1.200 observasi dengan imbalance ekstrem (2,6% kelas positif). Data penelitian ini telah diaugmentasi menjadi **1.500 observasi** dengan rasio kelas ≈ **90 : 10** agar eksperimen klasifikasi lebih representatif. |

**Dokumentasi terkait di proyek:**
- Proposal: `Proposal UTS Pembelajaran Mesin.md` (bagian Dataset & Representasi Data)
- Laporan eksperimen: `eksperimen-klasifikasi-depresi/reports/Laporan_Hasil_Eksperimen.md`
- EDA notebook: `eksperimen-klasifikasi-depresi/notebooks/01_eda.ipynb`

---

## 3. Statistik Deskriptif Awal

### 3.1 Ringkasan ukuran data

| Metrik | Nilai |
| --- | --- |
| Jumlah observasi (baris) | **1.500** |
| Jumlah kolom | **13** (12 fitur input + 1 target) |
| Missing values | **0** (semua kolom non-null) |
| Ukuran memori kasar | ~391 KB |
| Class imbalance | Ya — kelas mayoritas 90%, kelas minoritas 10% |

### 3.2 Distribusi target (`depression_label`)

| Label | Arti | Jumlah | Persentase |
| --- | --- | ---: | ---: |
| 0 | Tidak terindikasi depresi | 1.350 | 90,0% |
| 1 | Terindikasi depresi | 150 | 10,0% |

### 3.3 Tipe data per kolom

| Nama Kolom | Tipe Data | Deskripsi | Peran |
| --- | --- | --- | --- |
| `age` | Numerik (int) | Usia responden remaja | Fitur |
| `gender` | Kategorikal (object) | Jenis kelamin (`male` / `female`) | Fitur |
| `daily_social_media_hours` | Numerik (float) | Durasi harian media sosial (jam) | Fitur |
| `platform_usage` | Kategorikal (object) | Platform utama (`TikTok` / `Instagram` / `Both`) | Fitur |
| `sleep_hours` | Numerik (float) | Rata-rata jam tidur per hari | Fitur |
| `screen_time_before_sleep` | Numerik (float) | Durasi layar sebelum tidur (jam) | Fitur |
| `academic_performance` | Numerik (float) | Skor / performa akademik | Fitur |
| `physical_activity` | Numerik (float) | Durasi aktivitas fisik | Fitur |
| `social_interaction_level` | Kategorikal (object) | Interaksi sosial nyata (`low` / `medium` / `high`) | Fitur |
| `stress_level` | Numerik (int, ordinal) | Tingkat stres (skala 1–10) | Fitur |
| `anxiety_level` | Numerik (int, ordinal) | Tingkat kecemasan (skala 1–10) | Fitur |
| `addiction_level` | Numerik (int, ordinal) | Tingkat kecanduan digital (skala 1–10) | Fitur |
| `depression_label` | Numerik (int, biner) | Label depresi (0 / 1) | **Target** |

**Ringkasan tipe:** 10 kolom numerik + 3 kolom kategorikal.

### 3.4 Distribusi fitur kategorikal

| Fitur | Distribusi |
| --- | --- |
| `gender` | male: 759 · female: 741 |
| `platform_usage` | TikTok: 509 · Instagram: 497 · Both: 494 |
| `social_interaction_level` | medium: 525 · low: 522 · high: 453 |

### 3.5 Statistik deskriptif fitur numerik

| Fitur | Mean | Std | Min | 25% | Median | 75% | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `age` | 15,90 | 2,02 | 13,0 | 14,0 | 16,0 | 18,0 | 19,0 |
| `daily_social_media_hours` | 4,71 | 2,04 | 1,0 | 3,0 | 4,8 | 6,5 | 8,0 |
| `sleep_hours` | 6,31 | 1,46 | 4,0 | 5,0 | 6,3 | 7,5 | 9,0 |
| `screen_time_before_sleep` | 1,74 | 0,73 | 0,5 | 1,1 | 1,7 | 2,4 | 3,0 |
| `academic_performance` | 2,99 | 0,58 | 2,0 | 2,5 | 3,0 | 3,49 | 4,0 |
| `physical_activity` | 1,00 | 0,58 | 0,0 | 0,5 | 1,0 | 1,5 | 2,0 |
| `stress_level` | 5,70 | 2,94 | 1,0 | 3,0 | 6,0 | 8,0 | 10,0 |
| `anxiety_level` | 5,83 | 2,87 | 1,0 | 3,0 | 6,0 | 8,0 | 10,0 |
| `addiction_level` | 5,53 | 2,90 | 1,0 | 3,0 | 6,0 | 8,0 | 10,0 |

*Sumber tabel: `eksperimen-klasifikasi-depresi/results/tables/01_descriptive_statistics.csv` dan `05_class_distribution.csv`.*

### 3.6 Implikasi awal untuk pipeline ML

1. **Class imbalance** signifikan → evaluasi sebaiknya menekankan F1-Score, Recall, dan ROC-AUC, bukan accuracy saja; class weight / balancing perlu dipertimbangkan.
2. **Heterogenitas tipe data** (numerik kontinu, ordinal, dan kategorikal) → encoding (label/ordinal/one-hot) dan scaling diperlukan sebelum PCA serta sebagian metode filter.
3. **Tidak ada missing value**, tetapi outlier tetap perlu dicek (misalnya via IQR) agar seleksi fitur dan klasifikasi tidak terpengaruh nilai ekstrem.
4. Setelah preprocessing & encoding, jumlah fitur efektif bertambah (misalnya one-hot `platform_usage` + fitur engineered `screen_time_ratio`), sehingga perbandingan metode feature selection menjadi relevan.

---

*Dokumen ini disusun sesuai kondisi proyek eksperimen klasifikasi depresi remaja pada repository Tugas Akhir Pembelajaran Mesin.*
