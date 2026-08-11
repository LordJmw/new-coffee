# Stage 2 — Ekstraksi Fitur & Baseline Random Forest (7 Kelas)

Kode ini adalah penyempurnaan `extract_features.py` lama + skrip training/evaluasi
baru, disesuaikan dengan taksonomi 7 kelas pada `Ringkasan_Arah_Skripsi_Kopi.docx`
(Bagian 3) dan strategi imbalance pada Bagian 4.

## Struktur

```
utils/
  label_mapping.py     # peta 18 folder dataset -> 7 kelas final
  preprocessing.py      # + normalisasi iluminasi (white balance + CLAHE)
  morphology.py          # tidak berubah dari versi lama
  features_shape.py      # geometri + Hu moments + deteksi lubang serangga
  features_color.py      # chromaticity, HSV, Lab (illumination-robust)
  features_texture.py    # GLCM + LBP + edge density
  features_combine.py    # menggabungkan ketiganya jadi satu vektor fitur
extract_features.py     # STEP 1: dataset gambar -> CSV fitur
train_evaluate_rf.py    # STEP 2: CSV fitur -> model RF + evaluasi
```

## Cara pakai

1. Letakkan dataset penuh (Aset A: 979 cacat + 3600 normal) di
   `data/17 kelas cacat + normal/<nama folder subkelas>/*.jpg`, dengan nama folder
   PERSIS seperti kunci di `utils/label_mapping.py` (mis. `full sour bean`,
   `severe insect damage bean`, dst — sudah dicocokkan dengan struktur
   `17_kelas_cacat___normal.zip` yang dilampirkan).

2. Ekstraksi fitur:
   ```bash
   pip install opencv-python numpy pandas tqdm scikit-image
   python extract_features.py
   ```
   Output: `data/features_dataset_7kelas.csv` (setiap baris = 1 biji, kolom
   `subclass_name` = label granular 17+1, `class_name` = label final 7 kelas).

3. Training & evaluasi:
   ```bash
   pip install scikit-learn imbalanced-learn matplotlib seaborn
   python train_evaluate_rf.py --csv data/features_dataset_7kelas.csv
   ```
   Output di folder `outputs/`: `classification_report.txt`,
   `confusion_matrix.png`, `feature_importance.png` + `.csv`.

   Argumen opsional:
   - `--undersample-target 1200` — batas atas jumlah sampel kelas Normal
     setelah undersampling (default 1200, sesuai Bagian 4; uji beberapa
     nilai dan pilih berdasarkan macro-F1 validasi, sesuai catatan skripsi).
   - `--no-smote` — nonaktifkan SMOTE bila hanya ingin
     `class_weight='balanced'` saja.
   - `--n-splits 5` — jumlah fold cross-validation di train set.

## Catatan penting untuk eksperimen domain adaptation (few-shot/self-training)

- `preprocessing.normalize_illumination()` sengaja dipisah jadi fungsi
  tersendiri (`use_white_balance`, `use_clahe` bisa dimatikan satu-satu) agar
  bisa dibandingkan: fitur dari citra ternormalisasi vs mentah, sebagai bagian
  dari analisis seberapa besar domain gap tertutup oleh normalisasi vs oleh
  few-shot fine-tuning itu sendiri.
- `extract_features.py` saat ini ditulis untuk Jalur 2 (baseline RF dari Aset
  A langsung, TANPA compositing). Saat crop hasil Stage 1 (YOLO, foto real
  multi-bean) sudah tersedia, panggil `extract_features_from_image()` /
  `utils.features_combine.extract_all_features()` yang sama pada folder crop
  tersebut untuk menghasilkan CSV few-shot / self-training / test set Stage 2
  — strukturnya modular sehingga tidak perlu menulis ulang ekstraktor fitur.
